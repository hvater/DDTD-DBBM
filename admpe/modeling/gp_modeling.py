from typing import Callable

import tqdm
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx
import gpjax as gpx
import paramax
from jax_tqdm import scan_tqdm

from dmpe.utils.density_estimation import build_grid


class GPTransitionModel(eqx.Module):
    posteriors: tuple
    train_x: jax.Array
    train_y: jax.Array
    alpha: jax.Array
    featurize: Callable
    angle_indices: tuple[int, ...] = eqx.field(static=True)

    @property
    def N(self):
        assert self.train_x.shape[0] == self.train_y.shape[0]
        return self.train_x.shape[0]

    @property
    def x_dim(self):
        return self.train_x.shape[-1]

    @property
    def y_dim(self):
        return self.train_y.shape[-1]

    def __init__(
        self,
        obs_dim,
        action_dim,
        n_train_points,
        featurize,
        angle_indices=(),
        lengthscale=0.5,
        variance=1.0,
        obs_stddev=0.0,
        *,
        key=None,
    ):
        if n_train_points < 1:
            raise ValueError("n_train_points must be positive.")

        self.featurize = featurize
        self.angle_indices = tuple(angle_indices)

        input_dim = featurize(jnp.zeros(obs_dim)).shape[-1] + action_dim
        posteriors = []

        for _ in range(obs_dim):
            kernel = gpx.kernels.RBF(
                active_dims=list(range(input_dim)),
                lengthscale=jnp.full((input_dim,), lengthscale),
                variance=variance,
            )
            prior = gpx.gps.Prior(
                # Explicitly freeze the mean at zero.
                mean_function=gpx.mean_functions.Constant(constant=paramax.NonTrainable(jnp.asarray(0.0))),
                kernel=kernel,
            )
            likelihood = gpx.likelihoods.Gaussian(
                num_datapoints=n_train_points,
            )
            # Non trainable observation noise
            likelihood = eqx.tree_at(
                lambda l: l.obs_stddev,
                likelihood,
                paramax.NonTrainable(jnp.asarray(obs_stddev)),
            )

            posteriors.append(prior * likelihood)

        self.posteriors = tuple(posteriors)
        self.train_x = jnp.zeros((n_train_points, input_dim))
        self.train_y = jnp.zeros((n_train_points, obs_dim))
        self.alpha = jnp.zeros_like(self.train_y)

    def step(self, obs, action, tau):
        obs = jnp.asarray(obs, dtype=self.alpha.dtype)
        action = jnp.asarray(action, dtype=self.train_x.dtype)

        query = jnp.concatenate([self.featurize(obs), action])[None, :]
        delta = jnp.stack(
            [
                paramax.unwrap(posterior).prior.kernel.cross_covariance(self.train_x, query)[:, 0]
                @ self.alpha[:, index]
                for index, posterior in enumerate(self.posteriors)
            ]
        )
        next_obs = obs + delta

        # wraps angles
        for index in self.angle_indices:
            next_obs = next_obs.at[index].set((next_obs[index] + 1.0) % 2.0 - 1.0)

        return next_obs

    def query_uncertainty(self, obs, action):
        obs = jnp.asarray(obs, dtype=self.alpha.dtype)
        action = jnp.asarray(action, dtype=self.train_x.dtype)
        query = jnp.concatenate([self.featurize(obs), action])[None, :]

        total = 0.0
        for idx, posterior in enumerate(self.posteriors):
            D = gpx.Dataset(X=self.train_x, y=self.train_y[..., idx : idx + 1])
            total += jnp.sum(posterior.predict(query, train_data=D, return_covariance_type="diagonal").variance)
        return total

    def __call__(self, init_obs, actions, tau):
        init_obs = jnp.asarray(init_obs, dtype=self.alpha.dtype)

        def body(obs, action):
            next_obs = self.step(obs, action, tau)
            return next_obs, next_obs

        _, predictions = jax.lax.scan(body, init_obs, actions)
        return jnp.concatenate([init_obs[None, :], predictions], axis=0)


def fit_gp_transitions(
    model,
    observations,
    actions,
    *,
    max_iters=100,
    indices=None,
):
    observations = jnp.asarray(observations, dtype=model.train_x.dtype)
    actions = jnp.asarray(actions, dtype=model.train_x.dtype)

    if observations.shape[0] != actions.shape[0] + 1:
        raise ValueError("Expected one more observation than action.")

    inputs = jnp.concatenate(
        [jax.vmap(model.featurize)(observations[:-1]), actions],
        axis=-1,
    )
    targets = observations[1:] - observations[:-1]

    for index in model.angle_indices:
        targets = targets.at[:, index].set((targets[:, index] + 1.0) % 2.0 - 1.0)

    if indices is not None:
        inputs, targets = inputs[indices], targets[indices]

    if inputs.shape != model.train_x.shape or targets.shape != model.train_y.shape:
        raise ValueError("Training arrays do not match the model template.")

    posteriors, alphas, histories = [], [], []

    for index, posterior in tqdm.tqdm(enumerate(model.posteriors), total=len(model.posteriors)):
        dataset = gpx.Dataset(
            X=inputs,
            y=targets[:, index : index + 1],
        )
        posterior, history = gpx.fit_scipy(
            model=posterior,
            objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
            train_data=dataset,
            max_iters=max_iters,
            verbose=False,
        )

        fitted = paramax.unwrap(posterior)

        covariance = fitted.prior.kernel.gram(inputs).as_matrix()
        covariance = covariance + jnp.eye(inputs.shape[0], dtype=covariance.dtype) * fitted.jitter
        covariance = covariance + jnp.diag(fitted.likelihood.noise_vector(inputs.shape[0]))

        # compute: \alpha = C^{-1} = (\bm{K} + \sigma^2_n @ \bm{I})^{-1}
        cholesky = jnp.linalg.cholesky(covariance)
        alpha = jsp.linalg.cho_solve(
            (cholesky, True),
            targets[:, index],
        )

        if not bool(jnp.all(jnp.isfinite(alpha))):
            raise FloatingPointError("GP conditioning produced nonfinite weights.")

        posteriors.append(posterior)
        alphas.append(alpha)
        histories.append(history)

    fitted_model = eqx.tree_at(
        lambda m: (m.posteriors, m.train_x, m.train_y, m.alpha),
        model,
        (
            tuple(posteriors),
            inputs,
            targets,
            jnp.stack(alphas, axis=-1),
        ),
    )

    return fitted_model, histories


def visualize_transition_uncertainty(
    validation_points_per_dim: int,
    model: GPTransitionModel,
    labels: list[str] | None = None,
    *,
    batch_size: int = 256,
):
    """Plot pairwise means of the summed latent transition variance."""
    if validation_points_per_dim < 2 or batch_size < 1:
        raise ValueError("Use at least 2 points per dimension and a positive batch_size.")

    obs_dim = model.y_dim
    feature_dim = model.featurize(jnp.zeros(obs_dim, dtype=model.train_x.dtype)).shape[-1]
    action_dim = model.x_dim - feature_dim
    n_features = obs_dim + action_dim

    if labels is None:
        labels = [str(i) for i in range(n_features)]
    if len(labels) != n_features:
        raise ValueError("Provide one label per observation and action component.")

    points = build_grid(
        dim=n_features,
        low=-1.0,
        high=1.0,
        points_per_dim=validation_points_per_dim,
    ).astype(model.train_x.dtype)

    # @eqx.filter_jit
    # def evaluate(m, points):
    #     return jax.lax.map(
    #         lambda point: m.query_uncertainty(point[:obs_dim], point[obs_dim:]),
    #         points,
    #         batch_size=batch_size,
    #     )

    @eqx.filter_jit
    def evaluate(m, points):
        n_points = points.shape[0]
        n_batches = (n_points + batch_size - 1) // batch_size

        # Pad the final batch so every batch has the same shape.
        padding = n_batches * batch_size - n_points
        padded = jnp.pad(points, ((0, padding), (0, 0)), mode="edge")
        batches = padded.reshape(n_batches, batch_size, points.shape[-1])

        @scan_tqdm(n_batches, desc="Uncertainty", unit="batch", print_rate=10)
        def body(carry, item):
            _, batch = item
            values = eqx.filter_vmap(m.query_uncertainty)(
                batch[:, :obs_dim],
                batch[:, obs_dim:],
            )
            return carry, values

        _, values = jax.lax.scan(
            body,
            None,
            (jnp.arange(n_batches), batches),
        )
        return values.reshape(-1)[:n_points]

    uncertainty_map = np.asarray(evaluate(model, points)).reshape((validation_points_per_dim,) * n_features)
    vmin = float(uncertainty_map.min())
    vmax = float(uncertainty_map.max())

    fig, axs = plt.subplots(
        n_features,
        n_features,
        figsize=(6, 6),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )

    image_artist = None
    for i in range(n_features):
        for j in range(n_features):
            ax = axs[j, i]
            ax.grid(True)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)

            if i == j:
                continue

            reduction_axes = tuple(k for k in range(n_features) if k not in (i, j))
            image = uncertainty_map.mean(axis=reduction_axes)
            if i < j:
                image = image.T

            image_artist = ax.imshow(
                image,
                origin="lower",
                extent=[-1, 1, -1, 1],
                vmin=vmin,
                vmax=vmax,
            )

        axs[-1, i].set_xlabel(labels[i])
        axs[i, 0].set_ylabel(labels[i])

    if image_artist is not None:
        images = [im for ax in axs.flat for im in ax.images]

        vmin = min(float(im.get_array().min()) for im in images)
        vmax = max(float(im.get_array().max()) for im in images)

        for im in images:
            im.set_clim(vmin, vmax)

        fig.colorbar(
            image_artist,
            ax=axs,
            label="Summed posterior variance",
            shrink=0.8,
        )

    return fig, axs
    return fig, axs


class GPSequenceModel(eqx.Module):
    pass
