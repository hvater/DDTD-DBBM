from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx
import gpjax as gpx
import paramax


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

    def __call__(self, init_obs, actions, tau):
        raise NotImplementedError()


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

    for index, posterior in enumerate(model.posteriors):
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


class GPSequenceModel(eqx.Module):
    pass
