import jax
import jax.numpy as jnp
import equinox as eqx
import gpjax as gpx
import optax


# TODO Build this with a vmapped structure instead of lists?
# Is there really not an option to have multi-dim GPs in gpjax?


class GPModel(eqx.Module):
    posteriors: list[gpx.gps.AbstractPosterior]
    # train_x: jnp.array
    # train_y: jnp.array
    train_data: list[gpx.Dataset]

    @staticmethod
    def _make_prior(x_dim: int) -> gpx.gps.Prior:
        kernel = gpx.kernels.RBF(active_dims=list(range(x_dim)))
        return gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)

    @staticmethod
    def _make_posterior(n_datapoints, prior):
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=max(n_datapoints, 1))
        return prior * likelihood

    @classmethod
    def initialize_empty(cls, x_dim: int, y_dim: int):
        posteriors, train_data = [], []
        for j in range(y_dim):
            D = gpx.Dataset(X=jnp.empty((0, x_dim)), y=jnp.empty((0, 1)))

            prior = cls._make_prior(x_dim)

            posterior = cls._make_posterior(D.n, prior)
            posteriors.append(posterior)
            train_data.append(D)

        return cls(posteriors=posteriors, train_data=train_data)

    def predict_posterior_variance(self, x_query: jax.Array) -> jax.Array:

        total = 0.0
        for posterior, D in zip(self.posteriors, self.train_data):
            latent_dist = posterior.predict(x_query, train_data=D)
            total = total + jnp.sum(latent_dist.variance)
        return total


def update_gp_dataset(gp_model, observations, actions):
    y_dim = observations.shape[-1]
    new_posteriors, new_datasets = [], []

    inputs = jnp.concatenate([observations[:-1], actions], axis=-1)
    outputs = observations[1:]
    for j in range(y_dim):
        D = gpx.Dataset(X=inputs, y=outputs[..., j : j + 1])
        new_datasets.append(D)
    new_posteriors = gp_model.posteriors
    return GPModel(posteriors=new_posteriors, train_data=new_datasets)


def refit_gp_model(gp_model, observations, actions):
    """Updates the hyperparameters. Is this really necessary?"""
    y_dim = observations.shape[-1]
    new_posteriors, new_datasets = [], []

    inputs = jnp.concatenate([observations[:-1], actions], axis=-1)
    outputs = observations[1:]

    for j in range(y_dim):
        D = gpx.Dataset(X=inputs, y=outputs[..., j : j + 1])
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
        prior = gp_model._make_prior(inputs.shape[-1])
        posterior = prior * likelihood
        opt_posterior, _ = gpx.fit_scipy(
            model=posterior,
            objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
            train_data=D,
            verbose=False,
        )
        new_posteriors.append(opt_posterior)
        new_datasets.append(D)
    new_posteriors = gp_model.posteriors
    return GPModel(posteriors=new_posteriors, train_data=new_datasets)
