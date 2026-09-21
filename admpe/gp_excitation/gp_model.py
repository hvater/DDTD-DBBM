import jax
import jax.numpy as jnp
import equinox as eqx
import gpjax as gpx
import optax


# TODO Build this with a vmapped structure instead of lists?
# Is there really not an option to have multi-dim GPs in gpjax?


class GPModel(eqx.Module):
    posteriors: list[gpx.gps.AbstractPosterior]
    train_x: jnp.array
    train_y: jnp.array

    @property
    def x_dim(self):
        return self.train_x.shape[-1]

    @property
    def y_dim(self):
        return self.train_y.shape[-1]

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
        posteriors = []
        for _ in range(y_dim):
            D = gpx.Dataset(X=jnp.empty((0, x_dim)), y=jnp.empty((0, 1)))

            prior = cls._make_prior(x_dim)

            posterior = cls._make_posterior(D.n, prior)
            posteriors.append(posterior)

        return cls(posteriors=posteriors, train_x=jnp.empty((0, x_dim)), train_y=jnp.empty((0, y_dim)))

    def predict_posterior_variance(self, x_query: jax.Array) -> jax.Array:

        posterior_variances = []
        for idx, posterior in enumerate(self.posteriors):
            D = gpx.Dataset(X=self.train_x, y=self.train_y[..., idx : idx + 1])
            latent_dist = posterior.predict(x_query, train_data=D)
            posterior_variances.append(latent_dist.variance)
        return jnp.stack(posterior_variances, axis=-1)

    def predict_summed_posterior_variance(self, x_query: jax.Array) -> jax.Array:

        total = 0.0
        for idx, posterior in enumerate(self.posteriors):
            D = gpx.Dataset(X=self.train_x, y=self.train_y[..., idx : idx + 1])
            latent_dist = posterior.predict(x_query, train_data=D)
            total = total + jnp.sum(latent_dist.variance)
        return total


def update_gp_dataset(gp_model, observations, actions):

    inputs = jnp.concatenate([observations[:-1], actions], axis=-1)
    outputs = observations[1:]

    new_posteriors = gp_model.posteriors
    return GPModel(posteriors=new_posteriors, train_x=inputs, train_y=outputs)


def refit_gp_model(gp_model, observations, actions):
    """Updates the hyperparameters. Is this really necessary?"""
    y_dim = observations.shape[-1]
    new_posteriors = []

    inputs = jnp.concatenate([observations[:-1], actions], axis=-1)
    outputs = observations[1:]

    # for j in range(y_dim):
    #     D = gpx.Dataset(X=inputs, y=outputs[..., j : j + 1])
    #     likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
    #     prior = gp_model._make_prior(inputs.shape[-1])
    #     posterior = prior * likelihood
    #     opt_posterior, _ = gpx.fit_scipy(
    #         model=posterior,
    #         objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
    #         train_data=D,
    #         verbose=False,
    #     )
    #     new_posteriors.append(opt_posterior)

    new_posteriors = gp_model.posteriors
    return GPModel(posteriors=new_posteriors, train_x=inputs, train_y=outputs)
