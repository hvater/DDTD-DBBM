import jax
import equinox as eqx
import gpjax as gpx


class GPModel(eqx.Module):
    posteriors: list[gpx.gps.AbstractPosterior]
    train_data: list[gpx.Dataset]

    @staticmethod
    def _make_prior(x_dim: int) -> gpx.gps.Prior:
        kernel = gpx.kernels.RBF(active_dims=list(range(x_dim)))
        return gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)

    def __init__(self, x_dim: int, y_dim: int, x0: jax.Array, delta0: jax.Array):
        posteriors, datasets = [], []
        for j in range(y_dim):
            D = gpx.Dataset(X=x0[None, :], y=delta0[j][None, None])
            likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
            posterior = self._make_prior(x_dim) * likelihood
            posteriors.append(posterior)
            datasets.append(D)
        self.posteriors = posteriors
        self.datasets = datasets

    def refit(self):
        raise NotImplementedError()

    def predict_posterior_variance(self, x_query: jax.Array) -> jax.Array:
        raise NotImplementedError()
