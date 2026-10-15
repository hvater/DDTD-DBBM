import jax.numpy as jnp

from dmpe.evaluation.plotting_utils import plot_feature_combinations
from dmpe.utils.density_estimation import DensityEstimate


def plot_excitation_feature_grid(result, *, view="achieved", labels=None):
    """Use the existing full feature grids for one excitation run."""
    if view == "samples":
        data = jnp.concatenate(
            [result["observations"][:-1], result["actions"]],
            axis=-1,
        )
        if labels is None:
            labels = list(range(data.shape[-1]))

        return plot_feature_combinations(data, labels=labels, fig=None, axs=None)

    density = result["estimator"]

    if view == "target":
        density = DensityEstimate.from_estimate(
            p=jnp.asarray(result["metadata"]["params"]["alg_params"]["target_distribution"]),
            n_additional_observations=0,
            density_estimate=density,
        )
    elif view != "achieved":
        raise ValueError("view must be 'samples', 'target', or 'achieved'")

    fig, _ = density.visualize(labels=labels)
    return fig
