from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp

from dmpe.evaluation.plotting_utils import plot_feature_combinations
from dmpe.utils.density_estimation import DensityEstimate


def plot_excitation_feature_grid(
    result,
    *,
    view="achieved",
    labels=None,
    fig=None,
    axs=None,
):
    """Use the existing full feature grids for one excitation run."""
    if view == "samples":
        data = jnp.concatenate(
            [result["observations"][:-1], result["actions"]],
            axis=-1,
        )
        if labels is None:
            labels = list(range(data.shape[-1]))

        return plot_feature_combinations(data, labels=labels, fig=fig, axs=axs)

    density = result["estimator"]

    if view == "target":
        density = DensityEstimate.from_estimate(
            p=jnp.asarray(result["metadata"]["params"]["alg_params"]["target_distribution"]),
            n_additional_observations=0,
            density_estimate=density,
        )
    elif view != "achieved":
        raise ValueError("view must be 'samples', 'target', or 'achieved'")

    fig, _ = density.visualize(labels=labels, fig=fig, axs=axs)
    return fig


def compare_excitation_feature_grids(
    runs,
    *,
    view="achieved",
    labels=None,
    rows_per_figure=12,
    title=None,
):
    if not runs:
        raise ValueError("No runs selected.")
    if view not in {"samples", "target", "achieved"}:
        raise ValueError("Unknown view.")
    if rows_per_figure < 1:
        raise ValueError("rows_per_figure must be positive.")

    def feature_count(run):
        if view == "samples":
            return run["observations"].shape[-1] + run["actions"].shape[-1]
        return run["estimator"].dim

    n = feature_count(next(iter(runs.values())))
    if n < 2 or any(feature_count(run) != n for run in runs.values()):
        raise ValueError("Runs must have the same feature count, at least two.")

    labels = list(range(n)) if labels is None else labels
    pairs = list(combinations(range(n), 2))

    for start in range(0, len(pairs), rows_per_figure):
        page = pairs[start : start + rows_per_figure]

        fig, axes = plt.subplots(
            len(page),
            len(runs),
            figsize=(3.4 * len(runs), 2.7 * len(page)),
            sharex="row",
            sharey="row",
            squeeze=False,
            layout="constrained",
        )

        for column, (objective, run) in enumerate(runs.items()):
            # Map the existing feature-grid positions onto
            # this figure's comparison rows.
            grid = np.full((n, n), None, dtype=object)
            for row, (i, j) in enumerate(page):
                grid[j, i] = axes[row, column]

            plot_excitation_feature_grid(
                run,
                view=view,
                labels=labels,
                fig=fig,
                axs=grid,
            )

            axes[0, column].set_title(objective.replace("dmpe_", "").replace("_", " "))

        for row, (i, j) in enumerate(page):
            axes[row, 0].set_ylabel(str(labels[j]))
            for ax in axes[row]:
                ax.set_xlabel(str(labels[i]))

        fig.suptitle(f"{title or view} | " f"planes {start + 1}-{start + len(page)}")
        yield fig, axes
