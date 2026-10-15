from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp

from dmpe.evaluation.plotting_utils import plot_feature_combinations
from dmpe.utils.density_estimation import DensityEstimate


def shared_density_levels(runs, *, view="achieved", n_levels=21, per_pair=False):

    if not runs or n_levels < 2:
        raise ValueError("Select runs and request at least two levels.")
    if view not in {"achieved", "target", "both"}:
        raise ValueError("view must be 'achieved', 'target', or 'both'.")

    maxima = {}
    for run in runs.values():
        density = run["estimator"]
        if density.dim < 2:
            raise ValueError("Contour plots require at least two features.")

        values = []
        if view in {"achieved", "both"}:
            values.append(np.asarray(density.p_unflattened))

        if view in {"target", "both"}:
            target = run["metadata"]["params"]["alg_params"]["target_distribution"]
            values.append(np.asarray(target).reshape(density.unflattened_shape))

        values = [p / p.sum() for p in values]

        for i, j in combinations(range(density.dim), 2):
            other_axes = tuple(d for d in range(density.dim) if d not in (i, j))

            for p in values:
                marginal = p.sum(axis=other_axes)
                finite = marginal[np.isfinite(marginal)]
                if finite.size == 0:
                    raise ValueError("A density plane has no finite values.")
                maxima[i, j] = max(maxima.get((i, j), 0.0), float(finite.max()))

    def make_levels(vmax):
        return np.linspace(0.0, vmax if vmax > 0 else 1.0, n_levels)

    if per_pair:
        return {pair: make_levels(vmax) for pair, vmax in maxima.items()}

    return make_levels(max(maxima.values()))


def plot_excitation_feature_grid(
    result,
    *,
    view="achieved",
    labels=None,
    fig=None,
    axs=None,
    levels=None,
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
    density = DensityEstimate.from_estimate(
        p=density.p / density.p.sum(),
        n_additional_observations=0,
        density_estimate=density,
    )

    if view == "target":
        density = DensityEstimate.from_estimate(
            p=jnp.asarray(result["metadata"]["params"]["alg_params"]["target_distribution"]),
            n_additional_observations=0,
            density_estimate=density,
        )
    elif view != "achieved":
        raise ValueError("view must be 'samples', 'target', or 'achieved'")

    density_axes = axs[1, 0] if axs is not None and density.dim == 2 else axs
    fig, _ = density.visualize(
        reduction_method=jnp.sum,
        labels=labels,
        fig=fig,
        axs=density_axes,
        levels=levels,
    )
    return fig


def compare_excitation_feature_grids(
    runs,
    *,
    view="achieved",
    labels=None,
    rows_per_figure=12,
    title=None,
    levels=None,
):
    """Compare feature planes with optional shared contour boundaries."""
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
                levels=levels,
            )

            axes[0, column].set_title(objective.replace("dmpe_", "").replace("_", " "))

        for row, (i, j) in enumerate(page):
            axes[row, 0].set_ylabel(str(labels[j]))
            for ax in axes[row]:
                ax.set_xlabel(str(labels[i]))

        fig.suptitle(f"{title or view} | " f"planes {start + 1}-{start + len(page)}")
        yield fig, axes
