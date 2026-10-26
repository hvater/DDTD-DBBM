from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import jax.numpy as jnp

from dmpe.evaluation.plotting_utils import plot_feature_combinations
from dmpe.utils.density_estimation import DensityEstimate


def visualize_excitation_metric_boxplots(
    df,
    *,
    log_scale: bool = False,
    figure_width: float = 6.0,
    hide_outliers: bool = False,
):
    """Plot diced fill by excitation model and objective for each system.

    Points represent individual excitation runs.
    Counts show n=finite/total, including nonfinite runs in the denominator.
    Returns {system_name: (figure, axes)}.
    """
    metrics = {
        "state_coverage.diced_fill": "State coverage",
        "state_action_coverage.diced_fill": "State-action coverage",
    }
    required = {"system_name", "model_type", "excitation_objective", *metrics}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing dataframe columns: {sorted(missing)}")

    objectives = sorted(df["excitation_objective"].dropna().unique())
    colors = plt.get_cmap("tab10")
    objective_colors = {name: colors(i % colors.N) for i, name in enumerate(objectives)}
    figures = {}

    for system_name, system_df in df.groupby("system_name", sort=True):
        model_types = sorted(system_df["model_type"].dropna().unique())
        system_objectives = [name for name in objectives if name in system_df["excitation_objective"].values]
        if not model_types or not system_objectives:
            continue

        fig, axes = plt.subplots(
            1,
            len(metrics),
            figsize=(figure_width, 3.5),
            squeeze=False,
            constrained_layout=True,
        )
        axes = axes[0]
        box_width = 0.8 / len(system_objectives)

        for ax, (metric, label) in zip(axes, metrics.items()):
            for model_index, model_type in enumerate(model_types):
                for objective_index, objective in enumerate(system_objectives):
                    group = system_df.loc[
                        system_df["model_type"].eq(model_type) & system_df["excitation_objective"].eq(objective),
                        metric,
                    ]
                    values = group.to_numpy(dtype=float, na_value=np.nan)
                    values = values[np.isfinite(values)]

                    if log_scale and np.any(values <= 0):
                        raise ValueError(f"Logarithmic plots require positive {metric} values for {system_name}.")

                    position = model_index + (objective_index - (len(system_objectives) - 1) / 2) * box_width
                    color = objective_colors[objective]

                    if values.size:
                        ax.boxplot(
                            [values],
                            positions=[position],
                            widths=box_width * 0.8,
                            patch_artist=True,
                            manage_ticks=False,
                            boxprops={
                                "facecolor": color,
                                "edgecolor": color,
                                "alpha": 0.5,
                            },
                            whiskerprops={"color": color},
                            capprops={"color": color},
                            medianprops={"color": "black", "linewidth": 1.5},
                            showfliers=not hide_outliers,
                        )

                        if hide_outliers:
                            offsets = (
                                np.linspace(-0.2, 0.2, values.size) * box_width if values.size > 1 else np.zeros(1)
                            )
                            ax.scatter(
                                position + offsets,
                                values,
                                s=18,
                                color=color,
                                edgecolors="white",
                                linewidths=0.4,
                                zorder=3,
                            )

                    ax.text(
                        position,
                        0.98,
                        f"n={values.size}/{len(group)}",
                        transform=ax.get_xaxis_transform(),
                        ha="center",
                        va="top",
                        fontsize=7,
                        color=color,
                    )
            if log_scale:
                ax.set_yscale("log")
            ax.set_title(label)
            ax.set_xlabel("Model used for excitation")
            ax.set_xticks(
                range(len(model_types)),
                [m.replace("_", " ") for m in model_types],
            )
            ax.set_xlim(-0.5, len(model_types) - 0.5)
            ax.margins(y=0.25)
            ax.grid(True, alpha=0.4)
            ax.set_axisbelow(True)

        axes[0].set_ylabel("Diced fill (lower is better)")
        fig.suptitle(system_name.replace("_", " ").title(), fontsize=16)
        fig.legend(
            handles=[
                Patch(
                    facecolor=objective_colors[name],
                    alpha=0.5,
                    label=name.replace("_", " "),
                )
                for name in system_objectives
            ],
            title="Excitation objective",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(2, len(system_objectives)),
            fontsize=8,
            frameon=False,
        )
        figures[system_name] = (fig, axes)

    return figures


def visualize_excitation_history(
    history,
    *,
    figure_width=6.0,
):
    """Return {(system_name, model_type): (figure, axes)}."""
    metrics = {
        "state_coverage.diced_fill": "State coverage",
        "state_action_coverage.diced_fill": "State-action coverage",
    }

    objectives = sorted(history["excitation_objective"].dropna().unique())
    cmap = plt.get_cmap("tab10")
    colors = {name: cmap(i % cmap.N) for i, name in enumerate(objectives)}
    figures = {}

    for (system_name, model_type), system_df in history.groupby(
        ["system_name", "model_type"],
        sort=True,
    ):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(figure_width, 3.5),
            constrained_layout=True,
        )

        for objective, group in system_df.groupby(
            "excitation_objective",
            sort=True,
        ):
            color = colors[objective]
            n_runs = group["exp_id"].nunique()

            final = group.sort_values("n_steps").groupby("exp_id").tail(1)
            n_finite = int(final["validity.all_finite"].sum())
            label = f"{objective.replace('_', ' ')} " f"({n_finite}/{n_runs} finite at end)"

            for ax, (metric, title) in zip(axes, metrics.items()):
                for _, run in group.groupby("exp_id"):
                    run = run.sort_values("n_steps")
                    ax.plot(
                        run["n_steps"],
                        1 - run[metric],
                        color=color,
                        alpha=0.2,
                        lw=0.8,
                    )

                summary = group.groupby("n_steps")[metric].agg(["median", "count"])

                # Require contributions from every run.
                median = summary["median"].where(summary["count"].eq(n_runs))
                ax.plot(
                    summary.index,
                    1 - median,
                    color=color,
                    lw=2,
                    label=label,
                )
                ax.set_title(title)

        for ax in axes:
            ax.set_xlabel("Excitation steps")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.4)

        axes[0].set_ylabel("Covered-cell fraction")
        fig.suptitle(f"{system_name.replace('_', ' ').title()} | " f"{model_type.replace('_', ' ')}")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(2, len(labels)),
            fontsize=8,
            frameon=False,
        )

        figures[system_name, model_type] = (fig, axes)

    return figures


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
