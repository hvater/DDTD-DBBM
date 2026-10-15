from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import jax.numpy as jnp


def visualize_rollout_on_density(
    excitation_result,
    modeling_result,
    test_observations,
    test_actions,
    *,
    max_steps=200,
    labels=None,
    fig=None,
    axs=None,
    levels=None,
):
    observations = jnp.asarray(test_observations)
    actions = jnp.asarray(test_actions)

    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("Provide one trajectory: (time, features) arrays.")
    if len(observations) != len(actions) + 1:
        raise ValueError("Need one more observation than action.")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive or None.")

    actions = actions[:max_steps]
    observations = observations[: len(actions) + 1]
    if len(actions) == 0:
        raise ValueError("The trajectory must contain at least one action.")

    model = modeling_result["model"]

    predictions = model(observations[0], actions, tau=1.0)

    reference = np.asarray(observations)
    predicted = np.asarray(predictions)

    alg_params = excitation_result["metadata"]["params"]["alg_params"]
    if alg_params["consider_action_distribution"]:
        reference = np.concatenate([reference[:-1], np.asarray(actions)], axis=-1)
        predicted = np.concatenate([predicted[:-1], np.asarray(actions)], axis=-1)

    density = excitation_result["estimator"]

    if density.dim < 2 or reference.shape[1] != density.dim:
        raise ValueError("Trajectory features must match a density grid " "of at least 2 dimensions.")

    owns_axes = axs is None

    density_axes = axs[1, 0] if (not owns_axes and density.dim == 2) else axs

    fig, axes = density.visualize(
        labels=labels,
        fig=fig,
        axs=density_axes,
        levels=levels,
    )

    if density.dim == 2:
        planes = [(axes, 0, 1)]
    else:
        planes = [
            (axes[j, i], i, j)
            for i in range(density.dim)
            for j in range(density.dim)
            if i != j and axes[j, i] is not None
        ]

    # Avoid drawing lines across wrapped angle boundaries.
    angle_index = {"pendulum": 0, "cart_pole": 2}.get(excitation_result["metadata"].get("system_name"))

    def curve(values, i, j):
        xy = values[:, [i, j]]
        if angle_index in (i, j):
            jumps = np.flatnonzero(np.abs(np.diff(values[:, angle_index])) > 1.0) + 1
            xy = np.insert(xy, jumps, np.nan, axis=0)
        return xy.T

    for ax, i, j in planes:
        ax.plot(
            *curve(reference, i, j),
            color="white",
            linewidth=1.5,
            label="Test",
            zorder=3,
        )
        ax.plot(
            *curve(predicted, i, j),
            color="tab:red",
            linestyle="--",
            linewidth=1.5,
            label="Model",
            zorder=4,
        )
        for values, color in [(reference, "white"), (predicted, "tab:red")]:
            # Start: dot.
            ax.scatter(
                values[0, i],
                values[0, j],
                marker="o",
                s=45,
                color=color,
                # edgecolors="black",
                linewidths=0.7,
                zorder=5,
            )

            # End: star.
            ax.scatter(
                values[-1, i],
                values[-1, j],
                marker="*",
                s=80,
                color=color,
                # edgecolors="black",
                linewidths=0.7,
                zorder=6,
            )
    if owns_axes:
        handles, _ = planes[0][0].get_legend_handles_labels()
        handles.extend(
            [
                Line2D(
                    [],
                    [],
                    linestyle="None",
                    marker="o",
                    markersize=6,
                    markerfacecolor="white",
                    markeredgecolor="black",
                    label="Start",
                ),
                Line2D(
                    [],
                    [],
                    linestyle="None",
                    marker="*",
                    markersize=11,
                    markerfacecolor="white",
                    markeredgecolor="black",
                    label="End",
                ),
            ]
        )

        fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
        fig.subplots_adjust(top=0.93)
    return fig, axes


def compare_rollouts_on_density(
    runs,
    modeling_results,
    test_observations,
    test_actions,
    *,
    labels=None,
    max_steps=200,
    rows_per_figure=12,
    title=None,
    levels=None,
):
    """Compare rollout feature planes with optional shared contour boundaries."""
    if not runs:
        raise ValueError("No runs selected.")
    if rows_per_figure < 1:
        raise ValueError("rows_per_figure must be positive.")

    missing = set(runs).difference(modeling_results)
    if missing:
        raise ValueError(f"Missing models for objectives: {sorted(missing)}")

    first = next(iter(runs.values()))
    n = first["estimator"].dim
    system = first["metadata"]["system_name"]

    if n < 2 or any(run["estimator"].dim != n or run["metadata"]["system_name"] != system for run in runs.values()):
        raise ValueError("Compare one system with matching density dimensions.")

    labels = list(range(n)) if labels is None else labels
    if len(labels) != n:
        raise ValueError(f"Expected {n} feature labels, received {len(labels)}.")

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
            grid = np.full((n, n), None, dtype=object)
            for row, (i, j) in enumerate(page):
                grid[j, i] = axes[row, column]

            visualize_rollout_on_density(
                run,
                modeling_results[objective],
                test_observations,
                test_actions,
                max_steps=max_steps,
                labels=labels,
                fig=fig,
                axs=grid,
                levels=levels,
            )

            axes[0, column].set_title(objective.replace("dmpe_", "").replace("_", " "))

        for row, (i, j) in enumerate(page):
            for ax in axes[row]:
                ax.set_xlabel(labels[i])
                ax.set_ylabel(labels[j])

        handles, _ = axes[0, 0].get_legend_handles_labels()
        handles.extend(
            [
                Line2D(
                    [],
                    [],
                    linestyle="None",
                    marker=marker,
                    markerfacecolor="white",
                    markeredgecolor="black",
                    label=label,
                )
                for marker, label in (("o", "Start"), ("*", "End"))
            ]
        )

        fig.legend(
            handles=handles,
            loc="outside lower center",
            ncol=4,
            framealpha=1,
        )
        fig.suptitle(f"{title or 'Rollout comparison'} | " f"planes {start + 1}-{start + len(page)}")

        yield fig, axes
