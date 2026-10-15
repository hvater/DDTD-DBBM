import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def visualize_rollout_on_density(
    excitation_result,
    modeling_result,
    test_observations,
    test_actions,
    *,
    max_steps=200,
    labels=None,
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

    fig, axes = density.visualize(
        labels=labels,
        # use_contourf=False,
    )
    if density.dim == 2:
        planes = [(axes, 0, 1)]
    else:
        planes = [(axes[j, i], i, j) for i in range(density.dim) for j in range(density.dim) if i != j]

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
