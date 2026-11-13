import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx

from admpe.evaluation.model_wrapping import EnvWrapper, ModelWrapper
from admpe.evaluation.modeling_evaluation import model_prediction


def visualize_model_metric_boxplots(
    df,
    *,
    log_scale: bool = False,
    figure_width: float = 6.0,
    hide_outliers: bool = False,
):
    """Plot RMSE, MAE and WCE by model and excitation objective for each system.

    Each box contains the finite metric values from individual modeling runs.
    The sample count above each box excludes missing and infinite values.
    Returns a dictionary mapping system names to (figure, axes) pairs.
    """
    metrics = {"metrics.rmse": "RMSE", "metrics.mae": "MAE", "metrics.wce": "WCE"}
    required = {"system_name", "model_type", "excitation_objective", *metrics}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing dataframe columns: {sorted(missing)}")

    objectives = sorted(df["excitation_objective"].dropna().unique())
    colors = plt.get_cmap("tab10")
    objective_colors = {objective: colors(i % colors.N) for i, objective in enumerate(objectives)}
    figures = {}

    for system_name, system_df in df.groupby("system_name", sort=True):
        model_types = sorted(system_df["model_type"].dropna().unique())
        system_objectives = [
            objective for objective in objectives if objective in system_df["excitation_objective"].values
        ]
        if not model_types or not system_objectives:
            continue

        fig, axes = plt.subplots(1, len(metrics), figsize=(figure_width, 3.5), constrained_layout=True)
        box_width = 0.8 / len(system_objectives)

        for ax, (metric, label) in zip(axes, metrics.items()):
            for model_index, model_type in enumerate(model_types):
                for objective_index, objective in enumerate(system_objectives):
                    group = system_df.loc[
                        (system_df["model_type"] == model_type) & (system_df["excitation_objective"] == objective),
                        metric,
                    ]
                    values = group.to_numpy(dtype=float)
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
                            boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.7},
                            whiskerprops={"color": color},
                            capprops={"color": color},
                            medianprops={"color": "black", "linewidth": 1.5},
                            flierprops={"marker": ".", "markeredgecolor": color, "alpha": 0.6},
                            showfliers=not hide_outliers,
                        )
                    ax.text(
                        position,
                        0.98,
                        f"n={values.size}",
                        transform=ax.get_xaxis_transform(),
                        ha="center",
                        va="top",
                        fontsize=8,
                        color=color,
                    )

            if log_scale:
                ax.set_yscale("log")
            ax.set_title(label)
            ax.set_xlabel("Model type")
            ax.set_xticks(range(len(model_types)), [model.replace("_", " ") for model in model_types])
            ax.set_xlim(-0.5, len(model_types) - 0.5)
            ax.margins(y=0.15)
            ax.grid("both", alpha=0.4)
            ax.set_axisbelow(True)
            # ax.spines[["top", "right"]].set_visible(False)

        axes[0].set_ylabel("Error (log scale)" if log_scale else "Error")

        fig.suptitle(system_name.replace("_", " ").title(), fontsize=16)
        fig.legend(
            handles=[
                Patch(facecolor=objective_colors[objective], alpha=0.7, label=objective.replace("_", " "))
                for objective in system_objectives
            ],
            title="Excitation objective",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=len(system_objectives),
            frameon=True,
        )
        figures[system_name] = (fig, axes)

    return figures


def visualize_model_prediction_metric_boxplots(
    df,
    *,
    log_scale: bool = False,
    figure_width: float = 6.0,
    hide_outliers: bool = False,
):
    """Return {system_name: (figure, axes)} for prediction errors."""
    metadata = ["system_name", "model_type", "excitation_objective"]
    rename = {f"prediction.metrics.{metric}": f"metrics.{metric}" for metric in ("rmse", "mae", "wce")}

    columns = metadata + list(rename)
    missing = set(columns).difference(df.columns)
    if missing:
        raise ValueError(f"Missing dataframe columns: {sorted(missing)}")

    plot_df = df.loc[:, columns].rename(columns=rename)
    plot_df = plot_df.dropna(subset=list(rename.values()), how="all")

    return visualize_model_metric_boxplots(
        plot_df,
        log_scale=log_scale,
        figure_width=figure_width,
        hide_outliers=hide_outliers,
    )


def visualize_model_prediction(
    validation_points_per_dim: int,
    wrapped_model: ModelWrapper,
    wrapped_env: EnvWrapper,
    labels: list[str] | None = None,
):
    n_features = wrapped_env.model.obs_dim + wrapped_env.model.action_dim
    pred, pred_gt = model_prediction(
        wrapped_model,
        wrapped_env,
        validation_points=None,
        validation_points_per_dim=validation_points_per_dim,
    )
    difference_map = pred - pred_gt
    reshaped_difference_map = difference_map.reshape([validation_points_per_dim] * n_features + [-1])
    abs_map = jnp.linalg.norm(reshaped_difference_map, axis=-1)

    fig, axs = plt.subplots(nrows=n_features, ncols=n_features, figsize=(6, 6), sharex=True, sharey=True)

    feature_indices = jnp.arange(0, n_features, 1).tolist()

    if labels is None:
        labels = np.arange(0, n_features, 1)

    for i in range(n_features):
        for j in range(n_features):

            axs[j, i].grid(True)
            axs[j, i].set_xlim(-1.1, 1.1)
            axs[j, i].set_ylim(-1.1, 1.1)

            reduction_indices = [f_idx for f_idx in feature_indices if not (f_idx == i or f_idx == j)]
            if len(reduction_indices) == n_features - 1:
                continue

            image = jnp.mean(jnp.abs(abs_map), axis=tuple(reduction_indices))

            if i < j:
                image = jnp.transpose(image)

            axs[j, i].imshow(image, origin="lower", extent=[-1, 1, -1, 1])
            axs[j, 0].set_ylabel(labels[j])

        axs[-1, i].set_xlabel(labels[i])
    fig.tight_layout()

    return fig, axs


def visualize_model_rollout_metric_boxplots(
    df,
    *,
    signals: list[str] | None = None,
    log_scale: bool = False,
    figure_width: float = 6.0,
    hide_outliers: bool = False,
):
    """Return {(system_name, signal): (figure, axes)} for rollout errors.

    Automatically discovers test signals when signals=None.
    Finite-value filtering and sample counts follow the existing plotter.
    """
    metadata = ["system_name", "model_type", "excitation_objective"]
    metrics = ("rmse", "mae", "wce")

    if signals is None:
        signals = sorted(
            {
                column.removeprefix("rollout.").removesuffix(".metrics.rmse")
                for column in df.columns
                if column.startswith("rollout.") and column.endswith(".metrics.rmse")
            }
        )

    if not signals:
        raise ValueError("No rollout signals found; expected " "rollout.<signal>.metrics.rmse columns.")

    figures = {}

    for signal in signals:
        rename = {f"rollout.{signal}.metrics.{metric}": f"metrics.{metric}" for metric in metrics}
        missing = set(metadata + list(rename)).difference(df.columns)
        if missing:
            raise ValueError(f"Missing dataframe columns: {sorted(missing)}")

        plot_df = df.loc[:, metadata + list(rename)].rename(columns=rename)
        plot_df = plot_df.dropna(subset=list(rename.values()), how="all")

        signal_figures = visualize_model_metric_boxplots(
            plot_df,
            log_scale=log_scale,
            figure_width=figure_width,
            hide_outliers=hide_outliers,
        )

        for system_name, (fig, axes) in signal_figures.items():
            fig.suptitle(
                f"{system_name.replace('_', ' ').title()} — " f"{signal.replace('_', ' ')}",
                fontsize=16,
            )
            figures[(system_name, signal)] = (fig, axes)

    return figures


def visualize_model_rollouts(
    wrapped_model: ModelWrapper,
    observations: jax.Array,
    actions: jax.Array,
    *,
    trajectory_indices: list[int] | None = None,
    obs_labels: list[str] | None = None,
    n_examples=5,
    max_trajectory_length: int | None = 1000,
):
    if observations.ndim != 3 or actions.ndim != 3:
        raise ValueError("Expected three-dimensional trajectory arrays.")
    if observations.shape[0] != actions.shape[0]:
        raise ValueError("Trajectory counts must match.")
    if observations.shape[1] != actions.shape[1] + 1:
        raise ValueError("Each trajectory needs one more state than action.")

    pred_observations = eqx.filter_vmap(wrapped_model.rollout, in_axes=(0, 0, None))(
        observations[:, 0, :], actions, 1.0
    )
    feat_observations = wrapped_model.featurize(observations)

    pred_observations = pred_observations[:, :max_trajectory_length, :]
    feat_observations = feat_observations[:, :max_trajectory_length, :]

    if observations.shape[1] >= max_trajectory_length:
        actions = actions[:, : max_trajectory_length - 1, :]

    n_trajectories, n_observations, obs_dim = feat_observations.shape
    action_dim = actions.shape[-1]

    if obs_labels is None:
        obs_labels = np.arange(0, obs_dim, 1)
    if len(obs_labels) != obs_dim:
        raise ValueError("Provide one label per observation component.")

    if trajectory_indices is None:
        trajectory_indices = np.arange(0, min(n_examples, n_trajectories), 1, dtype=int)

    indices = np.asarray(trajectory_indices)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("Provide a nonempty sequence of trajectory indices.")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Trajectory indices must be integers.")
    if np.any((indices < 0) | (indices >= n_trajectories)):
        raise ValueError("Trajectory index is out of range.")

    time = np.arange(n_observations)

    fig, axs = plt.subplots(
        nrows=obs_dim + action_dim,
        ncols=len(indices),
        figsize=(4 * len(indices), 2.5 * (obs_dim + action_dim)),
        sharex=True,
        sharey="row",
        squeeze=False,
        layout="constrained",
    )

    for column, trajectory_index in enumerate(indices):
        for component in range(obs_dim):
            ax = axs[component, column]

            ax.plot(
                time,
                feat_observations[trajectory_index, :, component],
                color="black",
                linewidth=1.5,
                label="Reference",
            )
            ax.plot(
                time,
                pred_observations[trajectory_index, :, component],
                color="tab:orange",
                linestyle="--",
                linewidth=1.5,
                label="Model",
            )
            ax.grid(alpha=0.3)

            if column == 0:
                ax.set_ylabel(obs_labels[component])
            if component == 0:
                ax.set_title(f"Trajectory {trajectory_index}")

        if action_dim == 1:
            ax = axs[-1, column]

            ax.plot(
                time[:-1],
                actions[trajectory_index, :, 0],
                color="black",
                linewidth=1.5,
            )
            ax.grid(alpha=0.3)
            ax.set_xlabel(r"$t$ in s")
            if column == 0:
                ax.set_ylabel(r"$\tilde{F}$")
        else:
            raise NotImplementedError()

    axs[0, 0].legend()
    return fig, axs
