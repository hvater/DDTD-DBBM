import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx

from admpe.evaluation.model_wrapping import EnvWrapper, ModelWrapper
from admpe.evaluation.modeling_evaluation import model_prediction


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

    n_trajectories, n_observations, obs_dim = feat_observations.shape

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
        nrows=obs_dim,
        ncols=len(indices),
        figsize=(4 * len(indices), 2.5 * obs_dim),
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
            if component == obs_dim - 1:
                ax.set_xlabel("Time [s]")

    axs[0, 0].legend()
    return fig, axs
