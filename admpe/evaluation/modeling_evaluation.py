import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx

from dmpe.utils.density_estimation import build_grid

from admpe.evaluation.modeling_metrics import (
    prediction_metrics,
    rollout_metrics,
)
from admpe.evaluation.model_wrapping import EnvWrapper, ModelWrapper


def evaluate_model():
    raise NotImplementedError()


def update_modeling_df():
    raise NotImplementedError()


def visualize_model_prediction(
    validation_points_per_dim: int,
    wrapped_model: ModelWrapper,
    wrapped_env: EnvWrapper,
    labels: list[str] | None = None,
):
    n_features = wrapped_env.model.obs_dim + wrapped_env.model.action_dim
    validation_grid = build_grid(dim=n_features, low=-1.05, high=1.05, points_per_dim=validation_points_per_dim)

    observations = validation_grid[:, : wrapped_env.model.obs_dim]
    actions = validation_grid[:, wrapped_env.model.obs_dim :]

    pred = eqx.filter_vmap(wrapped_model.step, in_axes=(0, 0, None))(observations, actions, 1.0)
    pred_gt = eqx.filter_vmap(wrapped_env.step, in_axes=(0, 0, None))(observations, actions, wrapped_env.model.tau)

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


def visualize_model_rollouts():
    raise NotImplementedError()
