import pathlib
import logging as log

import matplotlib.pyplot as plt
import pandas as pd
import equinox as eqx
import jax.numpy as jnp

from dmpe.models.model_utils import MODEL_TAU

from admpe.data_management import load_validation_sequences
from admpe.excitation.excitation_experiment_setup import setup_environment
from admpe.evaluation.modeling_evaluation import (
    generate_default_prediction_validation_points,
)
from admpe.evaluation.model_wrapping import EnvWrapper


def generate_one_step_validation_data(
    system_name,
    points_per_dim=10,
):
    env, _, featurize, _, _ = setup_environment(system_name, model_type="neural_euler")

    # Match the deterministic environment used in model evaluation.
    env = eqx.tree_at(lambda e: e.process_noise_variance, env, 0.0)

    points = generate_default_prediction_validation_points(
        dim=env.obs_dim + env.action_dim,
        validation_points_per_dim=points_per_dim,
    )
    observations = points[:, : env.obs_dim]
    actions = points[:, env.obs_dim :]
    wrapped_env = EnvWrapper(env, featurize)
    step_batch = eqx.filter_jit(eqx.filter_vmap(wrapped_env.step, in_axes=(0, 0, None)))

    targets = step_batch(observations, actions, env.tau)

    return {
        "observations": observations,
        "actions": actions,
        "targets": targets,
    }


def load_rollout_validation_data(
    system_name,
):
    validation_data = load_validation_sequences(system_name)
    if not validation_data:
        log.warning("No rollout validation data available")
    return validation_data


def get_validation_data(
    system_name,
    one_step_points_per_dim: int = 10,
):

    validation_data = {}
    validation_data["one_step"] = generate_one_step_validation_data(
        system_name=system_name,
        points_per_dim=one_step_points_per_dim,
    )
    validation_data["rollout"] = load_rollout_validation_data(system_name)

    return validation_data


def rollout_mse(model, validation_rollout_data, tau, featurize, *, training_horizon=None):

    if training_horizon is not None and training_horizon < 1:
        raise ValueError("The training horizon must be positive.")

    rollout_mse_values = {}

    for signal_key, signal_validation_data in validation_rollout_data.items():

        observations = jnp.asarray(signal_validation_data["observations"])
        actions = jnp.asarray(signal_validation_data["actions"])

        predictions = eqx.filter_vmap(model, in_axes=(0, 0, None))(
            observations[:, 0, :],
            actions,
            MODEL_TAU,
        )

        squared_error = (featurize(observations)[:, 1:] - featurize(predictions)[:, 1:]) ** 2
        rollout_mse_values[f"rollout_{signal_key}_mse"] = jnp.mean(squared_error)

        key = f"rollout_{signal_key}_mse"
        rollout_mse_values[key] = jnp.mean(squared_error)

        for horizon in sorted({9, 29, 100, 249, training_horizon} - {None}):
            if horizon > squared_error.shape[1]:
                raise ValueError(
                    f"{signal_key}: requested {horizon} prediction steps, "
                    f"but only {squared_error.shape[1]} are available."
                )

            rollout_mse_values[f"{key}_h{horizon}"] = jnp.mean(squared_error[:, :horizon])

    if training_horizon is not None:
        rollout_mse_values["rollout_training_horizon_mse"] = jnp.mean(
            jnp.asarray(
                [rollout_mse_values[f"rollout_{signal}_mse_h{training_horizon}"] for signal in validation_rollout_data]
            )
        )

    return rollout_mse_values


def one_step_mse(model, validation_data, tau, featurize):
    observations = validation_data["observations"]
    actions = validation_data["actions"]
    targets = validation_data["targets"]

    # add one dimension for the rollout model interface
    action_sequences = actions[:, None, :]

    pred = eqx.filter_vmap(model, in_axes=(0, 0, None))(observations, action_sequences, MODEL_TAU)

    # the pred has two elements: the provided initial observation and the one predicted observation
    assert pred.shape[1] == 2
    next_observation = pred[:, 1, :]

    error = eqx.filter_vmap(featurize)(next_observation) - targets
    return jnp.mean(error**2)


def validate(model, validation_data, tau, featurize, *, training_horizon=None):

    return {
        "one_step_mse": one_step_mse(
            model,
            validation_data["one_step"],
            tau,
            featurize,
        ),
        **rollout_mse(
            model,
            validation_data["rollout"],
            tau,
            featurize,
            training_horizon=training_horizon,
        ),
    }


def visualize_validation_scores(
    modeling_result,
    selection_metric="one_step_mse",
):

    validation_result = modeling_result["metadata"]["validation"]

    fig, axs = plt.subplots(1, 1, figsize=(7, 4))

    history = pd.DataFrame(validation_result["history"])

    axs.plot(history["optimizer_steps"], history[selection_metric])
    axs.set_yscale("log")
    axs.grid(True, alpha=0.3)

    axs.set_ylabel(selection_metric)
    axs.set_xlabel("optimizer steps")

    fig.suptitle(selection_metric)

    return fig, axs
