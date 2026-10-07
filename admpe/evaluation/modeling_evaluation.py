import pathlib

import jax
import equinox as eqx

from dmpe.utils.density_estimation import build_grid

from admpe.data_management import load_excitation_result

from admpe.evaluation.modeling_metrics import (
    prediction_metrics,
    rollout_metrics,
)
from admpe.evaluation.model_wrapping import EnvWrapper, ModelWrapper


def load_corresponding_excitation_result(modeling_result):
    """TODO: Move to a modeling result dataclass? Also visualizations could go in there"""

    try:
        excitation_result = load_excitation_result(modeling_result["metadata"]["dataset_path"])
    except FileNotFoundError:

        path = pathlib.Path(modeling_result["metadata"]["dataset_path"].removeprefix("PosixPath('").removesuffix("')"))
        excitation_result = load_excitation_result(path)

    return excitation_result


def generate_default_prediction_validation_points(dim, validation_points_per_dim):

    validation_points_per_dim = 20 if validation_points_per_dim is None else validation_points_per_dim

    return build_grid(
        dim=dim,
        low=-1.05,
        high=1.05,
        points_per_dim=validation_points_per_dim,
    )


def model_prediction(
    wrapped_model: ModelWrapper,
    wrapped_env: EnvWrapper,
    validation_points: jax.Array | None = None,
    validation_points_per_dim: int | None = None,
):
    dim = wrapped_env.model.obs_dim + wrapped_env.model.action_dim

    if validation_points is None:
        validation_points = generate_default_prediction_validation_points(dim, validation_points_per_dim)

    observations = validation_points[:, : wrapped_env.model.obs_dim]
    actions = validation_points[:, wrapped_env.model.obs_dim :]

    pred = eqx.filter_vmap(wrapped_model.step, in_axes=(0, 0, None))(observations, actions, 1.0)
    pred_gt = eqx.filter_vmap(wrapped_env.step, in_axes=(0, 0, None))(observations, actions, wrapped_env.model.tau)

    return pred, pred_gt


def evaluate_model_prediction(
    wrapped_model: ModelWrapper,
    wrapped_env: EnvWrapper,
    validation_points: jax.Array | None = None,
    *,
    return_predictions: bool = False,
):
    pred, pred_gt = model_prediction(wrapped_model, wrapped_env, validation_points)

    result = {"metrics": prediction_metrics(reference=pred_gt, prediction=pred)}

    if return_predictions:
        result["predictions"] = pred
        result["gt"] = pred_gt

    return result


def evaluate_model_single_transitions():
    raise NotImplementedError()


def evaluate_model_rollout(
    wrapped_model,
    observations,
    actions,
    *,
    horizons: list[int] | None = None,
    return_predictions: bool = False,
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

    result = {
        "metrics": rollout_metrics(
            reference=feat_observations,
            prediction=pred_observations,
            horizons=horizons,
        )
    }
    if return_predictions:
        result["rollout_predictions"] = pred_observations
        result["gt"] = feat_observations

    return result


def evaluate_model():

    # load query points for one-step prediction
    # load testing trajectories for model-rollout testing

    # evaluate model performance

    # store results

    raise NotImplementedError()


def update_modeling_df():
    raise NotImplementedError()
