import pathlib
import json

import tqdm
import pandas as pd
import numpy as np

import jax
import equinox as eqx

from dmpe.utils.density_estimation import build_grid

from admpe.data_management import load_excitation_result, load_modeling_result

from admpe.evaluation.modeling_metrics import (
    prediction_metrics,
    rollout_metrics,
)
from admpe.evaluation.model_wrapping import EnvWrapper, ModelWrapper, get_wrapped_model_and_env


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


def to_python(value):
    """Transforms all jax arrays to floats or lists for storing in the results df."""

    if isinstance(value, dict):
        return {key: to_python(item) for key, item in value.items()}

    if isinstance(value, jax.Array):
        array = np.asarray(value)
        return float(array.item()) if array.ndim == 0 else array.tolist()

    return value


def update_modeling_df(
    df: pd.DataFrame,
    result_paths: list[pathlib.Path],
):
    """Append modeling experiments that are not already recorded.

    Args:
        df: Existing dataframe, or None to start a new one.
        result_paths: Paths to modeling experiment directories.

    Returns:
        A new dataframe containing existing and newly evaluated results.
    """

    if df is None:
        df = pd.DataFrame(columns=["exp_id"])
    else:
        df = df.copy()

    if "exp_id" not in df.columns:
        raise ValueError("The dataframe must contain an 'exp_id' column.")
    if df["exp_id"].isna().any() or df["exp_id"].duplicated().any():
        raise ValueError("Existing experiment exp_ids must be nonempty and unique.")

    existing_exp_ids = set(df["exp_id"])
    new_rows = []
    for result_path in tqdm.tqdm(result_paths):
        # TODO: actually reduce the result_paths to only paths...
        result_path = pathlib.Path(result_path["path"])

        metadata = json.loads((result_path / "experiment.json").read_text())
        exp_id = metadata["exp_id"]

        if exp_id in existing_exp_ids:
            continue

        if metadata.get("status") != "completed":
            raise ValueError(f"Experiment is not completed: {result_path}")

        result = load_modeling_result(result_path)

        # TODO: extend to rollout + prediction + all other relevant metrics
        # use 'evaluate_model'
        wrapped_model, wrapped_env = get_wrapped_model_and_env(result)
        metrics_jax = evaluate_model_prediction(wrapped_model, wrapped_env)

        metrics = to_python(metrics_jax)

        row = dict(
            exp_id=exp_id,
            path=str(result_path),
            system_name=metadata["system_name"],
            model_type=metadata["model_type"],
            training_seed=metadata["seed"],
            dataset_path=metadata["dataset_path"],
            use_f64=metadata.get("use_f64"),
        )

        if row.keys() & metrics.keys():
            raise ValueError("Metric names overlap with metadata columns.")
        row.update(metrics)
        new_rows.append(row)
        existing_exp_ids.add(exp_id)

    if not new_rows:
        return df

    new_df_entries = pd.json_normalize(
        [to_python(run) for run in new_rows],
        sep=".",
    )

    return pd.concat(
        [df, new_df_entries],
        ignore_index=True,
    )
