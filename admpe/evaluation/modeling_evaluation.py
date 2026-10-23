import pathlib
import json

import tqdm
import pandas as pd
import numpy as np

import jax
import equinox as eqx

from dmpe.utils.density_estimation import build_grid

from admpe.data_management import (
    load_excitation_result,
    load_modeling_result,
    load_testing_sequences,
    TESTING_TRAJECTORIES_ROOT,
)

from admpe.evaluation.modeling_metrics import (
    prediction_metrics,
    rollout_metrics,
)
from admpe.evaluation.model_wrapping import EnvWrapper, ModelWrapper, get_wrapped_model_and_env
from admpe.sets import CartesianGridSet


def load_corresponding_excitation_result(modeling_result):
    """TODO: Move to a modeling result dataclass? Also visualizations could go in there"""

    try:
        excitation_result = load_excitation_result(pathlib.Path(modeling_result["metadata"]["dataset_path"]))
    except FileNotFoundError:
        path = pathlib.Path(modeling_result["metadata"]["dataset_path"].removeprefix("PosixPath('").removesuffix("')"))
        excitation_result = load_excitation_result(pathlib.Path(path))

    return excitation_result


def generate_default_prediction_validation_points(dim, validation_points_per_dim):

    validation_points_per_dim = 20 if validation_points_per_dim is None else validation_points_per_dim

    return build_grid(
        dim=dim,
        low=-1.0,
        high=1.0,
        points_per_dim=validation_points_per_dim,
    )


def model_prediction(
    wrapped_model: ModelWrapper,
    wrapped_env: EnvWrapper,
    validation_points: jax.Array | None = None,
    validation_points_per_dim: int | None = None,
    *,
    safely_excitable_set: CartesianGridSet | None = None,
):
    dim = wrapped_env.model.obs_dim + wrapped_env.model.action_dim

    if validation_points is None:
        validation_points = generate_default_prediction_validation_points(dim, validation_points_per_dim)
    if safely_excitable_set is not None:
        membership = safely_excitable_set.contains(validation_points)
        validation_points = validation_points[membership]
        if validation_points.shape[0] == 0:
            raise ValueError("No validation points belong to the safely excitable set.")

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
    safely_excitable_set: CartesianGridSet | None = None,
):
    pred, pred_gt = model_prediction(
        wrapped_model,
        wrapped_env,
        validation_points,
        safely_excitable_set=safely_excitable_set,
    )

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


def evaluate_model(
    system_name: str,
    wrapped_model: ModelWrapper,
    wrapped_env: EnvWrapper,
    filter_testing_points: bool = False,
):
    safely_excitable_set = CartesianGridSet.from_system_name(system_name)[0] if filter_testing_points else None
    prediction_metric_values = evaluate_model_prediction(
        wrapped_model,
        wrapped_env,
        safely_excitable_set=safely_excitable_set,
    )

    if filter_testing_points:
        rollout_data_path = TESTING_TRAJECTORIES_ROOT / "safely_excitable_trajectories"
    else:
        rollout_data_path = TESTING_TRAJECTORIES_ROOT / "box_constrained_trajectories"
    testing_data = load_testing_sequences(system_name, root=rollout_data_path)
    rollout_metric_values = {}
    for signal_key, signal_testing_data in testing_data.items():
        rollout_metric_values[signal_key] = evaluate_model_rollout(
            wrapped_model,
            observations=signal_testing_data["observations"],
            actions=signal_testing_data["actions"],
        )

    result = {
        "prediction": prediction_metric_values,
        "rollout": rollout_metric_values,
    }

    return result


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
    filter_prediction_points: bool = False,
):
    """Append new modeling experiments and fill in their excitation objectives.

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

    prediction_domain = "safely_excitable" if filter_prediction_points else "full_box"

    if not df.empty:
        if "prediction_domain" not in df.columns:
            raise ValueError(
                "Existing results have no recorded prediction domain. "
                "Recompute them or explicitly migrate verified legacy results."
            )

        domains = df["prediction_domain"]

        if domains.isna().any() or not domains.eq(prediction_domain).all():
            raise ValueError(
                f"Existing results do not all use {prediction_domain!r}. "
                "Use a separate dataframe for this evaluation domain."
            )

    if "exp_id" not in df.columns:
        raise ValueError("The dataframe must contain an 'exp_id' column.")
    if df["exp_id"].isna().any() or df["exp_id"].duplicated().any():
        raise ValueError("Existing experiment exp_ids must be nonempty and unique.")

    excitation_objectives = {}

    def get_excitation_objective(dataset_path):
        # Older metadata stores paths as repr(PosixPath(...)).
        dataset_path = pathlib.Path(str(dataset_path).removeprefix("PosixPath('").removesuffix("')"))
        if dataset_path not in excitation_objectives:
            excitation_metadata = json.loads((dataset_path / "experiment.json").read_text())
            excitation_objectives[dataset_path] = excitation_metadata["excitation_objective"]
        return excitation_objectives[dataset_path]

    existing_exp_ids = set(df["exp_id"])
    new_rows = []
    for result_path in tqdm.tqdm(result_paths):

        try:
            result_path = pathlib.Path(result_path)
        except TypeError:
            result_path = pathlib.Path(result_path["path"])

        metadata = json.loads((result_path / "experiment.json").read_text())
        exp_id = metadata["exp_id"]

        if exp_id in existing_exp_ids:
            continue

        if metadata.get("status") != "completed":
            raise ValueError(f"Experiment is not completed: {result_path}")

        result = load_modeling_result(result_path)

        wrapped_model, wrapped_env = get_wrapped_model_and_env(result)
        metrics_jax = evaluate_model(
            metadata["system_name"], wrapped_model, wrapped_env, filter_prediction_points=filter_prediction_points
        )

        metrics = to_python(metrics_jax)

        row = dict(
            exp_id=exp_id,
            path=str(result_path),
            system_name=metadata["system_name"],
            model_type=metadata["model_type"],
            training_seed=metadata["seed"],
            dataset_path=metadata["dataset_path"],
            excitation_objective=get_excitation_objective(metadata["dataset_path"]),
            use_f64=metadata.get("use_f64"),
            prediction_domain=prediction_domain,
        )

        if row.keys() & metrics.keys():
            raise ValueError("Metric names overlap with metadata columns.")
        row.update(metrics)
        new_rows.append(row)
        existing_exp_ids.add(exp_id)

    if not new_rows:
        print("No new entries have been found. Returing data as provided.")
        return df

    print(f"{len(new_rows)} new modeling results have been found and added to the provided data frame.")

    new_df_entries = pd.json_normalize(
        [to_python(run) for run in new_rows],
        sep=".",
    )

    return pd.concat(
        [df, new_df_entries],
        ignore_index=True,
    )
