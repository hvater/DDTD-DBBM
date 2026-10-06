"""Locations and serialization for excitation experiment artifacts."""

import logging as log
import sys
import dataclasses
import json
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd

import equinox as eqx
import jax
import jax.numpy as jnp

from admpe.excitation.excitation_experiment_setup import setup_model, setup_estimator

DESIRED_DT_FMT = "%Y-%m-%d %H:%M:%S"  # desired datetime format
LOG_FORMATTER = log.Formatter("%(asctime)s | %(levelname)s : %(message)s", DESIRED_DT_FMT)
LOG_COLUMN_WIDTH = 40


def setup_package_logging():
    """Configure package-wide logging settings."""
    stream_handler = log.StreamHandler(sys.stdout)
    stream_handler.setFormatter(LOG_FORMATTER)

    log.basicConfig(
        level=log.INFO,
        handlers=[stream_handler],
    )


setup_package_logging()

## Common data paths:
EXCITATION_EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent / "data" / "excitation_experiments"
EXCITATION_EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)

MODELING_EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent / "data" / "modeling_experiments"
MODELING_EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)


def _json_value(value):
    """Record arrays and configuration objects without pickling executable code."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if hasattr(value, "tolist"):
        return value.tolist()
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}" if hasattr(value, "__qualname__") else repr(value)
    if dataclasses.is_dataclass(value):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {f.name: _json_value(getattr(value, f.name)) for f in dataclasses.fields(value)},
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return repr(value)


def write_metadata(path: Path, metadata: dict):
    path.write_text(json.dumps(_json_value(metadata), indent=2) + "\n")


def get_nested(metadata, key):
    value = metadata
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def provide_results_overview(results: dict, overview_fields: dict):
    """Based on a filtered list of results, provide an overview over the most relevant metadata."""

    rows = []
    for result in results:
        folder = Path(result["path"])
        read_error = None

        try:
            metadata = json.loads((folder / "experiment.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            metadata = {}
            read_error = str(error)

        rows.append(
            {
                "name": folder.name,
                "path": folder,
                "status": metadata.get("status", "unknown"),
                **{column: get_nested(metadata, key) for column, key in overview_fields.items()},
                "error": metadata.get("error") or read_error,
            }
        )

        columns = [
            "name",
            "path",
            "status",
            *overview_fields,
            "error",
        ]

    df = pd.DataFrame.from_records(rows, columns=columns).convert_dtypes()
    return df.sort_values("name").reset_index(drop=True)


def save_excitation_result(path: Path, result: tuple):
    """Save trajectories and array leaves; metadata records the setup separately.

    Equinox files require a matching template for deserialization. Callback
    outputs remain available to the callback and are not serialized here.
    """
    observations, actions, model, estimator, losses, proposed_actions, _ = result
    np.savez_compressed(
        path / "trajectory.npz",
        observations=np.asarray(observations),
        actions=np.asarray(actions),
        losses=np.asarray(losses),
        proposed_actions=np.asarray(proposed_actions),
    )
    eqx.tree_serialise_leaves(path / "model.eqx", model)
    eqx.tree_serialise_leaves(path / "estimator.eqx", estimator)


def list_excitation_results(
    path: Path | None = None,
    system_name: str | list[str] | None = None,
    excitation_objective: str | list[str] | None = None,
    model_type: str | list[str] | None = None,
    exp_name: str | None = None,
    enforce_identical_exp_name: bool = True,
    status: str | list[str] | None = "completed",
) -> list[str]:

    root = EXCITATION_EXPERIMENTS_ROOT if path is None else path

    def normalize(value: str | list[str] | None) -> set[str] | None:
        if value is None:
            return None
        return {value} if isinstance(value, str) else set(value)

    filters = [
        normalize(system_name),
        normalize(excitation_objective),
        normalize(model_type),
    ]

    allowed_statuses = normalize(status)

    results = []
    # iterate over all folders at the targeted path
    for folder in root.iterdir():
        if not folder.is_dir():
            continue

        # separate the seed from the remaining name
        name_without_seed, separator, seed_suffix = folder.name.rpartition("-seed")
        if not separator:
            continue

        # separate the uuid_suffic from the identifier and check that is follows the usual pattern
        prefix, separator, uuid_suffix = name_without_seed.rpartition("-")
        if not separator or len(uuid_suffix) != 8 or any(c not in "0123456789abcdef" for c in uuid_suffix):
            continue

        # check that the remaining prefix consists of three parts that are separated by '-'
        parts = prefix.split("-", maxsplit=3)
        if len(parts) < 3:
            continue

        # check that the filters are
        if any(allowed is not None and actual not in allowed for actual, allowed in zip(parts[:3], filters)):
            continue

        # check for 'exp_name' if one is given
        # depending on the flag 'enforce_identical_exp_name' the specified exp_name must be fully identical to the
        # excitation result or it may be a subset of the exp_name of the excitation result
        actual_exp_name = parts[3] if len(parts) == 4 else ""
        if exp_name is not None:
            if enforce_identical_exp_name:
                if actual_exp_name != exp_name:
                    continue
            elif exp_name not in actual_exp_name:
                continue

        try:
            metadata = json.loads((folder / "experiment.json").read_text())
            actual_status = metadata.get("status", "unknown")
        except (OSError, json.JSONDecodeError):
            actual_status = "unknown"

        if allowed_statuses is not None and actual_status not in allowed_statuses:
            continue

        results.append(
            {
                "name": folder.name,
                "path": folder,
                "status": actual_status,
            }
        )

    return sorted(results, key=lambda result: result["name"])


def provide_excitation_results_overview(results: dict):
    EXCITATION_OVERVIEW_FIELDS = {
        "system": "system_name",
        "objective": "excitation_objective",
        "model": "model_type",
        "seed": "seed",
        "n_time_steps": "n_time_steps",
        "elapsed_seconds": "elapsed_seconds",
        "use_f64": "use_f64",
        "device": "device",
        "prediction_horizon": "params.alg_params.n_prediction_steps",
        "n_opt_steps": "params.alg_params.n_opt_steps",
        "n_starts": "params.alg_params.n_starts",
        "bandwidth": "params.alg_params.bandwidth",
        "points_per_dim": "params.alg_params.points_per_dim",
        "model_width": "params.model_params.width_size",
        "model_depth": "params.model_params.depth",
        "training_batch_size": ("params.model_trainer_params.training_batch_size"),
        "model_lr": "params.model_trainer_params.model_lr",
    }
    return provide_results_overview(results, EXCITATION_OVERVIEW_FIELDS)


def load_excitation_result(path: Path):
    """Based on a given folder path, load the corresponding experiment results
    and reconstruct the necessary '.eqx'-objects.
    """
    foldername = Path(path)

    metadata = json.loads((foldername / "experiment.json").read_text())
    if metadata["status"] != "completed":
        raise ValueError(f"Experiment is not completed: {path}")

    # TODO: alternatively have the option to swith between f32 / f64?
    jax.config.update("jax_enable_x64", metadata["use_f64"])
    params = metadata["params"]

    with np.load(path / "trajectory.npz", allow_pickle=False) as archive:
        trajectory = {
            name: jnp.asarray(archive[name]) for name in ("observations", "actions", "losses", "proposed_actions")
        }

    # reconstruct objects
    loading_params = deepcopy(params["model_params"])
    loading_params["key"] = jnp.array(loading_params["key"], dtype=jnp.uint32)
    model_template = setup_model(metadata["system_name"], metadata["model_type"], loading_params)
    model = eqx.tree_deserialise_leaves(foldername / "model.eqx", model_template)

    estimator_template = setup_estimator(metadata["excitation_objective"], params)
    estimator = eqx.tree_deserialise_leaves(foldername / "estimator.eqx", estimator_template)

    return {
        "metadata": metadata,
        **trajectory,
        "model": model,
        "estimator": estimator,
    }


def save_modeling_result(path: Path, result):
    eqx.tree_serialise_leaves(path / "model.eqx", result)


def load_modeling_result(path: Path):
    foldername = Path(path)

    metadata = json.loads((foldername / "experiment.json").read_text())
    if metadata["status"] != "completed":
        raise ValueError(f"Experiment is not completed: {path}")

    params = metadata["params"]
    loading_params = deepcopy(params["model_params"])
    loading_params["key"] = jnp.array(loading_params["key"], dtype=jnp.uint32)
    model_template = setup_model(metadata["system_name"], metadata["model_type"], loading_params)
    model = eqx.tree_deserialise_leaves(foldername / "model.eqx", model_template)

    return {
        "metadata": metadata,
        "model": model,
    }


def list_modeling_results(
    path: Path | None = None,
    system_name: str | list[str] | None = None,
    model_type: str | list[str] | None = None,
    exp_name: str | None = None,
    enforce_identical_exp_name: bool = True,
    status: str | list[str] | None = "completed",
) -> list[dict]:

    root = MODELING_EXPERIMENTS_ROOT if path is None else Path(path)

    def normalize(value: str | list[str] | None) -> set[str] | None:
        if value is None:
            return None
        return {value} if isinstance(value, str) else set(value)

    filters = [
        normalize(system_name),
        normalize(model_type),
    ]
    allowed_statuses = normalize(status)

    results = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue

        name_without_seed, separator, seed_suffix = folder.name.rpartition("-seed")
        if not separator or not seed_suffix.isdigit():
            continue

        prefix, separator, uuid_suffix = name_without_seed.rpartition("-")
        if not separator or len(uuid_suffix) != 8 or any(c not in "0123456789abcdef" for c in uuid_suffix):
            continue

        if not prefix.startswith("modeling-"):
            continue

        # Preserve hyphens within the optional experiment name.
        parts = prefix.removeprefix("modeling-").split("-", maxsplit=2)
        if len(parts) < 2:
            continue

        if any(allowed is not None and actual not in allowed for actual, allowed in zip(parts[:2], filters)):
            continue

        actual_exp_name = parts[2] if len(parts) == 3 else ""
        if exp_name is not None:
            if enforce_identical_exp_name:
                if actual_exp_name != exp_name:
                    continue
            elif exp_name not in actual_exp_name:
                continue

        try:
            metadata = json.loads((folder / "experiment.json").read_text())
            actual_status = metadata.get("status", "unknown")
        except (OSError, json.JSONDecodeError):
            actual_status = "unknown"

        if allowed_statuses is not None and actual_status not in allowed_statuses:
            continue

        results.append(
            {
                "name": folder.name,
                "path": folder,
                "status": actual_status,
            }
        )

    return sorted(results, key=lambda result: result["name"])


def provide_modeling_results_overview(results: dict):
    MODELING_OVERVIEW_FIELDS = {
        "system": "system_name",
        "model": "model_type",
        "seed": "seed",
        "dataset_path": "dataset_path",
        "elapsed_seconds": "elapsed_seconds",
        "use_f64": "use_f64",
        "device": "device",
        "tau": "params.model_trainer_params.tau",
        "model_width": "params.model_params.width_size",
        "model_depth": "params.model_params.depth",
        "training_batch_size": "params.model_trainer_params.training_batch_size",
        "n_train_steps": "params.model_trainer_params.n_train_steps",
        "sequence_length": "params.model_trainer_params.sequence_length",
        "model_lr": "params.model_trainer_params.model_optimizer.lr",
    }
    return provide_results_overview(results, MODELING_OVERVIEW_FIELDS)
