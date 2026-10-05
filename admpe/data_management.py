"""Locations and serialization for excitation experiment artifacts."""

import dataclasses
import json
from pathlib import Path

import equinox as eqx
import numpy as np

import logging as log
import sys

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
):
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

    results = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue

        name_without_seed, separator, seed_suffix = folder.name.rpartition("-seed")
        if not separator:
            continue

        prefix, separator, uuid_suffix = name_without_seed.rpartition("-")
        if not separator or len(uuid_suffix) != 8 or any(c not in "0123456789abcdef" for c in uuid_suffix):
            continue

        parts = prefix.split("-", maxsplit=3)
        if len(parts) < 3:
            continue

        if any(allowed is not None and actual not in allowed for actual, allowed in zip(parts[:3], filters)):
            continue

        actual_exp_name = parts[3] if len(parts) == 4 else ""
        if exp_name is not None:
            if enforce_identical_exp_name:
                if actual_exp_name != exp_name:
                    continue
            elif exp_name not in actual_exp_name:
                continue

        results.append(folder)

    return sorted(results)


def load_excitation_result(path: Path):
    raise NotImplementedError()


def save_modeling_result(path: Path, result):
    raise NotImplementedError()


def save_modeling_result(path: Path, result):
    raise NotImplementedError()


def list_modeling_results(path: Path | None):
    raise NotImplementedError()
