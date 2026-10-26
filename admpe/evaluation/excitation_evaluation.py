from functools import lru_cache

import json
import pathlib

import numpy as np
import pandas as pd
import tqdm
from scipy.spatial import cKDTree

import jax
import jax.numpy as jnp

from admpe.utils.helper import to_python
from admpe.data_management import list_modeling_results, load_modeling_result, load_excitation_result
from admpe.sets import SafeStateSet, SafelyExcitableSet
from dmpe.utils.density_estimation import build_grid
from dmpe.utils.metrics import diced_fill


def load_corresponding_modeling_results(
    excitation_result: dict,
    path: pathlib.Path | None = None,
    model_type: str | list[str] | None = None,
    exp_name: str | None = None,
    enforce_identical_exp_name: bool = True,
    status: str | list[str] | None = "completed",
) -> list[dict]:
    excitation_metadata = excitation_result["metadata"]
    excitation_id = excitation_metadata["exp_id"]

    # filter out any modeling results that could fit
    candidates = list_modeling_results(
        path=path,
        system_name=excitation_metadata["system_name"],
        excitation_objective=excitation_metadata["excitation_objective"],
        model_type=model_type,
        exp_name=exp_name,
        enforce_identical_exp_name=enforce_identical_exp_name,
        status=status,
    )

    # iterate over candidates and check if the `dataset_path` fits to the provided `excitation_result`
    results = []
    for result_path in candidates:
        metadata = json.loads((result_path["path"] / "experiment.json").read_text())
        dataset_path = metadata.get("dataset_path")
        if not dataset_path:
            continue

        dataset_path = str(dataset_path).removeprefix("PosixPath('").removesuffix("')")
        if pathlib.Path(dataset_path).name == excitation_id:
            results.append(load_modeling_result(result_path["path"]))

    return results


def load_excitation_comparison(excitation_results_overview, *, objectives, seed):
    """Load exactly one run per objective for the selected excitation seed."""
    selected_paths = {}

    for objective in objectives:
        matches = excitation_results_overview.loc[
            excitation_results_overview["objective"].eq(objective) & excitation_results_overview["seed"].eq(seed)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{objective}, seed={seed}: expected one run, "
                f"found {len(matches)}. Inspect the inventory "
                "and narrow the selection."
            )

        selected_paths[objective] = matches.iloc[0]["path"]

    return {objective: load_excitation_result(path) for objective, path in selected_paths.items()}


def evaluate_finite_data(observations, actions):
    finite_observations = np.isfinite(observations).all(axis=1)
    finite_actions = np.isfinite(actions).all(axis=1)
    finite_transitions = finite_observations[:-1] & finite_actions & finite_observations[1:]
    all_finite = bool(finite_observations.all() and finite_actions.all())

    return finite_observations, finite_actions, finite_transitions, all_finite


@lru_cache(maxsize=12)
def get_sets_and_eval_helpers(system_name, obs_dim, action_dim, points_per_dim, filter_testing_points):
    safely_excitable_set, set_path = SafelyExcitableSet.from_system_name(system_name)
    safe_state_set = SafeStateSet.from_safely_excitable_set(safely_excitable_set, obs_dim)

    state_grid = np.asarray(build_grid(obs_dim, -1.0, 1.0, points_per_dim))
    joint_grid = np.asarray(build_grid(obs_dim + action_dim, -1.0, 1.0, points_per_dim))
    half_width = float(np.abs(joint_grid[1, -1] - joint_grid[0, -1]) / 2)

    if filter_testing_points:
        state_grid = state_grid[np.asarray(safe_state_set.contains(state_grid))]
        joint_grid = joint_grid[np.asarray(safely_excitable_set.contains(joint_grid))]
    if len(state_grid) == 0 or len(joint_grid) == 0:
        raise ValueError("The selected grid has no reference points in the evaluation domain.")

    return safe_state_set, safely_excitable_set, state_grid, joint_grid, str(set_path), half_width


_diced_fill_block = jax.jit(diced_fill)


def _blocked_diced_fill(samples, reference, half_width, *, block_size=256):
    """Evaluate the existing empty-cell fraction in bounded support blocks.

    Weight each block by its support count so a shorter final block has the
    correct contribution.
    """
    if len(reference) == 0:
        raise ValueError("Diced fill needs at least one support point.")
    if not np.isfinite(half_width) or half_width <= 0:
        raise ValueError("Diced fill half-width must be positive and finite.")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer.")

    samples = jnp.asarray(samples)
    reference = jnp.asarray(reference)
    half_width = jnp.asarray(half_width)

    weighted_empty_fraction = 0.0
    for start in range(0, len(reference), block_size):
        block = reference[start : start + block_size]
        weighted_empty_fraction += float(_diced_fill_block(samples, block, half_width)) * len(block)

    return float(np.clip(weighted_empty_fraction / len(reference), 0.0, 1.0))


def evaluate_excitation_result(
    system_name: bool,
    observations: jax.Array,
    actions: jax.Array,
    *,
    filter_testing_points: bool = True,
    points_per_dim: int = 15,
    bound_tolerance: float = 1e-4,
    coverage_radius: float = 0.1,
):

    observations = np.asarray(observations, dtype=float)
    actions = np.asarray(actions, dtype=float)
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("Expected (time, features) observation and action arrays.")
    if len(actions) == 0 or len(observations) != len(actions) + 1:
        raise ValueError("Need at least one action and exactly one more observation than action.")
    if observations.shape[1] == 0 or actions.shape[1] == 0:
        raise ValueError("Observation and action feature dimensions must be nonempty.")

    obs_dim = observations.shape[-1]
    act_dim = actions.shape[-1]

    finite_observations, finite_actions, finite_transitions, all_finite = evaluate_finite_data(observations, actions)

    joint_samples = np.concatenate([observations[:-1], actions], axis=-1)

    safe_state_set, safely_excitable_set, state_grid, joint_grid, set_path, half_width = get_sets_and_eval_helpers(
        system_name,
        obs_dim,
        act_dim,
        points_per_dim,
        filter_testing_points,
    )

    def first_invalid(finite):
        indices = np.flatnonzero(~finite)
        return int(indices[0]) if indices.size else None

    def describe(values, finite):
        valid_values = values[finite]
        return {
            "n_samples": int(len(values)),
            "n_finite": int(finite.sum()),
            "nonfinite_fraction": float((~finite).mean()),
            "first_nonfinite_index": first_invalid(finite),
            "outside_box_fraction_of_finite": (
                float((np.abs(valid_values) > 1.0 + bound_tolerance).any(axis=1).mean())
                if len(valid_values)
                else float("nan")
            ),
            "max_box_excess": (
                float(np.maximum(np.abs(valid_values) - 1.0, 0.0).max()) if len(valid_values) else float("nan")
            ),
            "min_per_feature": valid_values.min(axis=0).tolist() if len(valid_values) else [None] * values.shape[1],
            "max_per_feature": valid_values.max(axis=0).tolist() if len(valid_values) else [None] * values.shape[1],
        }

    def not_safe_fraction(values, finite, domain):
        valid_values = values[finite]
        return float((~np.asarray(domain.contains(valid_values))).mean()) if len(valid_values) else float("nan")

    def coverage(samples, reference, domain):
        # A partial valid prefix is not scored as a successful complete run.
        if not all_finite:
            samples = samples[:0]
        elif filter_testing_points:
            samples = samples[np.asarray(domain.contains(samples))]
        else:
            samples = samples[(np.abs(samples) <= 1.0).all(axis=1)]

        distances = cKDTree(samples).query(reference, k=1)[0] if len(samples) else None

        return {
            "n_samples_in_domain": int(len(samples)),
            "n_reference_points": int(len(reference)),
            "mean_nearest_distance": float(distances.mean()) if distances is not None else float("nan"),
            "p95_nearest_distance": float(np.quantile(distances, 0.95)) if distances is not None else float("nan"),
            "max_nearest_distance": float(distances.max()) if distances is not None else float("nan"),
            "fraction_within_radius": (
                float((distances <= coverage_radius).mean()) if distances is not None else float("nan")
            ),
            "diced_fill": _blocked_diced_fill(samples, reference, half_width) if all_finite else float("nan"),
        }

    return {
        "evaluation": {
            "metric_version": 2,
            "domain": "safely_excitable" if filter_testing_points else "full_box",
            "points_per_dim": points_per_dim,
            "coverage_radius": float(coverage_radius),
            "bound_tolerance": float(bound_tolerance),
            "safely_excitable_set_path": set_path,
            "distance": "euclidean_normalized_coordinates",
            "diced_fill_half_width": half_width,
            "diced_fill_boundary": "strict_open_cells",
        },
        "validity": {
            "all_finite": all_finite,
            "n_transitions": int(len(actions)),
            "n_finite_transitions": int(finite_transitions.sum()),
            "first_nonfinite_transition": first_invalid(finite_transitions),
        },
        "observations": describe(observations, finite_observations),
        "actions": describe(actions, finite_actions),
        "constraints": {
            "n_finite_observations": int(finite_observations.sum()),
            "n_finite_state_action_pairs": int((finite_observations[:-1] & finite_actions).sum()),
            "outside_safe_state_fraction_of_finite": not_safe_fraction(
                observations, finite_observations, safe_state_set
            ),
            "outside_safely_excitable_fraction_of_finite": not_safe_fraction(
                joint_samples, finite_observations[:-1] & finite_actions, safely_excitable_set
            ),
        },
        "state_coverage": coverage(observations, state_grid, safe_state_set),
        "state_action_coverage": coverage(joint_samples, joint_grid, safely_excitable_set),
    }


def update_excitation_df(
    df: pd.DataFrame,
    result_paths: list[pathlib.Path],
):
    """Append new excitation experiments and their metric values.

    Args:
        df: Existing dataframe, or None to start a new one.
        result_paths: Paths to excitation experiment directories.

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

        result = load_excitation_result(result_path)

        metrics_jax = evaluate_excitation_result(
            system_name=metadata["system_name"],
            observations=result["observations"],
            actions=result["actions"],
        )

        metrics = to_python(metrics_jax)
        row = dict(
            exp_id=exp_id,
            path=str(result_path),
            system_name=metadata["system_name"],
            model_type=metadata["model_type"],
            excitation_seed=metadata["seed"],
            n_time_steps=metadata["n_time_steps"],
            consider_action_distribution=metadata["params"]["alg_params"]["consider_action_distribution"],
            excitation_objective=metadata["excitation_objective"],
            use_f64=metadata.get("use_f64"),
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


def evaluate_excitation_history(df, *, n_checkpoints=20):
    """Return one row per excitation run and evaluated checkpoint."""
    if n_checkpoints < 2:
        raise ValueError("n_checkpoints must be at least two.")

    rows = []

    for _, run in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        domain = run["evaluation.domain"]
        if domain not in {"safely_excitable", "full_box"}:
            raise ValueError(f"Unknown evaluation domain: {domain!r}")

        with np.load(
            pathlib.Path(run["path"]) / "trajectory.npz",
            allow_pickle=False,
        ) as data:
            observations = data["observations"]
            actions = data["actions"]

        if len(actions) == 0:
            raise ValueError(f"No excitation steps in {run['exp_id']}")

        # More checkpoints early on, while retaining the final step.
        checkpoints = np.unique(np.rint(np.geomspace(1, len(actions), n_checkpoints)).astype(int))

        for k in checkpoints:
            result = evaluate_excitation_result(
                system_name=run["system_name"],
                observations=observations[: k + 1],
                actions=actions[:k],
                filter_testing_points=domain == "safely_excitable",
                points_per_dim=int(run["evaluation.points_per_dim"]),
            )

            rows.append(
                {
                    "exp_id": run["exp_id"],
                    "system_name": run["system_name"],
                    "model_type": run["model_type"],
                    "excitation_objective": run["excitation_objective"],
                    "excitation_seed": run["excitation_seed"],
                    "n_steps": int(k),
                    "validity.all_finite": result["validity"]["all_finite"],
                    "state_coverage.diced_fill": result["state_coverage"]["diced_fill"],
                    "state_action_coverage.diced_fill": result["state_action_coverage"]["diced_fill"],
                }
            )

    return pd.DataFrame.from_records(rows)
