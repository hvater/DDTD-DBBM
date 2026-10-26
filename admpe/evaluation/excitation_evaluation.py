import json
import pathlib

import pandas as pd
import tqdm

from admpe.utils.helper import to_python
from admpe.data_management import list_modeling_results, load_modeling_result, load_excitation_result


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


def evaluate_excitation_result(system_name, observations, actions):
    return {"test_metrics": 0.0}


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
