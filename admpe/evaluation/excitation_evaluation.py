import json
from pathlib import Path

from admpe.data_management import list_modeling_results, load_modeling_result


def load_corresponding_modeling_results(
    excitation_result: dict,
    path: Path | None = None,
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
        if Path(dataset_path).name == excitation_id:
            results.append(load_modeling_result(result_path["path"]))

    return results
