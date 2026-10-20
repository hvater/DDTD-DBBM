import argparse
import logging as log
import os
from pathlib import Path
import re
from time import perf_counter
from uuid import uuid4

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

from admpe.data_management import (
    MODELING_EXPERIMENTS_ROOT,
    write_metadata,
    save_modeling_result,
    load_excitation_result,
)
from admpe.excitation.excitation_experiment_setup import setup_environment

from admpe.modeling.modeling_experiment_setup import (
    SUPPORTED_MODELS,
    SUPPORTED_SYSTEMS,
    setup_parameterization,
    setup_experiment,
)
from admpe.modeling.modeling_parameterization import parse_neural_euler_variant
from admpe.modeling.model_fitting import fit
from admpe.modeling.validation import get_validation_data


def model_type_argument(value: str) -> str:
    if value in SUPPORTED_MODELS:
        return value

    try:
        parse_neural_euler_variant(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error

    return value


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run modeling experiments for one or more seeds.")
    parser.add_argument("--system_name", choices=SUPPORTED_SYSTEMS, required=True)
    parser.add_argument("--dataset_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--model_type",
        type=model_type_argument,
        default="neural_euler",
    )
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--gpu_id", type=int, default=-1)
    parser.add_argument("--use_f64", action="store_true")
    parser.add_argument("--output_root", type=Path, default=MODELING_EXPERIMENTS_ROOT)
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args(argv)


def run_experiment_for_seed(
    train_data,
    dataset_path,
    system_name,
    model_type,
    seed,
    base_id,
    params,
    output_root,
    validation_data,
    validation_every,
    selection_metric,
):
    root = Path(output_root)
    incomplete_root = root / "_incomplete"
    incomplete_root.mkdir(parents=True, exist_ok=True)

    # generate and store metadata
    exp_id = f"{base_id}-seed{seed}"
    working_path = incomplete_root / exp_id
    final_path = root / exp_id
    working_path.mkdir(parents=True, exist_ok=False)

    assert dataset_path.name == train_data["metadata"]["exp_id"]

    metadata = dict(
        exp_id=exp_id,
        seed=seed,
        system_name=system_name,
        model_type=model_type,
        use_f64=bool(jax.config.jax_enable_x64),
        device=str(jax.config.jax_default_device),
        status="running",
        dataset_path=str(dataset_path),
        excitation_objective=train_data["metadata"]["excitation_objective"],
    )
    write_metadata(working_path / "experiment.json", metadata)

    started = perf_counter()
    try:
        model, model_trainer, opt_state, loader_key, params = setup_experiment(model_type, seed, params)
        metadata["params"] = params
        write_metadata(working_path / "experiment.json", metadata)

        log.info(
            "Starting run %s: modeling_seed=%d; dataset=%s; objective=%s; device=%s; precision=%s",
            exp_id,
            seed,
            dataset_path.resolve(),
            metadata["excitation_objective"],
            metadata["device"],
            "float64" if metadata["use_f64"] else "float32",
        )

        result, validation_summary = fit(
            train_data,
            model_type,
            model,
            model_trainer,
            opt_state,
            loader_key,
            params,
            validation_data=validation_data,
            validation_every=validation_every,
            selection_metric=selection_metric,
        )
        save_modeling_result(working_path, result)
        metadata["status"] = "completed"
        metadata["elapsed_seconds"] = perf_counter() - started

        metadata["validation"] = {
            "every": validation_every,
            **validation_summary,
        }

        write_metadata(working_path / "experiment.json", metadata)

    except Exception as error:
        metadata.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            elapsed_seconds=perf_counter() - started,
        )
        write_metadata(working_path / "experiment.json", metadata)
        raise

    # move modeling artifacts from '_incomplete' to final output folder
    try:
        working_path.rename(final_path)
    except OSError:
        log.exception(
            "Experiment %s completed, but could not be moved to %s. " "Results remain at %s.",
            exp_id,
            final_path,
            working_path,
        )
        raise
    log.info("Saved experiment to %s", final_path)
    return final_path


def run_modeling_experiment(
    system_name: str,
    dataset_paths: list[str],
    seeds: list[int],
    model_type: str = "neural_euler",
    exp_name: str | None = None,
    use_f64: bool = False,
    gpu_id: int = -1,
    output_root: Path = MODELING_EXPERIMENTS_ROOT,
    continue_on_error: bool = False,
    perform_validation: bool = False,
    validation_every: int = 50,
    selection_metric: str = "one_step_mse",
    *,
    model_trainer_overrides=None,
    model_overrides=None,
):
    # validate further inputs
    if gpu_id < -1:
        raise ValueError("gpu_id must be -1 for CPU or a nonnegative GPU index.")

    if exp_name is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", exp_name):
        raise ValueError("exp_name may contain only letters, digits, underscores, dots, and hyphens.")

    seeds_to_run = list(seeds) if seeds else [0]
    if any(not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in seeds_to_run):
        raise ValueError("seeds must be integers in [0, 2**32).")

    if len(set(seeds_to_run)) != len(seeds_to_run):
        raise ValueError("seeds must be unique within an experiment.")

    jax.config.update("jax_enable_x64", use_f64)
    devices = jax.devices("cpu" if gpu_id == -1 else "gpu")
    index = 0 if gpu_id == -1 else gpu_id
    if index >= len(devices):
        raise ValueError(f"GPU index {gpu_id} is unavailable; found {len(devices)} GPU(s).")

    tag = f"-{exp_name}" if exp_name else ""

    log.info(
        "Starting modeling experiments: system=%s; model_type=%s; #datasets=%d; modeling_seeds=%s; total_trials=%d; device=%s",
        system_name,
        model_type,
        len(dataset_paths),
        seeds_to_run,
        len(dataset_paths) * len(seeds_to_run),
        devices[index],
    )

    # start experiment:
    paths = []
    with jax.default_device(devices[index]):

        validation_data = get_validation_data(system_name) if perform_validation else None

        for dataset_idx, dataset_path in enumerate(dataset_paths, start=1):
            dataset_path = Path(dataset_path)
            log.info(
                "Loading excitation dataset %d/%d: %s",
                dataset_idx,
                len(dataset_paths),
                dataset_path.resolve(),
            )

            try:
                train_data = load_excitation_result(dataset_path)

                log.info(
                    "Loaded excitation dataset: system=%s; objective=%s; "
                    "excitation_seed=%s; transitions=%d; observations_shape=%s; actions_shape=%s",
                    train_data["metadata"]["system_name"],
                    train_data["metadata"]["excitation_objective"],
                    train_data["metadata"]["seed"],
                    train_data["actions"].shape[0],
                    train_data["observations"].shape,
                    train_data["actions"].shape,
                )

                _, _, featurize, _, model_class = setup_environment(system_name)
                params = setup_parameterization(
                    train_data,
                    model_type,
                    featurize,
                    model_class,
                    model_overrides=model_overrides,
                    model_trainer_overrides=model_trainer_overrides,
                )

            except:
                log.exception("Could not prepare dataset %s", dataset_path)
                if not continue_on_error:
                    raise
                continue

            assert system_name == train_data["metadata"]["system_name"], (
                f"Dataset is from system {train_data["metadata"]["system_name"]}"
                + " while the target system is {system_name}."
            )

            base_id = f"modeling-{system_name}-{model_type}{tag}-{uuid4().hex[:8]}"
            for seed in seeds_to_run:
                try:
                    paths.append(
                        run_experiment_for_seed(
                            train_data=train_data,
                            dataset_path=dataset_path,
                            system_name=system_name,
                            model_type=model_type,
                            seed=seed,
                            base_id=base_id,
                            params=params,
                            output_root=output_root,
                            validation_data=validation_data,
                            validation_every=validation_every,
                            selection_metric=selection_metric,
                        )
                    )
                except Exception:
                    log.exception("Modeling failed: dataset=%s seed=%d", dataset_path, seed)
                    if not continue_on_error:
                        raise
                finally:
                    jax.clear_caches()

    log.info(
        "Completed %s of %s trials",
        len(paths),
        len(dataset_paths) * len(seeds_to_run),
    )
    return paths


if __name__ == "__main__":
    run_modeling_experiment(**vars(parse_args()))
