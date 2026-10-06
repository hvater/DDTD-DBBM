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
from admpe.modeling.model_fitting import fit


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run modeling experiments for one or more seeds.")
    parser.add_argument("--system_name", choices=SUPPORTED_SYSTEMS, required=True)
    parser.add_argument("--dataset_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--model_type", choices=SUPPORTED_MODELS, default="neural_euler")
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
        dataset_path=dataset_path,
    )
    write_metadata(working_path / "experiment.json", metadata)

    started = perf_counter()
    try:
        model, model_trainer, opt_state, loader_key, params = setup_experiment(model_type, seed, params)
        metadata["params"] = params
        write_metadata(working_path / "experiment.json", metadata)

        result = fit(
            train_data,
            model_type,
            model,
            model_trainer,
            opt_state,
            loader_key,
            params,
        )
        save_modeling_result(working_path, result)
        metadata["status"] = "completed"
        metadata["elapsed_seconds"] = perf_counter() - started
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
    callback=None,
    callback_every: int | None = None,
    *,
    model_trainer_overrides=None,
    model_overrides=None,
):
    # validate further inputs
    if gpu_id < -1:
        raise ValueError("gpu_id must be -1 for CPU or a nonnegative GPU index.")

    if callback_every is not None and callback_every < 1:
        raise ValueError("callback_every must be positive.")

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

    # start experiment:
    paths = []
    with jax.default_device(devices[index]):
        # TODO: load validation and test data sets?
        # maybe validation would be sufficient and test data can be done separately?

        for dataset_path in dataset_paths:
            try:
                train_data = load_excitation_result(dataset_path)
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
