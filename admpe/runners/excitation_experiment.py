"""Run excitation algorithm either directly from Python or with ``python -m admpe.runners.excitation_experiment``."""

import argparse
import logging as log
import os
from pathlib import Path
import re
from time import perf_counter
from uuid import uuid4

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

from dmpe.algorithms.algorithms import excite_with_dmpe

from admpe.excitation.gp_excitation.algorithm import excite_with_gp_uncertainty
from admpe.data_management import (
    EXCITATION_EXPERIMENTS_ROOT,
    save_excitation_result,
    write_metadata,
)
from admpe.excitation.excitation_experiment_setup import (
    SUPPORTED_MODELS,
    SUPPORTED_OBJECTIVES,
    SUPPORTED_SYSTEMS,
    setup_experiment,
    validate_experiment,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run excitation experiments for one or more seeds.")
    parser.add_argument("--system_name", choices=SUPPORTED_SYSTEMS, required=True)
    parser.add_argument("--excitation_objective", choices=SUPPORTED_OBJECTIVES, required=True)
    parser.add_argument("--model_type", choices=SUPPORTED_MODELS, default="neural_euler")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--gpu_id", type=int, default=-1, help="GPU index; -1 selects CPU.")
    parser.add_argument("--use_f64", action="store_true")
    parser.add_argument(
        "--n_time_steps", type=int, default=7_000, help="Number of observations, including the initial one."
    )
    # parser.add_argument("--points_per_dim", type=int, default=10)
    parser.add_argument("--output_root", type=Path, default=EXCITATION_EXPERIMENTS_ROOT)
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args(argv)


def run_experiment_for_seed(
    seed: int,
    base_id: str,
    system_name: str,
    excitation_objective: str,
    model_type: str,
    n_time_steps: int = 7_000,
    noise_variance: float | None = None,
    constrain_to_ses: bool = False,
    output_root: Path = EXCITATION_EXPERIMENTS_ROOT,
    callback=None,
    callback_every: int | None = None,
    *,
    alg_overrides=None,
    model_trainer_overrides=None,
    model_overrides=None,
) -> Path:
    """Build, execute, and persist one trial; return its artifact directory.

    Device and precision are configured by ``run_excitation_experiment``.
    A failed trial keeps metadata with its error and resolved parameters.
    """
    root = Path(output_root)
    incomplete_root = root / "_incomplete"
    incomplete_root.mkdir(parents=True, exist_ok=True)

    # generate and store metadata
    exp_id = f"{base_id}-seed{seed}"
    working_path = incomplete_root / exp_id
    final_path = root / exp_id
    working_path.mkdir(exist_ok=False)
    metadata = dict(
        exp_id=exp_id,
        seed=seed,
        system_name=system_name,
        excitation_objective=excitation_objective,
        model_type=model_type,
        n_time_steps=n_time_steps,
        use_f64=bool(jax.config.jax_enable_x64),
        device=str(jax.config.jax_default_device),
        status="running",
    )
    write_metadata(working_path / "experiment.json", metadata)

    started = perf_counter()

    log.info(
        "Preparing run %s: system=%s; objective=%s; model_type=%s; excitation_seed=%d; "
        "n_time_steps=%d; device=%s; precision=%s; constrain_to_ses=%s",
        exp_id,
        system_name,
        excitation_objective,
        model_type,
        seed,
        n_time_steps,
        metadata["device"],
        "float64" if metadata["use_f64"] else "float32",
        constrain_to_ses,
    )

    try:
        env, params, proposed_actions, loader_key, expl_key, process_noise_key = setup_experiment(
            system_name,
            excitation_objective,
            model_type,
            seed,
            n_time_steps,
            noise_variance,
            constrain_to_ses,
            alg_overrides=alg_overrides,
            model_trainer_overrides=model_trainer_overrides,
            model_overrides=model_overrides,
        )
        metadata["params"] = params
        write_metadata(working_path / "experiment.json", metadata)

        if excitation_objective.startswith("gp_"):
            excite = excite_with_gp_uncertainty
        else:
            excite = excite_with_dmpe

        alg_params = params["alg_params"]
        target = alg_params.get("target_distribution")
        log.info(
            "Starting run %s: preparation_seconds=%.2f; obs_dim=%d; action_dim=%d; "
            "horizon=%s; opt_steps=%s; starts=%s; points_per_dim=%s; bandwidth=%s; "
            "consider_actions=%s; target_shape=%s",
            exp_id,
            perf_counter() - started,
            env.obs_dim,
            env.action_dim,
            alg_params.get("n_prediction_steps"),
            alg_params.get("n_opt_steps"),
            alg_params.get("n_starts"),
            alg_params.get("points_per_dim"),
            alg_params.get("bandwidth"),
            alg_params.get("consider_action_distribution"),
            None if target is None else target.shape,
        )
        result = excite(
            env,
            params,
            proposed_actions,
            loader_key,
            expl_key,
            process_noise_key,
            callback=callback,
            callback_every=callback_every,
        )
        save_excitation_result(working_path, result)
        metadata["status"] = "completed"
        metadata["elapsed_seconds"] = perf_counter() - started
        write_metadata(working_path / "experiment.json", metadata)

    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        metadata["elapsed_seconds"] = perf_counter() - started
        write_metadata(working_path / "experiment.json", metadata)
        raise

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
    log.info(
        "Completed run %s: elapsed_seconds=%.2f; observations_shape=%s; actions_shape=%s; saved_to=%s",
        exp_id,
        metadata["elapsed_seconds"],
        result[0].shape,
        result[1].shape,
        final_path,
    )
    return final_path


def run_excitation_experiment(
    system_name: str,
    excitation_objective: str,
    seeds: list[int],
    model_type: str = "neural_euler",
    exp_name: str | None = None,
    use_f64: bool = False,
    gpu_id: int = -1,
    n_time_steps: int = 7000,
    noise_variance: float | None = None,
    constrain_to_ses: bool = False,
    output_root: Path = EXCITATION_EXPERIMENTS_ROOT,
    continue_on_error: bool = False,
    callback=None,
    callback_every: int | None = None,
    *,
    alg_overrides=None,
    model_trainer_overrides=None,
    model_overrides=None,
) -> list[Path]:
    """Run seeds sequentially and return directories of successful trials.

    Empty seeds default to [0]. Failures propagate unless continue_on_error
    is enabled; failed trials still record their errors in experiment.json.
    Callbacks follow the selected algorithm's callback API.
    """
    validate_experiment(system_name, excitation_objective, model_type, n_time_steps)

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

    base_id = f"{system_name}-{excitation_objective}-{model_type}{tag}-{uuid4().hex[:8]}"
    batch_started = perf_counter()
    log.info(
        "Starting excitation experiments: system=%s; objective=%s; model_type=%s; "
        "excitation_seeds=%s; total_trials=%d; n_time_steps=%d; device=%s; precision=%s; "
        "constrain_to_ses=%s",
        system_name,
        excitation_objective,
        model_type,
        seeds_to_run,
        len(seeds_to_run),
        n_time_steps,
        devices[index],
        "float64" if use_f64 else "float32",
        constrain_to_ses,
    )

    # start experiment:
    paths = []
    with jax.default_device(devices[index]):
        for seed_idx, seed in enumerate(seeds_to_run, start=1):
            log.info(
                "Excitation trial %d/%d: excitation_seed=%d",
                seed_idx,
                len(seeds_to_run),
                seed,
            )
            try:
                paths.append(
                    run_experiment_for_seed(
                        seed=seed,
                        base_id=base_id,
                        system_name=system_name,
                        excitation_objective=excitation_objective,
                        model_type=model_type,
                        n_time_steps=n_time_steps,
                        noise_variance=noise_variance,
                        constrain_to_ses=constrain_to_ses,
                        output_root=output_root,
                        callback=callback,
                        callback_every=callback_every,
                        alg_overrides=alg_overrides,
                        model_trainer_overrides=model_trainer_overrides,
                        model_overrides=model_overrides,
                    )
                )
            except Exception:
                log.exception(
                    "Excitation failed: system=%s; objective=%s; model_type=%s; excitation_seed=%d",
                    system_name,
                    excitation_objective,
                    model_type,
                    seed,
                )
                if not continue_on_error:
                    raise
            finally:
                jax.clear_caches()

    log.info(
        "Completed %d/%d excitation trials in %.2fs: system=%s; objective=%s; model_type=%s",
        len(paths),
        len(seeds_to_run),
        perf_counter() - batch_started,
        system_name,
        excitation_objective,
        model_type,
    )
    return paths


if __name__ == "__main__":
    run_excitation_experiment(**vars(parse_args()))
