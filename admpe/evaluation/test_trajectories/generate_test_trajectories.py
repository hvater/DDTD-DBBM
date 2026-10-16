import argparse
import os
import logging as log
from pathlib import Path
import zlib
import re
from time import perf_counter
from uuid import uuid4

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from admpe.data_management import write_metadata, TESTING_TRAJECTORIES_ROOT
from admpe.excitation.excitation_experiment_setup import SUPPORTED_SYSTEMS, _SYSTEM_SETUP


import optax

from admpe.evaluation.test_trajectories import (
    simple_test_trajectories as simple,
    ocp_test_trajectories as ocp,
)

SUPPORTED_SIGNALS = (
    *simple.SUPPORTED_SIGNALS,
    *ocp.SUPPORTED_SIGNALS,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate feasible datasets for one or more seeds.")
    parser.add_argument("--system_name", choices=SUPPORTED_SYSTEMS, required=True)
    parser.add_argument("--signal_name", choices=SUPPORTED_SIGNALS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--gpu_id", type=int, default=-1, help="GPU index; -1 selects CPU.")
    parser.add_argument("--use_f64", action="store_true")
    parser.add_argument("--n_sequences", type=int, default=100, help="Accepted sequences per seed.")
    parser.add_argument(
        "--n_time_steps",
        type=int,
        default=1001,
        help="Number of observations per sequence, including the initial one.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--deterministic_initial_state", action="store_true")
    parser.add_argument("--noise_variance", type=float, default=None)
    parser.add_argument("--max_attempts", type=int, default=None, help="Optional candidate cap per seed.")
    parser.add_argument("--output_root", type=Path, default=TESTING_TRAJECTORIES_ROOT)
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args(argv)


def setup_experiment(
    system_name,
    signal_name,
    seed,
    n_sequences,
    n_time_steps,
    batch_size,
    amplitude,
    deterministic_initial_state,
    noise_variance,
    max_attempts,
    *,
    signal_overrides=None,
    ocp_overrides=None,
):
    setup_env, _ = _SYSTEM_SETUP[system_name]
    env, penalty_function, featurize, env_params = setup_env()

    if noise_variance is not None:
        # the eval trajectories should generally be noiseless
        env = eqx.tree_at(lambda e: e.process_noise_variance, env, noise_variance)
    if env.process_noise_variance > 0.0:
        log.warning(f"Noise is set to {env.process_noise_variance} for eval sequence generation.")
    env_params["process_noise_variance"] = float(env.process_noise_variance)

    if signal_name == "ocp":
        signal_params = {}
    else:
        signal_params = dict(simple.SIGNALS[signal_name][1])
        signal_params.update(signal_overrides or {})

    if signal_name == "ocp":
        overrides = ocp_overrides or {}
        unknown = set(overrides) - set(ocp.OCP_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown OCP parameters: {sorted(unknown)}")

        params.update({**ocp.OCP_DEFAULTS, **overrides})

        if params["n_starts"] < 1 or params["n_opt_steps"] < 1:
            raise ValueError("OCP starts and optimization steps must be positive.")

        if env.process_noise_variance != 0.0:
            raise ValueError("The current OCP loss requires a noiseless environment.")

    # TODO: is this consistent with the excitation experiments?
    n_steps = n_time_steps - 1
    params = dict(
        seed=seed,
        signal_name=signal_name,
        n_sequences=n_sequences,
        n_time_steps=n_time_steps,
        n_steps=n_steps,
        batch_size=batch_size,
        amplitude=amplitude,
        signal_params=signal_params,
        deterministic_initial_state=deterministic_initial_state,
        max_attempts=max_attempts,
        tau=float(env.tau),
        env_params=env_params,
        environment_properties=env.env_properties,
    )
    # Candidate keys depend on the signal family and candidate ID, not batch size.
    data_key = jax.random.fold_in(jax.random.PRNGKey(seed), zlib.crc32(signal_name.encode()))
    return env, params, penalty_function, featurize, data_key


def save_sequence(path: Path, arrays: dict):
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
    os.replace(temporary, path)


def get_candidate_batch_function(
    env,
    params,
    penalty_function,
    featurize,
    data_key,
):
    if params["signal_name"] in simple.SUPPORTED_SIGNALS:
        generate = simple.get_candidate_batch_function(env, params, penalty_function, data_key)

        def batch(candidate_ids):
            observations, actions, penalties, valid = generate(candidate_ids)
            return {
                "observations": observations,
                "actions": actions,
                "max_penalty": penalties,
            }, valid

        # returns the batched generation function
        return batch

    elif params["signal_name"] in ocp.SUPPORTED_SIGNALS:
        learning_rate = optax.schedules.exponential_decay(
            init_value=params["lr_init"],
            transition_steps=max(1, params["n_opt_steps"] // 3),
            transition_begin=0,
            decay_rate=params["lr_decay_rate"],
            end_value=params["lr_end"],
        )
        optimizer = optax.adam(learning_rate)

        generate = ocp.get_candidate_batch_function(
            env,
            params,
            optimizer,
            featurize,
            ocp.loss_function,
            penalty_function,
            data_key,
            state_tolerance=params["state_tolerance"],
            action_tolerance=params["action_tolerance"],
        )

        def batch(candidate_ids):
            (
                observations,
                actions,
                state_excess,
                action_excess,
                valid,
                _,
                targets,
            ) = generate(candidate_ids)

            return {
                "observations": observations,
                "actions": actions,
                "state_excess": state_excess,
                "action_excess": action_excess,
                "target_observations": targets,
            }, valid

        return batch
    else:
        raise ValueError(f"The signal type '{params["signal"]}' is unknown.")


def generate_testing_dataset(
    system_name,
    signal_name,
    seed,
    base_id,
    n_sequences,
    n_time_steps,
    batch_size,
    output_root,
    amplitude,
    deterministic_initial_state,
    noise_variance,
    max_attempts,
    signal_overrides,
):
    root = Path(output_root)
    incomplete_root = root / "_incomplete"
    incomplete_root.mkdir(parents=True, exist_ok=True)
    exp_id = f"{base_id}-seed{seed}"
    working_path = incomplete_root / exp_id
    final_path = root / exp_id
    if final_path.exists():
        raise FileExistsError(f"Dataset already exists: {final_path}")
    working_path.mkdir(exist_ok=False)
    metadata = dict(
        exp_id=exp_id,
        seed=seed,
        system_name=system_name,
        signal_name=signal_name,
        n_sequences=n_sequences,
        n_time_steps=n_time_steps,
        use_f64=bool(jax.config.jax_enable_x64),
        device=str(jax.config.jax_default_device),
        status="running",
        attempted=0,
        feasible_candidates=0,
        saved_sequences=0,
        constraint="finite observations/actions/penalties and penalty_function <= 0",
    )
    write_metadata(working_path / "experiment.json", metadata)
    started = perf_counter()
    try:
        env, params, penalty_function, featurize, data_key = setup_experiment(
            system_name,
            signal_name,
            seed,
            n_sequences,
            n_time_steps,
            batch_size,
            amplitude,
            deterministic_initial_state,
            noise_variance,
            max_attempts,
            signal_overrides=signal_overrides,
        )

        batch_candidate_function = get_candidate_batch_function(
            env,
            params,
            penalty_function,
            featurize,
            data_key,
        )

        log.info("Starting dataset %s", exp_id)
        while metadata["saved_sequences"] < n_sequences:
            # uint32 candidate IDs must stay unique within this seed.
            limit = 2**32 if max_attempts is None else max_attempts
            count = min(batch_size, limit - metadata["attempted"])
            if count <= 0:
                raise RuntimeError(
                    f"Saved {metadata['saved_sequences']}/{n_sequences} sequences "
                    f"after {metadata['attempted']} attempts; candidate limit reached."
                )
            candidate_ids = np.arange(metadata["attempted"], metadata["attempted"] + count, dtype=np.uint32)

            fields, valid = batch_candidate_function(jnp.asarray(candidate_ids))
            fields, valid = jax.device_get((fields, valid))
            valid = np.asarray(valid)

            metadata["attempted"] += count
            metadata["feasible_candidates"] += int(valid.sum())

            # count number of sequences remaining until 'full'
            remaining = n_sequences - metadata["saved_sequences"]

            # store all successful sequences until n_sequences is 'full'
            for index in np.flatnonzero(valid)[:remaining]:
                sequence = {name: np.asarray(values[index]) for name, values in fields.items()}
                sequence["candidate_id"] = candidate_ids[index]

                path = working_path / f"sequence_{metadata['saved_sequences']:06d}.npz"
                save_sequence(path, sequence)
                metadata["saved_sequences"] += 1

            metadata["elapsed_seconds"] = perf_counter() - started
            write_metadata(working_path / "experiment.json", metadata)
            log.info(
                "%s: saved %s/%s; feasible %s/%s candidates",
                exp_id,
                metadata["saved_sequences"],
                n_sequences,
                metadata["feasible_candidates"],
                metadata["attempted"],
            )
        metadata["status"] = "combining"
        write_metadata(working_path / "experiment.json", metadata)
        dataset_path = combine_sequences(working_path, n_sequences)
        metadata["dataset_file"] = dataset_path.name
        metadata["status"] = "completed"
        metadata["elapsed_seconds"] = perf_counter() - started
        write_metadata(working_path / "experiment.json", metadata)
    except BaseException as error:
        metadata["status"] = "failed" if isinstance(error, Exception) else "interrupted"
        metadata["error"] = f"{type(error).__name__}: {error}"
        metadata["elapsed_seconds"] = perf_counter() - started
        write_metadata(working_path / "experiment.json", metadata)
        raise
    try:
        working_path.rename(final_path)
    except OSError:
        log.exception(
            "Dataset %s completed, but could not be moved to %s. Results remain at %s.",
            exp_id,
            final_path,
            working_path,
        )
        raise
    log.info("Saved dataset to %s", final_path)
    return final_path


def run_testing_dataset_experiment(
    system_name: str,
    signal_name: str,
    seed: int = 0,
    exp_name: str | None = None,
    use_f64: bool = False,
    gpu_id: int = -1,
    n_sequences: int = 100,
    n_time_steps: int = 1001,
    batch_size: int = 32,
    output_root: Path = TESTING_TRAJECTORIES_ROOT,
    *,
    amplitude: float = 1.0,
    deterministic_initial_state: bool = False,
    noise_variance: float | None = 0.0,
    max_attempts: int | None = int(1e6),
    signal_overrides=None,
) -> Path:
    if gpu_id < -1:
        raise ValueError("gpu_id must be -1 for CPU or a nonnegative GPU index.")
    if exp_name is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", exp_name):
        raise ValueError("exp_name may contain only letters, digits, underscores, dots, and hyphens.")
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be integers in [0, 2**32).")

    jax.config.update("jax_enable_x64", use_f64)
    devices = jax.devices("cpu" if gpu_id == -1 else "gpu")
    index = 0 if gpu_id == -1 else gpu_id

    if index >= len(devices):
        raise ValueError(f"GPU index {gpu_id} is unavailable; found {len(devices)} GPU(s).")

    tag = f"-{exp_name}" if exp_name else ""
    base_id = f"{system_name}-{signal_name}{tag}-{uuid4().hex[:8]}"
    with jax.default_device(devices[index]):
        try:
            if output_root is TESTING_TRAJECTORIES_ROOT:
                output_root = (
                    ocp.OCP_TESTING_SIGNALS_ROOT if signal_name == "ocp" else simple.SIMPLE_TESTING_SIGNALS_ROOT
                )
            final_path = generate_testing_dataset(
                system_name=system_name,
                signal_name=signal_name,
                seed=seed,
                base_id=base_id,
                n_sequences=n_sequences,
                n_time_steps=n_time_steps,
                batch_size=batch_size,
                output_root=output_root,
                amplitude=amplitude,
                deterministic_initial_state=deterministic_initial_state,
                noise_variance=noise_variance,
                max_attempts=max_attempts,
                signal_overrides=signal_overrides,
            )
            return final_path
        except Exception:
            log.exception("Dataset generation failed for seed %s", seed)
            raise
        finally:
            jax.clear_caches()


def combine_sequences(path: Path, n_sequences: int) -> Path:
    """Stack sequence archives into dataset.npz, then remove the source archives."""

    source_files = [path / f"sequence_{index:06d}.npz" for index in range(n_sequences)]
    names = None
    collected = {}

    for source_file in source_files:
        with np.load(source_file, allow_pickle=False) as sequence:
            if names is None:
                names = tuple(sequence.files)
                collected = {name: [] for name in names}
            elif set(sequence.files) != set(names):
                raise ValueError(f"Inconsistent fields in {source_file}")

            for name in names:
                collected[name].append(sequence[name])

    destination = path / "dataset.npz"
    save_sequence(
        destination,
        {name: np.stack(values, axis=0) for name, values in collected.items()},
    )

    for source_file in source_files:
        source_file.unlink()

    return destination
