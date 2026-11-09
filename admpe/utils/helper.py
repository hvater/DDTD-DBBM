import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import jax
import equinox as eqx
import jax.numpy as jnp

import exciting_environments as excenvs
from admpe.pmsm import NominalSpeedPMSM


def convert_lists(x):
    if isinstance(x, dict):
        return {k: convert_lists(v) for k, v in x.items()}
    if isinstance(x, list):
        try:
            a = np.asarray(x)
            if np.issubdtype(a.dtype, np.number):
                return a
        except (TypeError, ValueError, OverflowError):
            return [convert_lists(v) for v in x]
    if isinstance(x, tuple):
        return tuple(convert_lists(v) for v in x)
    return x


def to_python(value):
    """Transforms all jax arrays to floats or lists for storing in the results df."""

    if isinstance(value, dict):
        return {key: to_python(item) for key, item in value.items()}

    if isinstance(value, jax.Array):
        array = np.asarray(value)
        return float(array.item()) if array.ndim == 0 else array.tolist()

    return value


def pprint_helper(x):
    return eqx.tree_pprint(
        convert_lists(x),
        width=120,
    )


def get_plot_labels(system_name: str) -> dict[str, list[str]]:
    definitions = {
        "cart_pole": (
            (
                r"Cart position $\tilde{x}$",
                r"Cart velocity $\tilde{v}$",
                r"Pole angle $\tilde{\theta}$",
                r"Angular velocity $\tilde{\omega}$",
            ),
            (r"Force $\tilde{F}$",),
        ),
        "pendulum": (
            (
                r"Angle $\tilde{\theta}$",
                r"Angular velocity $\tilde{\omega}$",
            ),
            (r"Torque $\tilde{T}$",),
        ),
        "fluid_tank": (
            (r"Water level $\tilde{h}$",),
            (r"Inflow $\tilde{q}_{\mathrm{in}}$",),
        ),
        "pmsm_nominal": (
            (
                r"$\tilde{i}_\mathrm{d}$",
                r"$\tilde{i}_\mathrm{q}$",
            ),
            (
                r"$\tilde{v}_\mathrm{d}$",
                r"$\tilde{v}_\mathrm{q}$",
            ),
        ),
    }

    if system_name not in definitions:
        raise ValueError(f"Unknown system {system_name!r}. " f"Choose from {tuple(definitions)}.")

    observations, actions = map(list, definitions[system_name])

    # Model time-series plots use the existing sin/cos featurization.
    if system_name == "cart_pole":
        featurized = [
            *observations[:2],
            r"$\sin(\theta)$",
            r"$\cos(\theta)$",
            observations[3],
        ]
    elif system_name == "pendulum":
        featurized = [
            r"$\sin(\theta)$",
            r"$\cos(\theta)$",
            observations[1],
        ]
    else:
        featurized = observations.copy()

    return {
        "observations": observations,
        "actions": actions,
        "both": observations + actions,
        "featurized_observations": featurized,
    }


def get_state_difference_for_env(env):
    if isinstance(env, (excenvs.FluidTank, NominalSpeedPMSM)):
        return jnp.subtract
    elif isinstance(env, excenvs.Pendulum):

        def norm_pendulum_difference(next_state, state):
            delta = next_state - state
            return delta.at[0].set((delta[0] + 1.0) % 2.0 - 1.0)

        return norm_pendulum_difference

    elif isinstance(env, excenvs.CartPole):

        def norm_cart_pole_difference(next_state, state):
            delta = next_state - state
            return delta.at[2].set((delta[2] + 1.0) % 2.0 - 1.0)

        return norm_cart_pole_difference
    else:
        raise NotADirectoryError("No state difference defined for env with type:", type(env))


def missing_batches(system_name, model_type, exp_name, seeds):

    from admpe.data_management import list_excitation_results, list_modeling_results

    completed = set()

    for result in list_modeling_results(
        system_name=system_name,
        model_type=model_type,
        exp_name=exp_name,
        status="completed",
    ):
        path = Path(result["path"])
        if not (path / "model.eqx").is_file():
            continue

        metadata = json.loads((path / "experiment.json").read_text())
        saved = str(metadata["dataset_path"])
        saved = saved.removeprefix("PosixPath('").removesuffix("')")
        completed.add((Path(saved).name, int(metadata["seed"])))

    groups = defaultdict(list)

    for result in list_excitation_results(
        system_name=system_name,
        model_type="perfect",
        exp_name="submission_nov",
        status="completed",
    ):
        path = Path(result["path"])
        missing = tuple(seed for seed in seeds if (path.name, seed) not in completed)
        if missing:
            groups[missing].append(path)

    return groups
