import numpy as np
import equinox as eqx
import jax.numpy as jnp

import exciting_environments as excenvs


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
    if isinstance(env, excenvs.FluidTank):
        return jnp.substract
    elif isinstance(env, excenvs.Pendulum):

        def norm_pendulum_difference(next_state, state):
            delta = next_state - state
            return delta.at[0].set((delta[0] + 1.0) % 2.0 - 1.0)

        return norm_pendulum_difference

    elif isinstance(env, excenvs.CartPole):

        def norm_cart_pole_difference(next_state, state):
            delta = next_state - state
            return delta.at[2].set((delta[0] + 1.0) % 2.0 - 1.0)

        return norm_cart_pole_difference
    else:
        raise NotADirectoryError("No state difference defined for env with type:", type(env))
