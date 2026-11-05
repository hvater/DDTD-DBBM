"""See, e.g.,

@book{Nelles2020,
  title={Nonlinear system identification},
  author={Nelles, Oliver},
  year={2020},
  publisher={Springer},
  doi={10.1007/978-3-030-47439-3},
}

for descriptions of the utilized signals.

Except for the quintic smoothing of the APRBS which is based on qunitic interpolation, see, e.g.:

@book{Lynch2017,
  title={Modern Robotics: Mechanics, Planning, and Control},
  author={Lynch, Kevin M. and Park, Frank C.},
  year={2017},
  publisher={Cambridge University Press},
}

and more specifically, within the accompanying code:
https://github.com/NxRLab/ModernRobotics/blob/master/packages/MATLAB/mr/QuinticTimeScaling.m

"""

import jax
import jax.numpy as jnp


def aprbs(
    n_steps: int,
    t_min: int,
    t_max: int,
    batch_size: int,
    key: jax.random.PRNGKey,
    *,
    scaling: float = 1.0,
    n_actions: int = 1,
    transition_type: str = "step",
    transition_steps: int = 1,
):
    if not 1 <= t_min < t_max:
        raise ValueError("Require 1 <= t_min < t_max; t_max is exclusive.")

    if transition_type not in ("step", "linear", "cosine", "quintic"):
        raise ValueError(f"Unknown transition: {transition_type}")

    if transition_type != "step" and not 0 < transition_steps < t_min:
        raise ValueError("Require 0 < transition_steps < t_min.")

    n_segments = (n_steps + t_min - 1) // t_min + 1
    keys = jax.random.split(key, batch_size * n_actions)

    time = jnp.arange(n_steps)

    def one_signal(signal_key):
        dwell_key, level_key = jax.random.split(signal_key)
        dwell = jax.random.randint(dwell_key, (n_segments - 1,), t_min, t_max)
        levels = jax.random.uniform(level_key, (n_segments,), minval=-1.0, maxval=1.0)

        switches = jnp.cumsum(dwell)

        if transition_type == "step":
            index = jnp.searchsorted(switches, time, side="right")
            return levels[index]

        # Each transition begins half its duration before the switch, i.e., the switch
        # would occur at the middle of the transition for a step aprbs
        starts = switches - transition_steps / 2

        # get targeted index and level + previous index and level
        index = jnp.searchsorted(starts, time, side="right")
        previous_index = jnp.maximum(index - 1, 0)
        previous = levels[previous_index]
        target = levels[index]

        # fraction of transition completed
        phase = jnp.clip(
            (time - starts[previous_index]) / transition_steps,
            0.0,
            1.0,
        )

        if transition_type == "linear":
            weight = phase
        elif transition_type == "cosine":
            weight = 0.5 * (1.0 - jnp.cos(jnp.pi * phase))
        else:
            weight = phase**3 * (10.0 + phase * (-15.0 + 6.0 * phase))

        weight = jnp.clip(weight, 0.0, 1.0)
        return (1.0 - weight) * previous + weight * target

    signals = jax.vmap(one_signal)(keys) * scaling
    return signals.reshape(batch_size, n_actions, n_steps).transpose(0, 2, 1)


def ramp():
    raise NotImplementedError()


def multisine(
    n_steps: int,
    fundamental_frequency: float,
    n_sines: int,
    batch_size: int,
    key: jax.random.PRNGKey,
    *,
    scaling: float = 1.0,
    n_actions: int = 1,
    debug: bool = False,
) -> jax.Array:
    """The fundamental frequency measure cycles per sample and is not given in Hz."""

    assert n_sines * fundamental_frequency < 0.5, "frequencies must stay below the Nyquist frequency"

    time = jnp.arange(n_steps)
    frequencies = jnp.arange(1, n_sines + 1, 1) * fundamental_frequency

    keys = jax.random.split(key, batch_size * n_actions)

    def one_signal(signal_key):
        # draw phase_shift and amplitudes
        ampl_key, phase_key = jax.random.split(signal_key, 2)
        amplitudes = jax.random.uniform(ampl_key, shape=(1, n_sines), minval=0, maxval=1)
        phase_shifts = jax.random.uniform(phase_key, shape=(1, n_sines), minval=-1, maxval=1) * jnp.pi

        sines = amplitudes * jnp.sin((2 * jnp.pi * frequencies[None] * time[:, None]) + phase_shifts)
        actions = jnp.mean(sines, axis=-1)

        return actions, sines

    actions, sines = jax.vmap(one_signal)(keys)
    sines = sines.reshape(batch_size, n_actions, n_steps, n_sines).transpose(0, 2, 1, 3)
    actions = actions.reshape(batch_size, n_actions, n_steps).transpose(0, 2, 1) * scaling

    if debug:
        return actions, sines

    return actions


def chirp():
    raise NotImplementedError()


# def random_walk():
#     raise NotImplementedError()
