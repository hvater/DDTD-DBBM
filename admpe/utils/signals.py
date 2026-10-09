"""See, e.g.,

```
@book{Nelles2020,
  title={Nonlinear system identification},
  author={Nelles, Oliver},
  year={2020},
  publisher={Springer},
  doi={10.1007/978-3-030-47439-3},
}

for descriptions of the utilized signals.
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
    n_actions: int = 1,
    transition_type: str = "step",
    transition_steps: int = 0,
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
        else:
            raise NotImplementedError()

    signals = jax.vmap(one_signal)(keys)
    return signals.reshape(batch_size, n_actions, n_steps).transpose(0, 2, 1)


def ramp():
    raise NotImplementedError()


def multisine(
    n_steps: int,
    fundamental_frequency: float,
    n_sines: int,
    batch_size: int,
    key: jax.random.PRNGKey,
    debug: bool = False,
) -> jax.Array:

    t = jnp.arange(0, n_steps, 1)
    frequencies = jnp.arange(1, n_sines + 1, 1) * fundamental_frequency

    # draw phase_shift and amplitudes
    ampl_key, phase_key = jax.random.split(key, 2)
    amplitudes = jax.random.uniform(ampl_key, shape=(batch_size, 1, n_sines), minval=-1, maxval=1)
    phase_shifts = jax.random.uniform(phase_key, shape=(batch_size, 1, n_sines), minval=-1, maxval=1) * jnp.pi

    sines = amplitudes * jnp.sin((frequencies[None] * t[:, None])[None] + phase_shifts)
    actions = jnp.mean(sines, axis=-1, keepdims=True)

    if debug:
        return actions, sines

    return actions


def chirp():
    raise NotImplementedError()


# def random_walk():
#     raise NotImplementedError()
