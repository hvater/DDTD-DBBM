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
    transition_type: str = "step",
    transition_steps: int = 0,
):
    if transition_type not in ("step", "linear", "cosine", "quintic"):
        raise ValueError(f"Unknown transition: {transition_type}")

    raise NotImplementedError()


def ramp():
    raise NotImplementedError()


def multisine(
    n_steps: int, fundamental_frequency: float, n_sines: int, batch_size: int, key: jax.random.PRNGKey
) -> jax.Array:

    t = jnp.arange(0, n_steps, 1)
    frequencies = jnp.arange(1, n_sines + 1, 1) * fundamental_frequency

    # draw phase_shift and amplitudes
    ampl_key, phase_key = jax.random.split(key, 2)

    amplitudes = jax.random.uniform(ampl_key, shape=(batch_size, 1, n_sines), minval=-1, maxval=1)
    phase_shifts = jax.random.uniform(phase_key, shape=(batch_size, 1, n_sines), minval=-1, maxval=1) * jnp.pi

    actions = jnp.mean(amplitudes * jnp.sin((frequencies[None] * t[:, None])[None] + phase_shifts), axis=-1)

    return actions[..., None], amplitudes * jnp.sin((frequencies[None] * t[:, None])[None] + phase_shifts)


def chirp():
    raise NotImplementedError()


# def random_walk():
#     raise NotImplementedError()
