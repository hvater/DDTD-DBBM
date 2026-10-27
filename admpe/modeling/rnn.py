from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx


class GRU(eqx.Module):
    """GRU based state-space model."""

    hidden_size: int = eqx.field(static=True)
    cell: eqx.Module
    obs_dim: int

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int, *, key):

        if hidden_size < obs_dim:
            raise ValueError("'hidden_size' must be greater than 'obs_dim'")

        self.hidden_size = hidden_size
        self.cell = eqx.nn.GRUCell(action_dim, hidden_size, key=key)
        self.obs_dim = obs_dim

    def step(self, action, carry):
        return self.cell(action, carry)

    def __call__(self, init_obs: jax.Array, actions: jax.Array, tau: float) -> jax.Array:

        hidden = jnp.hstack([init_obs, jnp.zeros(self.hidden_size - self.obs_dim)])

        def f(carry, action):
            rnn_out = self.step(action, carry)
            pred_observation = rnn_out[: self.obs_dim]
            return rnn_out, pred_observation

        _, pred_observations = jax.lax.scan(f, hidden, actions)
        return jnp.concatenate([init_obs[None], pred_observations], axis=0)


class GRUPendulum(GRU):

    def step(self, action, carry):
        rnn_out = super().step(action, carry)

        next_obs = rnn_out[: self.obs_dim]
        wrapped_obs = jnp.stack([(((next_obs[..., 0] + 1) % 2) - 1), next_obs[..., 1]], axis=-1)
        return jnp.hstack([wrapped_obs, rnn_out[self.obs_dim :]])


class GRUCartpole(GRU):

    def step(self, action, carry):
        rnn_out = super().step(action, carry)

        next_obs = rnn_out[: self.obs_dim]
        wrapped_obs = jnp.stack(
            [next_obs[..., 0], next_obs[..., 1], (((next_obs[..., 2] + 1) % 2) - 1), next_obs[..., 3]], axis=-1
        )
        return jnp.hstack([wrapped_obs, rnn_out[self.obs_dim :]])
