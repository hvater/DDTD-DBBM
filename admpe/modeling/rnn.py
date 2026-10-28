from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx


class GRU(eqx.Module):
    """GRU based state-space model."""

    hidden_size: int = eqx.field(static=True)
    cell: eqx.Module
    linear: eqx.Module
    output_dim: int
    input_dim: int
    featurize: Callable

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int, featurize: Callable, *, key):

        self.hidden_size = hidden_size
        self.output_dim = obs_dim

        self.featurize = featurize
        feat_obs_dim = self.featurize(jnp.zeros(obs_dim)).shape[-1]
        self.input_dim = feat_obs_dim + action_dim

        cell_key, linear_key = jax.random.split(key)
        self.cell = eqx.nn.GRUCell(self.input_dim, self.hidden_size, key=cell_key)
        self.linear = eqx.nn.Linear(self.hidden_size, self.output_dim, use_bias=True, key=linear_key)

    def step(self, obs, action, hidden):
        next_hidden = self.cell(jnp.hstack([self.featurize(obs), action]), hidden)
        next_obs = obs + self.linear(next_hidden)
        return next_obs, next_hidden

    def __call__(self, init_obs: jax.Array, actions: jax.Array, tau: float) -> jax.Array:

        hidden = jnp.zeros(self.hidden_size)
        carry = (init_obs, hidden)

        def f(carry, action):
            obs, hidden = carry
            next_obs, next_hidden = self.step(obs, action, hidden)
            return (next_obs, next_hidden), next_obs

        _, pred_observations = jax.lax.scan(f, carry, actions)
        return jnp.concatenate([init_obs[None], pred_observations], axis=0)


def wrap_angle(obs, indices):
    indices = jnp.asarray(indices, dtype=jnp.int32)
    angles = obs[..., indices]
    wrapped = (angles + 1) % 2 - 1
    return obs.at[..., indices].set(wrapped)


class GRUPendulum(GRU):

    def step(self, obs, action, hidden):
        next_obs, next_hidden = super().step(obs, action, hidden)
        wrapped_obs = wrap_angle(next_obs, 0)
        return wrapped_obs, next_hidden


class GRUCartpole(GRU):

    def step(self, obs, action, hidden):
        next_obs, next_hidden = super().step(obs, action, hidden)
        wrapped_obs = wrap_angle(next_obs, 2)
        return wrapped_obs, next_hidden
