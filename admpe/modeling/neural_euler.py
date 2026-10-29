from typing import Callable

import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx


class MLP(eqx.Module):
    mlp: eqx.nn.MLP
    featurize: Callable

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        width_size: int,
        depth: int,
        featurize: Callable | None,
        *,
        key: jax.random.PRNGKey,
        **kwargs,
    ):
        """Basic state-space MLP. Prediction is based on observation and action.

        Is generally used here to predict the dx/dt in NODE models.

        Args:
            obs_dim (int): Observation dimensionality
            action_dim (int): Action dimensionality
            width_size (int): Layer width (same for all layer) in the MLP
            depth (int): Number of hidden layers in the MLP
            key (jax.random.PRNGKey): Random key for model initialization
        """
        super().__init__(**kwargs)

        self.featurize = (lambda obs: obs) if featurize is None else featurize
        feat_obs_dim = self.featurize(jnp.zeros(obs_dim)).shape[-1]

        self.mlp = eqx.nn.MLP(
            in_size=(feat_obs_dim + action_dim),
            out_size=obs_dim,
            width_size=width_size,
            depth=depth,
            activation=jnn.leaky_relu,
            key=key,
        )

    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        obs_action = jnp.hstack([self.featurize(obs), action])
        return self.mlp(obs_action)


class NeuralEulerODE(eqx.Module):
    func: MLP

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        width_size: int,
        depth: int,
        featurize: Callable | None,
        *,
        key: jax.random.PRNGKey,
        **kwargs,
    ):
        """Neural ODE model with Euler as the ODE solver.

        Very fast to use compared to the models based on more complex ODE solvers.
        However, will generally be less accurate. In essence this model boils down
        to a difference equation model instead of a true NODE.

        Args:
            solver (diffrax.AbstractSolver): The ODE solver to use for the Neural ODE.
            obs_dim (int): Observation dimensionality
            action_dim (int): Action dimensionality
            width_size (int): Layer width (same for all layer) in the MLP
            depth (int): Number of hidden layers in the MLP
            key (jax.random.PRNGKey): Random key for model initialization
        """
        super().__init__(**kwargs)
        self.func = MLP(
            obs_dim,
            action_dim,
            width_size,
            depth,
            featurize=featurize,
            key=key,
        )

    def step(self, obs: jax.Array, action: jax.Array, tau: float) -> jax.Array:
        """Takes a single step in the simulation using the Euler method.

        Args:
            obs (jax.Array): Momentary observation
            action (jax.Array): Action to be applied
            tau (float): Time step size for the simulation

        Returns:
            next_obs (jax.Array): The predicted next observation.
        """
        next_obs = obs + tau * self.func(obs, action)
        return next_obs

    def __call__(self, init_obs: jax.Array, actions: jax.Array, tau: float) -> jax.Array:
        """Simulate the observation trajectory using the Euler ODE solver.

        Args:
            init_obs (jax.Array): Initial observation to start the simulation from
            actions (jax.Array): Actions / inputs to be applied
            tau (float): Time step size for the simulation
        """

        def body_fun(carry, action):
            obs = carry
            obs = self.step(obs, action, tau)
            return obs, obs

        _, observations = jax.lax.scan(body_fun, init_obs, actions)
        observations = jnp.concatenate([init_obs[None, :], observations], axis=0)
        return observations


class NeuralEulerODEPendulum(NeuralEulerODE):
    """Pendulum specific model that deals with the periodic properties of the angle information."""

    def step(self, obs: jax.Array, action: jax.Array, tau: float) -> jax.Array:
        next_obs = super().step(obs, action, tau)
        next_obs = jnp.stack([(((next_obs[..., 0] + 1) % 2) - 1), next_obs[..., 1]], axis=-1)
        return next_obs


class NeuralEulerODECartpole(NeuralEulerODE):
    """Cartpole specific model that deals with the periodic properties of the angle information."""

    def step(self, obs: jax.Array, action: jax.Array, tau: float) -> jax.Array:
        next_obs = super().step(obs, action, tau)
        next_obs = jnp.stack(
            [next_obs[..., 0], next_obs[..., 1], (((next_obs[..., 2] + 1) % 2) - 1), next_obs[..., 3]], axis=-1
        )
        return next_obs
