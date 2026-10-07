import abc

import jax
import equinox as eqx

from dmpe.models.models import NeuralEulerODE
from dmpe.models.model_utils import simulate_ahead_with_env
import exciting_environments as excenvs

from admpe.excitation.excitation_experiment_setup import setup_environment


class ModelWrapper(abc.ABC):
    """
    A base class for wrapping models, providing an interface for step-wise predictions,
    derivative computations, and simulation rollouts.
    """

    model: eqx.Module
    featurize: callable

    def __init__(self, model, featurize):
        self.model = model
        self.featurize = featurize

    @abc.abstractmethod
    def step(self, obs: jax.Array, action: jax.Array, tau: float = 1.0):
        """
        Perform a one-step prediction given the current observation, action, and time step.

        Args:
            obs: The current state or observation.
            action: The action taken at the current state.
            tau: The time step for the prediction.

        Returns:
            The predicted next observation after applying the action.
        """
        return

    @abc.abstractmethod
    def rollout(self, init_obs: jax.Array, actions: jax.Array, tau: float = 1.0):
        """
        Simulate a trajectory starting from an initial observation and a sequence of actions.

        Args:
            init_obs: The initial state or observation.
            actions: A sequence of actions to apply over time.
            tau: The time step for each action in the trajectory.

        Returns:
            A sequence of observations representing the simulation rollout.
        """
        return


class NodeModelWrapper(ModelWrapper):
    """Wraps a DMPE NODE model for comparison."""

    model: NeuralEulerODE
    featurize: callable

    def step(self, obs, action, tau):
        return self.featurize(self.model.step(obs, action, tau))

    def derivative(self, obs, action):
        return self.model.func(obs, action)

    def rollout(self, init_obs, actions, tau):
        pred = self.model(init_obs, actions, tau)
        return eqx.filter_vmap(self.featurize)(pred)


class EnvWrapper(ModelWrapper):
    """Wraps an exciting_environments env for comparison.

    can you do this by differentiating the step function by time?
    """

    model: excenvs.CoreEnvironment
    featurize: callable

    @eqx.filter_jit
    def step(self, obs, action, tau):
        assert tau == self.model.tau
        state = self.model.generate_state_from_observation(obs)
        next_obs, _ = self.model.step(state, action)
        return self.featurize(next_obs)

    @eqx.filter_jit
    def rollout(self, init_obs, actions, tau):
        init_state = self.model.generate_state_from_observation(init_obs)
        observations, _ = simulate_ahead_with_env(self.model, init_obs, init_state, actions)
        return eqx.filter_vmap(self.featurize)(observations)


def get_wrapped_model_and_env(modeling_result):
    env, _, featurize, _, _ = setup_environment(modeling_result["metadata"]["system_name"])
    env = eqx.tree_at(lambda s: s.process_noise_variance, env, 0.0)  # force env to be deterministic

    # wrap model and env:
    wrapped_model = NodeModelWrapper(
        model=modeling_result["model"],
        featurize=featurize,
    )

    wrapped_env = EnvWrapper(
        model=env,
        featurize=featurize,
    )
    return wrapped_model, wrapped_env
