import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp


from dmpe.models.model_utils import simulate_ahead_with_env

from admpe.utils.signals import aprbs, multisine
from admpe.data_management import TESTING_TRAJECTORIES_ROOT
from admpe.excitation.excitation_experiment_setup import SUPPORTED_SYSTEMS


def aprbs_actions(key, n_steps, n_actions, **kwargs):
    return aprbs(n_steps=n_steps, batch_size=1, n_actions=n_actions, key=key, **kwargs)[0]


def multisine_actions(key, n_steps, n_actions, **kwargs):
    return multisine(n_steps=n_steps, batch_size=1, n_actions=n_actions, key=key, **kwargs)[0]


SIGNALS = {
    "aprbs": (
        aprbs_actions,
        {"t_min": 10, "t_max": 100, "transition_type": "step", "transition_steps": 1},
    ),
    "multisine": (
        multisine_actions,
        {"n_sines": 8, "fundamental_frequency": 0.005},
    ),
}

SUPPORTED_SIGNALS = tuple(SIGNALS)
SIMPLE_TESTING_SIGNALS_ROOT = TESTING_TRAJECTORIES_ROOT / "simple_signals"
SIMPLE_TESTING_SIGNALS_ROOT.mkdir(parents=True, exist_ok=True)


def get_candidate_batch_function(env, params, penalty_function, data_key):
    signal_function = SIGNALS[params["signal_name"]][0]
    n_steps = params["n_steps"]

    def generate_candidate(candidate_id):
        key = jax.random.fold_in(data_key, candidate_id)

        # generate signal
        signal_key, state_key = jax.random.split(key)
        actions = params["amplitude"] * signal_function(signal_key, n_steps, env.action_dim, **params["signal_params"])
        if actions.shape != (n_steps, env.action_dim):
            raise ValueError("Signal generator returned an unexpected shape.")

        # simulate sequence from random initial state
        initial_state = env.init_state(state_key, deterministic_state=params["deterministic_initial_state"])
        initial_observation = env.generate_observation(initial_state)
        observations, _ = simulate_ahead_with_env(env, initial_observation, initial_state, actions)

        trajectory_penalties = penalty_function(x=observations, u=actions)
        finite = (
            jnp.all(jnp.isfinite(observations))
            & jnp.all(jnp.isfinite(actions))
            & jnp.all(jnp.isfinite(trajectory_penalties))
        )

        valid = finite & (jnp.all(trajectory_penalties <= 0.0))
        return observations, actions, trajectory_penalties, valid

    return jax.jit(jax.vmap(generate_candidate))
