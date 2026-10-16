from typing import Callable

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import jax_tqdm

import exciting_environments as excenvs
from dmpe.models.model_utils import simulate_ahead_with_env


def generate_initial_actions(action_key, n_starts, n_steps, action_dim):
    return jax.random.uniform(
        key=action_key,
        shape=(n_starts, n_steps, action_dim),
        minval=-1,
        maxval=1,
    )


def generate_target_observation(target_key, obs_dim):
    return jax.random.uniform(
        key=target_key,
        shape=(obs_dim,),
        minval=-1,
        maxval=1,
    )


def loss_function(
    actions: jax.Array,
    target: jax.Array,
    init_obs: jax.Array,
    penalty_function: Callable,
    featurize: Callable,
    env: excenvs.CoreEnvironment,
):
    init_state = env.generate_state_from_observation(init_obs)
    observations, _ = simulate_ahead_with_env(env, init_obs, init_state, actions)
    loss = jnp.mean((featurize(observations) - featurize(target[None])) ** 2)
    penalties = penalty_function(observations, actions) * 1e3
    return loss + penalties


@eqx.filter_jit
def optimize_actions(
    proposed_actions: jax.Array,
    target: jax.Array,
    init_obs: jax.Array,
    gradient_function: Callable,
    penalty_function: Callable,
    featurize: Callable,
    env: excenvs.CoreEnvironment,
    optimizer: optax.GradientTransformation,
    n_opt_steps: int,
):
    opt_state = optimizer.init(proposed_actions)

    @jax_tqdm.loop_tqdm(n_opt_steps, print_rate=min(100, proposed_actions.shape[0]))
    def body_fun(i, carry):
        proposed_actions, opt_state = carry
        grad = gradient_function(
            proposed_actions,
            target,
            init_obs,
            penalty_function,
            featurize,
            env,
        )
        updates, opt_state = optimizer.update(grad, opt_state, proposed_actions)
        proposed_actions = optax.apply_updates(proposed_actions, updates)

        return (proposed_actions, opt_state)

    proposed_actions, _ = jax.lax.fori_loop(0, n_opt_steps, body_fun, (proposed_actions, opt_state))
    return proposed_actions


@eqx.filter_jit
def optimize_actions_multistart(
    proposed_actions: jax.Array,
    target: jax.Array,
    init_obs: jax.Array,
    loss_function: Callable,
    penalty_function: Callable,
    featurize: Callable,
    env: excenvs.CoreEnvironment,
    optimizer: optax.GradientTransformation,
    n_opt_steps: int,
) -> tuple[jax.Array, jax.Array]:

    gradient_function = eqx.filter_grad(loss_function)
    actions = eqx.filter_vmap(optimize_actions, in_axes=(0, None, None, None, None, None, None, None, None))(
        proposed_actions, target, init_obs, gradient_function, penalty_function, featurize, env, optimizer, n_opt_steps
    )

    losses = eqx.filter_vmap(loss_function, in_axes=(0, None, None, None, None, None))(
        actions, target, init_obs, penalty_function, featurize, env
    )
    best_idx = jnp.argmin(losses)
    return actions[best_idx], losses[best_idx]


def get_candidate_batch_function(env, params, optimizer, featurize, loss_function, penalty_function, data_key):

    n_steps = params["n_steps"]
    n_starts = params["n_starts"]
    n_opt_steps = params["n_opt_steps"]

    def generate_candidate(candidate_id):
        key = jax.random.fold_in(data_key, candidate_id)

        action_key, state_key, target_key = jax.random.split(key, 3)

        proposed_actions = generate_initial_actions(action_key, n_starts, n_steps, env.action_dim)
        if proposed_actions.shape != (n_starts, n_steps, env.action_dim):
            raise ValueError("Initial_actions have an unexpected shape.")

        initial_state = env.init_state(state_key, deterministic_state=params["deterministic_initial_state"])
        initial_observation = env.generate_observation(initial_state)
        target_observation = generate_target_observation(target_key, env.obs_dim)

        chosen_actions, losses = optimize_actions_multistart(
            proposed_actions,
            target_observation,
            initial_observation,
            loss_function,
            penalty_function,
            featurize,
            env,
            optimizer,
            n_opt_steps,
        )

        observations, _ = simulate_ahead_with_env(env, initial_observation, initial_state, chosen_actions)

        trajectory_penalties = penalty_function(x=observations[:-1], u=chosen_actions)
        finite = (
            jnp.all(jnp.isfinite(observations))
            & jnp.all(jnp.isfinite(chosen_actions))
            & jnp.all(jnp.isfinite(trajectory_penalties))
        )

        valid = finite & (jnp.all(trajectory_penalties <= 0.0))
        return observations, chosen_actions, trajectory_penalties, valid, proposed_actions, target_observation

    return jax.jit(jax.vmap(generate_candidate))
