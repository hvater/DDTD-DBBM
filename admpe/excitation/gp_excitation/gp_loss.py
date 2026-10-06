from typing import Callable

import jax
import jax.numpy as jnp
import gpjax as gpx
import equinox as eqx

import exciting_environments as excenvs
from dmpe.models.model_utils import simulate_ahead, simulate_ahead_with_env

from admpe.excitation.gp_excitation.gp_model import GPModel, update_gp_dataset


def compute_GP_posterior_variance(
    gp_model: GPModel,
    observations: jax.Array,
    actions: jax.Array,
    consider_action_distribution: bool,
) -> jax.Array:

    if consider_action_distribution:
        if observations.shape[0] == actions.shape[0] + 1:
            query_points = jnp.concatenate([observations[0:-1, :], actions], axis=-1)
        else:
            query_points = jnp.concatenate([observations, actions], axis=-1)
    else:
        query_points = observations
    summed_variance = gp_model.predict_summed_posterior_variance(query_points)
    return summed_variance


def traj_GP_posterior_uncertainty_loss_function(
    model: eqx.Module,
    init_obs: jax.Array,
    init_state: excenvs.CoreEnvironment.State,
    actions: jax.Array,
    gp_model: GPModel,
    tau: float,
    consider_action_distribution: bool,
    target_distribution: jax.Array,
    target_support_points: jax.Array,
    penalty_function: Callable,
):
    if not isinstance(model, excenvs.CoreEnvironment):
        observations = simulate_ahead(model=model, init_obs=init_obs, actions=actions, tau=tau)
    else:
        observations, _ = simulate_ahead_with_env(env=model, init_obs=init_obs, init_state=init_state, actions=actions)

    gp_posterior_variance = compute_GP_posterior_variance(
        gp_model=gp_model,
        observations=observations,
        actions=actions,
        consider_action_distribution=consider_action_distribution,
    )
    loss = -gp_posterior_variance  # maximize the gp posterior variance at the targeted points

    penalty_terms = penalty_function(observations, actions)

    return loss + penalty_terms


from dmpe.utils.density_estimation import build_grid


def grid_GP_posterior_uncertainty_loss_function(
    model: eqx.Module,
    init_obs: jax.Array,
    init_state: excenvs.CoreEnvironment.State,
    actions: jax.Array,
    gp_model: GPModel,
    tau: float,
    consider_action_distribution: bool,
    target_distribution: jax.Array,
    target_support_points: jax.Array,
    penalty_function: Callable,
):

    if not isinstance(model, excenvs.CoreEnvironment):
        observations = simulate_ahead(model=model, init_obs=init_obs, actions=actions, tau=tau)
    else:
        observations, _ = simulate_ahead_with_env(env=model, init_obs=init_obs, init_state=init_state, actions=actions)

    updated_gp_model = GPModel(
        posteriors=gp_model.posteriors,
        train_x=jnp.concatenate([gp_model.train_x, jnp.concatenate([observations[:-1], actions], axis=-1)], axis=0),
        train_y=jnp.concatenate([gp_model.train_y, observations[1:]], axis=0),
    )

    query_points = target_support_points

    gp_posterior_variance = compute_GP_posterior_variance(
        gp_model=updated_gp_model,
        observations=query_points[:, : gp_model.y_dim],
        actions=query_points[:, gp_model.y_dim :],
        consider_action_distribution=consider_action_distribution,
    )
    loss = gp_posterior_variance  # minimize the gp posterior variance at the query points

    penalty_terms = penalty_function(observations, actions)

    return loss + penalty_terms
