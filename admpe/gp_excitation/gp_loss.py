from typing import Callable

import jax
import jax.numpy as jnp
import gpjax as gpx
import equinox as eqx

import exciting_environments as excenvs
from dmpe.models.model_utils import simulate_ahead, simulate_ahead_with_env

from admpe.gp_excitation.gp_model import GPModel


def compute_GP_posterior_variance(
    gp_model: GPModel,
    observations: jax.Array,
    actions: jax.Array,
    consider_action_distribution: bool,
) -> jax.Array:

    if consider_action_distribution:
        query_points = jnp.concatenate([observations[0:-1, :], actions], axis=-1)
    else:
        query_points = observations
    latent_dist = gp_model.predict_posterior_variance(query_points)
    variance = latent_dist.variance()
    return jnp.sum(variance)


def GP_posterior_uncertainty_loss_function(
    model: eqx.Module,
    init_obs: jax.Array,
    init_state: excenvs.CoreEnvironment.State,
    actions: jax.Array,
    gp_model: GPModel,
    tau: float,
    consider_action_distribution: bool,
    penalty_function: Callable,
):
    if isinstance(model, eqx.Module):
        observations = simulate_ahead(model=model, init_obs=init_obs, actions=actions, tau=tau)
    else:
        observations, _ = simulate_ahead_with_env(env=model, init_obs=init_obs, init_state=init_state, actions=actions)

    gp_posterior_variance = compute_GP_posterior_variance(
        gp_model=gp_model,
        observations=observations,
        actions=actions,
        consider_action_distribution=consider_action_distribution,
    )
    loss = -gp_posterior_variance

    penalty_terms = penalty_function(observations, actions)

    return loss + penalty_terms
