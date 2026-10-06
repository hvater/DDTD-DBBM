from typing import Callable

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import gpjax as gpx

import exciting_environments as excenvs

from dmpe.models.model_utils import simulate_ahead, simulate_ahead_with_env
from dmpe.models.model_training import ModelTrainer
from dmpe.excitation.excitation_utils import optimize_actions_multistart

from admpe.excitation.gp_excitation.gp_model import GPModel


def consult_gp_exciter(
    k: int,
    exciter: GPExciter,
    obs: jax.Array,
    state: excenvs.CoreEnvironment.State,
    model: eqx.Module,
    gp_model: GPModel,
    proposed_actions: jax.Array,
    expl_key: jax.random.PRNGKey,
):
    if k > exciter.start_optimizing:
        return exciter.choose_action(
            obs=obs,
            state=state,
            model=model,
            gp_model=gp_model,
            proposed_actions=proposed_actions,
            expl_key=expl_key,
        )
    else:
        next_expl_key, expl_action_key = jax.random.split(expl_key, 2)
        action, next_proposed_actions = exciter.process_propositions(
            obs=obs,
            proposed_actions=proposed_actions,
            expl_action_key=expl_action_key,
        )
        prediction_loss = 0.0
        return action, next_proposed_actions, prediction_loss, next_expl_key


class GPExciter(eqx.Module):
    start_optimizing: int
    loss_function: Callable
    grad_loss_function: Callable
    target_support_points: jax.Array
    excitation_optimizer: optax.GradientTransformation
    tau: float
    n_opt_steps: int
    penalty_function: Callable
    clip_action: bool
    n_starts: int
    consider_action_distribution: bool
    reuse_proposed_actions: bool

    @eqx.filter_jit
    def choose_action(
        self,
        obs: jax.Array,
        state: excenvs.CoreEnvironment.State,
        model: eqx.Module,
        gp_model: GPModel,
        proposed_actions: jax.Array,
        expl_key: jax.random.PRNGKey,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.random.PRNGKey]:
        if self.reuse_proposed_actions:
            n_random_starts = self.n_starts - 1
        else:
            n_random_starts = self.n_starts

        expl_key, new_proposed_actions_key, expl_action_key = jax.random.split(expl_key, 3)

        if n_random_starts > 0:
            random_proposed_actions = jax.random.uniform(
                key=new_proposed_actions_key, shape=(n_random_starts, *proposed_actions.shape), minval=-1, maxval=1
            )

            if self.reuse_proposed_actions:
                all_proposed_actions = jnp.concatenate([proposed_actions[None, :], random_proposed_actions], axis=0)
            else:
                all_proposed_actions = random_proposed_actions
        else:
            all_proposed_actions = proposed_actions[None, :]

        # TODO Is it possible to use `optimize_actions_multistart` for GP and DMPE?
        proposed_actions, loss = optimize_actions_multistart(
            loss_function=self.loss_function,
            grad_loss_function=self.grad_loss_function,
            all_proposed_actions=all_proposed_actions,
            model=model,
            optimizer=self.excitation_optimizer,
            init_obs=obs,
            init_state=state,
            density_estimate=gp_model,
            n_opt_steps=self.n_opt_steps,
            tau=self.tau,
            consider_action_distribution=self.consider_action_distribution,
            target_distribution=None,
            target_support_points=self.target_support_points,
            penalty_function=self.penalty_function,
        )

        action, next_proposed_actions = self.process_propositions(
            obs=obs,
            proposed_actions=proposed_actions,
            expl_action_key=expl_action_key,
        )

        return action, next_proposed_actions, loss, expl_key

    @eqx.filter_jit
    def process_propositions(
        self,
        obs: jax.Array,
        proposed_actions: jax.Array,
        expl_action_key: jax.random.PRNGKey,
    ) -> tuple[jax.Array, jax.Array]:
        action = proposed_actions[0, :]

        if self.clip_action:
            action = jnp.clip(action, -1, 1)

        next_proposed_actions = proposed_actions.at[:-1, :].set(proposed_actions[1:, :])

        new_proposed_action = jax.random.uniform(key=expl_action_key, minval=-1, maxval=1)
        next_proposed_actions = next_proposed_actions.at[-1, :].set(new_proposed_action)
        return action, next_proposed_actions
