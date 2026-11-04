from typing import Callable

import tqdm

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import gpjax as gpx

import exciting_environments as excenvs
from dmpe.models.model_training import ModelTrainer
from dmpe.algorithms.algorithm_utils import interact_and_observe

from admpe.excitation.gp_excitation.gp_exciter import GPExciter, consult_gp_exciter
from admpe.excitation.gp_excitation.gp_model import GPModel, refit_gp_model, update_gp_dataset
from admpe.excitation.gp_excitation.gp_loss import (
    traj_GP_posterior_uncertainty_loss_function,
    grid_GP_posterior_uncertainty_loss_function,
)


def excite_and_fit_with_gp(
    n_time_steps: int,
    env: excenvs.CoreEnvironment,
    model: eqx.Module,
    obs: jax.Array,
    state: excenvs.CoreEnvironment.State,
    proposed_actions: jax.Array,
    exciter: GPExciter,
    model_trainer: ModelTrainer,
    gp_model: GPModel,
    observations: jax.Array,
    actions: jax.Array,
    opt_state_model: optax.OptState,
    loader_key: jax.random.PRNGKey,
    expl_key: jax.random.PRNGKey,
    callback_every: int,
    callback: Callable | None = None,
):
    prediction_losses = []
    mean_posterior_variances = []  # diagnostic analog of data_losses
    callback_out = []

    for k in tqdm.tqdm(range(n_time_steps)):
        action, next_proposed_actions, prediction_loss, next_expl_key = consult_gp_exciter(
            k=k,
            exciter=exciter,
            obs=obs,
            state=state,
            model=model,
            gp_model=gp_model,
            proposed_actions=proposed_actions,
            expl_key=expl_key,
        )
        prediction_losses.append(float(prediction_loss))

        next_obs, next_state, actions, observations = interact_and_observe(
            env=env, k=jnp.array([k]), action=action, state=state, actions=actions, observations=observations
        )

        if model_trainer is not None:
            if k > model_trainer.start_learning:
                model, opt_state_model, average_loss, loader_key = model_trainer.fit(
                    model=model,
                    k=jnp.array([k]),
                    observations=observations,
                    actions=actions,
                    opt_state=opt_state_model,
                    loader_key=loader_key,
                )
        elif hasattr(model, "fit"):
            model = model.fit(model, jnp.array([k]), observations, actions)
        else:
            if k == 0:
                print("Model is used statically and not re-fitted or updated otherwise.")

        # TODO: gp_model update, what is the role of this update? Parameter optimization? Then maybe unneeded?
        # I do not really need the model to predict something?
        ### ...

        # loader_key, gp_fit_key = jax.random.split(loader_key)

        if k % 500 == 0 and k > 0:
            gp_model = refit_gp_model(gp_model, k, observations, actions)
        else:
            gp_model = update_gp_dataset(gp_model, k, observations, actions)

        # TODO add consider action distribution
        # this checks out how good the last obs, action combination actually was
        mean_var = gp_model.predict_summed_posterior_variance(jnp.concatenate([obs[None, :], action[None, :]], axis=-1))
        mean_posterior_variances.append(mean_var)

        # callback
        if k % callback_every == 0 and k > 0:
            if callback is not None:
                callback_out.append(
                    callback(
                        k=jnp.array([k]),
                        env=env,
                        obs=obs,
                        state=state,
                        action=action,
                        next_obs=next_obs,
                        next_state=next_state,
                        observations=observations,
                        actions=actions,
                        model=model,
                        proposed_actions=proposed_actions,
                        next_proposed_actions=next_proposed_actions,
                        mean_posterior_variances=mean_posterior_variances,
                        prediction_losses=prediction_losses,
                    )
                )
        ## End Monitoring

        # k <- k + 1
        obs = next_obs
        state = next_state
        proposed_actions = next_proposed_actions
        expl_key = next_expl_key

    return observations, actions, model, gp_model, prediction_losses, proposed_actions, callback_out


def excite_with_gp_uncertainty(
    env: excenvs.CoreEnvironment,
    exp_params: dict,
    proposed_actions: jax.Array,
    loader_key: jax.random.PRNGKey,
    expl_key: jax.random.PRNGKey,
    process_noise_key: jax.random.PRNGKey | None = None,
    callback_every: int | None = None,
    callback: Callable | None = None,
    *,
    planning_model=None,
):
    if planning_model is not None:
        raise NotImplementedError()

    obs, state = env.reset(rng=process_noise_key, deterministic_state=True)

    dim_obs_space = obs.shape[0]
    dim_action_space = env.action_dim

    if exp_params["alg_params"]["consider_action_distribution"]:
        dim = dim_obs_space + dim_action_space
    else:
        dim = dim_obs_space

    observations = jnp.zeros((exp_params["n_time_steps"], dim_obs_space))
    observations = observations.at[0].set(obs)
    actions = jnp.zeros((exp_params["n_time_steps"] - 1, dim_action_space))

    loss_type = exp_params["alg_params"]["loss_type"]
    if loss_type == "trajectory":
        loss_function = traj_GP_posterior_uncertainty_loss_function
    elif loss_type == "grid":
        loss_function = grid_GP_posterior_uncertainty_loss_function
    else:
        raise ValueError(f"Specified 'loss_type' {loss_type} is unknown.")
    grad_loss_function = jax.value_and_grad(loss_function, argnums=(3))

    exciter = GPExciter(
        start_optimizing=exp_params["alg_params"]["start_optimizing"],
        loss_function=loss_function,
        grad_loss_function=grad_loss_function,
        target_support_points=exp_params["alg_params"]["target_support_points"],
        excitation_optimizer=exp_params["alg_params"]["excitation_optimizer"],
        tau=env.tau,
        n_opt_steps=exp_params["alg_params"]["n_opt_steps"],
        penalty_function=exp_params["alg_params"]["penalty_function"],
        clip_action=exp_params["alg_params"]["clip_action"],
        n_starts=exp_params["alg_params"]["n_starts"],
        consider_action_distribution=exp_params["alg_params"]["consider_action_distribution"],
        reuse_proposed_actions=exp_params["alg_params"]["reuse_proposed_actions"],
    )

    if exp_params["model_trainer_params"] is None and exp_params["model_params"] is None:
        model_trainer = None
        model = env
        opt_state_model = None
    elif exp_params["model_trainer_params"] is None and exp_params["model_params"] is not None:
        model_trainer = None
        model = exp_params["model_class"](**exp_params["model_params"])
        opt_state_model = None
    else:
        model_trainer = ModelTrainer(
            start_learning=exp_params["model_trainer_params"]["start_learning"],
            training_batch_size=exp_params["model_trainer_params"]["training_batch_size"],
            n_train_steps=exp_params["model_trainer_params"]["n_train_steps"],
            sequence_length=exp_params["model_trainer_params"]["sequence_length"],
            featurize=exp_params["model_trainer_params"]["featurize"],
            model_optimizer=optax.adabelief(exp_params["model_trainer_params"]["model_lr"]),
            tau=env.tau,
        )
        model = exp_params["model_class"](**exp_params["model_params"])
        opt_state_model = model_trainer.model_optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # Establish GP model # TODO, what is x0, and delta_0?
    gp_model = GPModel.initialize_empty(x_dim=dim, y_dim=dim_obs_space)

    # TODO excite_and_fit needs to be rebuild for this purpose
    observations, actions, model, gp_model, losses, proposed_actions, callback_out = excite_and_fit_with_gp(
        n_time_steps=exp_params["n_time_steps"],
        env=env,
        model=model,
        obs=obs,
        state=state,
        proposed_actions=proposed_actions,
        exciter=exciter,
        model_trainer=model_trainer,
        gp_model=gp_model,
        observations=observations,
        actions=actions,
        opt_state_model=opt_state_model,
        loader_key=loader_key,
        expl_key=expl_key,
        callback_every=callback_every if callback_every is not None else exp_params["n_time_steps"] + 1,
        callback=callback,
    )

    return observations, actions, model, gp_model, losses, proposed_actions, callback_out
