import math

from tqdm import tqdm

import jax
import jax.numpy as jnp

from dmpe.models.model_training import ModelTrainer
from admpe.modeling.validation import validate
from admpe.modeling.gp_modeling import fit_gp_transitions, GPTransitionModel


def fit_gp(
    train_data,
    model_type,
    model: GPTransitionModel,
    model_trainer: ModelTrainer | None,
    opt_state,
    loader_key,
    params,
    *,
    validation_data=None,
    validation_every=50,
    selection_metric=None,
):
    if model_type != "gp_transition":
        raise NotImplementedError(model_type)
    if selection_metric is not None:
        raise ValueError("GP checkpoint selection is not implemented.")

    observations = train_data["observations"]
    actions = train_data["actions"]

    # Select model.N transitions evenly across the complete trajectory.
    # When model.N equals the transition count, every transition is selected.
    indices = jnp.linspace(
        0,
        actions.shape[0] - 1,
        num=model.N,
        dtype=jnp.int32,
    )

    model, histories = fit_gp_transitions(
        model,
        observations,
        actions,
        indices=indices,
        max_iters=params["model_trainer_params"]["max_iters"],
    )

    entry = {
        "iteration": 1,
        "optimizer_steps": sum(len(h) - 1 for h in histories),
        "training_loss": sum(float(h[-1]) for h in histories),
    }

    if validation_data is not None:
        entry.update(
            {
                name: float(value)
                for name, value in validate(
                    model,
                    validation_data,
                    tau=1.0,
                    featurize=model.featurize,
                ).items()
            }
        )

    return model, {
        "selection_metric": None,
        "best_iteration": None,
        "best_validation_loss": None,
        "history": [entry],
        "training_loss_type": "summed_negative_log_marginal_likelihood",
        "gp_optimizer_histories": [h.tolist() for h in histories],
        "training_indices": indices.tolist(),
        "validation_schedule": "final_only",
    }


def fit_nn(
    train_data,
    model_type,
    model,
    model_trainer: ModelTrainer | None,
    opt_state,
    loader_key,
    params,
    *,
    validation_data=None,
    validation_every=50,
    selection_metric=None,
):
    if validation_data is not None:
        if validation_every < 1:
            raise ValueError("validation_every must be positive.")

    if selection_metric is not None and validation_data is None:
        raise ValueError("Model selection requires validation data.")

    observations = train_data["observations"]
    actions = train_data["actions"]

    n_iters = params["n_iters"]

    best_model, best_score, best_iteration = model, math.inf, None
    history = []

    for iteration in tqdm(range(n_iters + 1)):

        # the model is only evaluated in the first iteration
        if iteration > 0:
            model, opt_state, average_loss, loader_key = model_trainer.fit(
                model=model,
                k=jnp.array([observations.shape[0]]),
                observations=observations,
                actions=actions,
                opt_state=opt_state,
                loader_key=loader_key,
            )

        if iteration % validation_every != 0 and iteration != n_iters:
            continue

        # storing basic metrics
        history_entry = {
            "iteration": iteration,
            "optimizer_steps": iteration * model_trainer.n_train_steps,
            "training_loss": float(average_loss) if iteration > 0 else None,
        }
        history.append(history_entry)

        # perform validation if validation data is available
        if validation_data is not None:
            # skip evaluation -> wait for next validation_every

            # evaluate model performance
            metrics = {
                name: float(value)
                for name, value in validate(
                    model,
                    validation_data,
                    model_trainer.tau,
                    model_trainer.featurize,
                    training_horizon=model_trainer.sequence_length - 1,
                ).items()
            }
            history_entry.update(metrics)

            # replace best model if adequate
            if selection_metric is not None:
                score = metrics[selection_metric]
                if math.isfinite(score) and score < best_score:
                    best_model, best_score, best_iteration = model, score, iteration

    # if no selection metric is specified, the last model is returned after training conclusion
    if selection_metric is None:
        best_model = model
        best_score = None
        best_iteration = None

    return best_model, {
        "selection_metric": selection_metric,
        "best_iteration": best_iteration,
        "best_validation_loss": best_score,
        "history": history,
    }


def fit(
    train_data,
    model_type,
    model,
    model_trainer: ModelTrainer | None,
    opt_state,
    loader_key,
    params,
    *,
    validation_data=None,
    validation_every=50,
    selection_metric=None,
):

    if model_type.startswith("gp_"):
        return fit_gp(
            train_data,
            model_type,
            model,
            model_trainer,
            opt_state,
            loader_key,
            params,
            validation_data=validation_data,
            validation_every=validation_every,
            selection_metric=selection_metric,
        )
    else:
        return fit_nn(
            train_data,
            model_type,
            model,
            model_trainer,
            opt_state,
            loader_key,
            params,
            validation_data=validation_data,
            validation_every=validation_every,
            selection_metric=selection_metric,
        )
