import math

from tqdm import tqdm

import jax.numpy as jnp

from dmpe.models.model_training import ModelTrainer
from admpe.modeling.validation import validate


def fit(
    train_data,
    model_type,
    model,
    model_trainer: ModelTrainer,
    opt_state,
    loader_key,
    params,
    *,
    validation_data=None,
    validation_every=50,
    selection_metric=None,
):

    if model_type.startswith("gp_"):
        raise NotImplementedError()

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
    average_losses = []

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
            average_losses.append(average_loss)

        if validation_data is not None:
            # skip evaluation -> wait for next validation_every
            if iteration % validation_every != 0 and iteration != n_iters:
                continue

            # evaluate model performance
            metrics = {
                name: float(value)
                for name, value in validate(
                    model,
                    validation_data,
                    model_trainer.tau,
                    model_trainer.featurize,
                ).items()
            }
            history.append(
                {
                    "iteration": iteration,
                    "optimizer_steps": iteration * model_trainer.n_train_steps,
                    **metrics,
                }
            )

            # replace best model if adequate
            if selection_metric is not None:
                score = metrics[selection_metric]
                if math.isfinite(score) and score < best_score:
                    best_model, best_score, best_iteration = model, score, iteration

    if selection_metric is None:
        best_model = model
        best_score = None
        best_iteration = None

    return best_model, {
        "selection_metric": selection_metric,
        "best_iteration": best_iteration,
        "best_validation_loss": best_score,
        "history": history,
        "average_losses": average_losses,
    }
