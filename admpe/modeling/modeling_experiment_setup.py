import jax
import optax
import equinox as eqx

from dmpe.models.model_training import ModelTrainer

from admpe.excitation.excitation_experiment_setup import SUPPORTED_SYSTEMS, apply_overrides
from admpe.modeling.modeling_parameterization import fetch_parameterization


# TODO: differentiate between different types, i.e., simulation or 1step prediction etc.
SUPPORTED_MODELS = ["neural_euler", "gp_transition", "gp_sequence"]


def setup_parameterization(
    train_data, model_type, featurize, proposed_model_class, *, model_overrides, model_trainer_overrides
):
    model_params, model_trainer_params, model_class = fetch_parameterization(
        train_data["metadata"]["system_name"],
        model_type,
        featurize,
        proposed_model_class,
    )
    model_trainer_params = apply_overrides(model_trainer_params, model_trainer_overrides)
    model_params = apply_overrides(model_params, model_overrides)

    n_iters = model_trainer_params["n_iters"]
    del model_trainer_params["n_iters"]

    # TODO: Is more parameterization required?
    params = dict(
        model_trainer_params=model_trainer_params,
        model_params=model_params,
        model_class=model_class,
        n_iters=n_iters,
    )
    return params


def setup_experiment(model_type, seed, params):

    if model_type == "neural_euler":
        model_key, loader_key = jax.random.split(jax.random.PRNGKey(seed))
        params["model_params"]["key"] = model_key
        model = params["model_class"](**params["model_params"])
        model_trainer = ModelTrainer(**params["model_trainer_params"])
        opt_state = model_trainer.model_optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        return model, model_trainer, opt_state, loader_key, params
    else:
        raise NotImplementedError()
