from admpe.excitation_experiment_setup import SUPPORTED_SYSTEMS, apply_overrides
from admpe.modeling_parameterization import fetch_parameterization


# TODO: differentiate between different types, i.e., simulation or 1step prediction etc.
SUPPORTED_MODELS = ["neural_euler"]


def setup_parameterization(
    train_data, model_type, featurize, proposed_model_class, *, model_overrides, model_trainer_overrides
):
    model_params, model_trainer_params, model_class = fetch_parameterization(
        train_data.metadata["system_name"],
        model_type,
        featurize,
        proposed_model_class,
    )
    model_trainer_params = apply_overrides(model_trainer_params, model_trainer_overrides)
    model_params = apply_overrides(model_params, model_overrides)

    # TODO: Is more parameterization required?
    params = dict(
        model_trainer_params=model_trainer_params,
        model_params=model_params,
        model_class=model_class,
    )
    return params


def setup_experiment(train_data, seed, params):
    raise NotImplementedError()
    model, optimizer, opt_state, key, featurize, params = None
    return model, optimizer, opt_state, key, featurize, params


def fit():
    raise NotImplementedError()
