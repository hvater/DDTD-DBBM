import re
import optax

from dmpe.models.model_utils import MODEL_TAU

from admpe.modeling.gp_modeling import GPSequenceModel, GPTransitionModel

SYSTEM_SPECS = {
    "fluid_tank": (1, 1, 10, ()),
    "pendulum": (2, 1, 20, (0,)),
    "cart_pole": (4, 1, 50, (2,)),
    "pmsm_nominal": (2, 2, 5, ()),
}


def parse_neural_euler_variant(model_type: str) -> tuple[str, int]:
    match = re.fullmatch(r"neural_euler_(tiny|small|medium|large)_(-?[0-9]+)", model_type)
    if match is None:
        raise ValueError("Expected neural_euler_<size>_<horizon>, " "where size is tiny, small, medium, or large.")

    size, horizon_text = match.groups()
    horizon = int(horizon_text)
    if horizon < 1:
        raise ValueError("Training horizon must be at least 1.")

    return size, horizon


def get_base_model_type(model_type: str) -> str:
    if model_type == "neural_euler":
        return model_type

    if model_type.startswith("neural_euler_"):
        parse_neural_euler_variant(model_type)  # validate size and horizon
        return "neural_euler"

    return model_type


def fetch_parameterization(
    system_name,
    model_type,
    featurize,
    proposed_model_class,
):
    obs_dim, action_dim, default_length, angle_indices = SYSTEM_SPECS[system_name]

    if model_type.startswith("neural_euler"):
        if model_type == "neural_euler":
            size_identifier = "large"
            sequence_length = default_length
        else:
            size_identifier, horizon = parse_neural_euler_variant(model_type)
            sequence_length = horizon + 1

        match size_identifier:
            case "tiny":
                width_size = 16
                depth = 1
            case "small":
                width_size = 32
                depth = 1
            case "medium":
                width_size = 64
                depth = 2
            case "large":
                width_size = 128
                depth = 3

        model_params = dict(
            obs_dim=obs_dim,
            action_dim=action_dim,
            width_size=width_size,
            depth=depth,
            featurize=featurize,
            key=None,
        )
        model_trainer_params = dict(
            start_learning=None,
            training_batch_size=128,
            n_train_steps=25,
            sequence_length=sequence_length,
            featurize=featurize,
            model_optimizer=optax.adabelief(learning_rate=1e-4),
            tau=MODEL_TAU,  # NOTE: physically not true, but intended: tau is to be removed from the models
            n_iters=4000,
        )
        model_class = proposed_model_class

    elif model_type.startswith("gru"):
        model_params = dict(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_size=12,
            featurize=featurize,
            key=None,
        )
        model_trainer_params = dict(
            start_learning=None,
            training_batch_size=128,
            n_train_steps=25,
            sequence_length=default_length,
            featurize=featurize,
            model_optimizer=optax.adabelief(learning_rate=1e-4),
            tau=MODEL_TAU,  # NOTE: physically not true, but intended: tau is to be removed from the models
            n_iters=4000,
        )
        model_class = proposed_model_class

    elif model_type == "gp_transition":
        model_params = dict(
            obs_dim=obs_dim,
            action_dim=action_dim,
            featurize=featurize,
            angle_indices=angle_indices,
            lengthscale=0.5,
            variance=1.0,
            obs_stddev=0.0,
            key=None,
        )
        model_trainer_params = dict(
            max_iters=100,
            max_train_points=15000,
        )
        model_class = GPTransitionModel

    elif model_type == "gp_sequence":
        raise NotImplementedError("GPSequenceModel is not implemented.")

    return model_params, model_trainer_params, model_class
