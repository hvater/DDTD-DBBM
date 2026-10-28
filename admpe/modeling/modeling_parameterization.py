import re
import optax

from admpe.modeling.gp_modeling import GPSequenceModel, GPTransitionModel


def parse_neural_euler_variant(model_type: str) -> tuple[str, int]:
    match = re.fullmatch(r"neural_euler_(tiny|small|medium|large)_(-?[0-9]+)", model_type)
    if match is None:
        raise ValueError("Expected neural_euler_<size>_<horizon>, " "where size is tiny, small, medium, or large.")

    size, horizon_text = match.groups()
    horizon = int(horizon_text)
    if horizon < 1:
        raise ValueError("Training horizon must be at least 1.")

    return size, horizon


def fetch_parameterization(
    system_name,
    model_type,
    featurize,
    proposed_model_class,
):

    if model_type.startswith("neural_euler"):
        if model_type == "neural_euler":
            size_identifier = "large"
            sequence_length = {"fluid_tank": 10, "pendulum": 20, "cart_pole": 50}[system_name]
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
            obs_dim={"fluid_tank": 1, "pendulum": 2, "cart_pole": 4}[system_name],
            action_dim=1,
            width_size=width_size,
            depth=depth,
            key=None,
        )
        model_trainer_params = dict(
            start_learning=None,
            training_batch_size=128,
            n_train_steps=25,
            sequence_length=sequence_length,
            featurize=featurize,
            model_optimizer=optax.adabelief(learning_rate=1e-4),
            tau=1,  # NOTE: physically not true, but likely not a problem
            n_iters=4000,
        )
        model_class = proposed_model_class

    elif model_type.startswith("gru"):
        model_params = dict(
            obs_dim={"fluid_tank": 1, "pendulum": 2, "cart_pole": 4}[system_name],
            action_dim=1,
            hidden_size=12,
            featurize=featurize,
            key=None,
        )
        model_trainer_params = dict(
            start_learning=None,
            training_batch_size=128,
            n_train_steps=25,
            sequence_length={"fluid_tank": 10, "pendulum": 20, "cart_pole": 50}[system_name],
            featurize=featurize,
            model_optimizer=optax.adabelief(learning_rate=1e-4),
            tau=1,  # NOTE: physically not true, but likely not a problem
            n_iters=4000,
        )
        model_class = proposed_model_class

    elif model_type.startswith("gp_"):
        model_params = dict(
            lengthscale=0.5,
            variance=1.0,
            obs_stddev=0.1,
        )
        model_trainer_params = dict(
            max_iters=100,
            max_train_points=500,
        )
        if model_type == "gp_transition":
            model_class = GPTransitionModel
        elif model_type == "gp_sequence":
            model_class = GPSequenceModel

    return model_params, model_trainer_params, model_class
