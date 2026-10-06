def fetch_parameterization(
    system_name,
    model_type,
    featurize,
    proposed_model_class,
):
    if system_name == "fluid_tank":
        if model_type == "neural_euler":
            model_params = dict(
                obs_dim=1,
                action_dim=1,
                width_size=128,
                depth=3,
                key=None,
            )
            model_trainer_params = dict(
                start_learning=None,
                training_batch_size=128,
                n_train_steps=int(1e5),
                sequence_length=10,
                featurize=featurize,
                model_lr=1e-4,
            )
            model_class = proposed_model_class
    elif system_name == "pendulum":
        if model_type == "neural_euler":
            model_trainer_params = dict(
                start_learning=None,
                training_batch_size=128,
                n_train_steps=int(1e5),
                sequence_length=20,
                featurize=featurize,
                model_lr=1e-4,
            )
            model_params = dict(
                obs_dim=2,
                action_dim=1,
                width_size=128,
                depth=3,
                key=None,
            )
            model_class = proposed_model_class
    elif system_name == "cart_pole":
        if model_type == "neural_euler":
            model_trainer_params = dict(
                start_learning=None,
                training_batch_size=128,
                n_train_steps=int(1e5),
                sequence_length=50,
                featurize=featurize,
                model_lr=1e-4,
            )
            model_params = dict(
                obs_dim=4,
                action_dim=1,
                width_size=128,
                depth=3,
                key=None,
            )
            model_class = proposed_model_class

    return model_params, model_trainer_params, model_class
