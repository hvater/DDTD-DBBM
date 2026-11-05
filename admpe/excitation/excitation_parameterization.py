import optax


from dmpe.utils.env_utils.fluid_tank_utils import setup_env as setup_fluid_tank_env
from dmpe.utils.env_utils.pendulum_utils import setup_env as setup_pendulum_env
from dmpe.utils.env_utils.cart_pole_utils import setup_env as setup_cart_pole_env
from admpe.pmsm import setup_env as setup_pmsm


def fetch_parameterization(system_name, model_type, excitation_objective):

    # TODO: differentiate between the algorithms? What needs to be adapted for the other versions?

    env, penalty_function, featurize, env_params = setup_fluid_tank_env()
    if system_name == "fluid_tank":
        alg_params = dict(
            bandwidth=0.1,  # TODO: Choose according to the new formula?
            n_prediction_steps=10,
            points_per_dim=50,
            grid_extend=1.05,
            excitation_optimizer=optax.adabelief(1e-1),
            n_opt_steps=10,
            start_optimizing=5,
            consider_action_distribution=True,
            penalty_function=penalty_function,
            target_distribution=None,
            clip_action=True,
            n_starts=5,
            reuse_proposed_actions=True,
        )

        model_trainer_params = dict(
            start_learning=None,
            training_batch_size=128,
            n_train_steps=1,
            sequence_length=None,
            featurize=featurize,
            model_lr=1e-4,
        )
        model_params = dict(
            obs_dim=env.physical_state_dim,
            action_dim=env.action_dim,
            width_size=128,
            featurize=featurize,
            depth=3,
            key=None,
        )

    elif system_name == "pendulum":

        env, penalty_function, featurize, env_params = setup_pendulum_env()
        alg_params = dict(
            bandwidth=0.08,
            n_prediction_steps=20,
            points_per_dim=21,
            grid_extend=1.05,
            excitation_optimizer=optax.adabelief(1e-1),
            n_opt_steps=10,
            start_optimizing=5,
            consider_action_distribution=True,
            penalty_function=penalty_function,
            target_distribution=None,
            clip_action=True,
            n_starts=5,
            reuse_proposed_actions=True,
        )

        model_trainer_params = dict(
            start_learning=None,
            training_batch_size=128,
            n_train_steps=1,
            sequence_length=None,
            featurize=featurize,
            model_lr=1e-4,
        )
        model_params = dict(
            obs_dim=env.physical_state_dim,
            action_dim=env.action_dim,
            width_size=128,
            featurize=featurize,
            depth=3,
            key=None,
        )

    elif system_name == "cart_pole":

        env, penalty_function, featurize, env_params = setup_cart_pole_env()
        alg_params = dict(
            bandwidth=0.12,
            n_prediction_steps=50,
            points_per_dim=10,
            grid_extend=1.05,
            excitation_optimizer=optax.adabelief(1e-1),
            n_opt_steps=5,
            start_optimizing=5,
            consider_action_distribution=True,
            penalty_function=penalty_function,
            target_distribution=None,
            clip_action=True,
            n_starts=5,
            reuse_proposed_actions=True,
        )

        model_trainer_params = dict(
            start_learning=None,
            training_batch_size=128,
            n_train_steps=10,
            sequence_length=None,
            featurize=featurize,
            model_lr=1e-4,
        )
        model_params = dict(
            obs_dim=env.physical_state_dim,
            action_dim=env.action_dim,
            width_size=128,
            featurize=featurize,
            depth=3,
            key=None,
        )

    elif system_name == "pmsm_nominal":
        env, penalty_function, featurize, _ = setup_pmsm()

        alg_params = dict(
            bandwidth=0.08,
            n_prediction_steps=15,
            points_per_dim=21,
            grid_extend=1.05,
            excitation_optimizer=optax.adabelief(1e-2),
            n_opt_steps=100 if model_type == "perfect" else 200,
            start_optimizing=5,
            consider_action_distribution=True,
            penalty_function=penalty_function,
            target_distribution=None,
            clip_action=False,
            n_starts=10,
            reuse_proposed_actions=True,
        )
        model_trainer_params = dict(
            start_learning=None,
            training_batch_size=64,
            n_train_steps=5,
            sequence_length=None,
            featurize=featurize,
            model_lr=1e-4,
        )
        model_params = dict(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            width_size=64,
            depth=3,
            featurize=featurize,
            key=None,
        )

    else:
        raise ValueError(f"No parameterization could be found for system_name: '{system_name}'")

    return alg_params, model_trainer_params, model_params
