"""Construct excitation components."""

from copy import deepcopy

import jax
import jax.numpy as jnp

from dmpe.models.models import NeuralEulerODE, NeuralEulerODECartpole, NeuralEulerODEPendulum
from dmpe.utils.density_estimation import build_grid, DensityEstimate, get_uniform_target_distribution
from dmpe.utils.signals import aprbs
from dmpe.utils.env_utils.cart_pole_utils import setup_env as setup_cart_pole
from dmpe.utils.env_utils.fluid_tank_utils import setup_env as setup_fluid_tank
from dmpe.utils.env_utils.pendulum_utils import setup_env as setup_pendulum

from admpe.excitation.gp_excitation.gp_model import GPModel, update_gp_dataset
from admpe.adaptive_densities import TargetDensity
from admpe.vector_fields import DynamicsVectorField, LolaVectorField
from admpe.excitation.excitation_parameterization import fetch_parameterization

SUPPORTED_SYSTEMS = ("fluid_tank", "pendulum", "cart_pole")
SUPPORTED_OBJECTIVES = (
    "dmpe_uniform",
    "dmpe_static_dynamics",  # mixture of dynamics based and uniform
    "dmpe_static_mixture_dynamics",
    "dmpe_static_LOLA_ls",
    "dmpe_static_LOLA_analytical",
    "gp_grid",
    "gp_trajectory",
)
SUPPORTED_MODELS = ("neural_euler", "perfect")

_SYSTEM_SETUP = {
    "fluid_tank": (setup_fluid_tank, NeuralEulerODE),
    "pendulum": (setup_pendulum, NeuralEulerODEPendulum),
    "cart_pole": (setup_cart_pole, NeuralEulerODECartpole),
}


def validate_experiment(system_name, excitation_objective, model_type, n_time_steps):
    for name, value, choices in (
        ("system_name", system_name, SUPPORTED_SYSTEMS),
        ("excitation_objective", excitation_objective, SUPPORTED_OBJECTIVES),
        ("model_type", model_type, SUPPORTED_MODELS),
    ):
        if value not in choices:
            raise ValueError(f"Unknown {name} {value!r}. Choose from {choices}.")
    if n_time_steps < 2:
        raise ValueError("n_time_steps must be at least 2 (initial observation plus one transition).")


def apply_overrides(defaults, overrides):
    resolved = deepcopy(defaults)

    if overrides is None or resolved is None:
        # nothing to be overwritten
        return resolved

    unknown = overrides.keys() - resolved.keys()
    if unknown:
        raise ValueError(f"Unknown parameters: {sorted(unknown)}")

    resolved.update(deepcopy(overrides))
    return resolved


def construct_target_distribution(
    alg_params,
    env,
    excitation_objective,
    penalty_function,
):
    """Constructs target distribution p^* based of the given parameterization.

    The state-action joint distribution is constructed and then the actions are marginalized if only
    states are to be considered.
    """

    assert alg_params["target_distribution"] is None

    # construct full state-action distribution
    obs_dim = env.obs_dim
    act_dim = env.action_dim

    uniform_distribution = get_uniform_target_distribution(
        dim=obs_dim + act_dim,
        points_per_dim=alg_params["points_per_dim"],
        bandwidth=alg_params["bandwidth"],
        grid_extend=alg_params["grid_extend"],
        consider_action_distribution=True,
        penalty_function=alg_params["penalty_function"],
        act_dim=act_dim,
        obs_dim=obs_dim,
    )

    if excitation_objective == "dmpe_uniform":
        target_distribution = uniform_distribution

    elif excitation_objective == "dmpe_static_dynamics":
        vf = DynamicsVectorField.from_env(
            env,
            penalty_function,
            alg_params["points_per_dim"],
            low=-alg_params["grid_extend"],
            high=alg_params["grid_extend"],
        )
        target = TargetDensity.from_vector_field(
            vf=vf,
            bandwidth=alg_params["bandwidth"],
            penalty_function=penalty_function,
            consider_action_distribution=True,
            obs_dim=env.obs_dim,
            act_dim=env.action_dim,
        )
        target_distribution = target.p[0]

    elif excitation_objective == "dmpe_static_mixture_dynamics":
        vf = DynamicsVectorField.from_env(
            env,
            penalty_function,
            alg_params["points_per_dim"],
            low=-alg_params["grid_extend"],
            high=alg_params["grid_extend"],
        )
        target = TargetDensity.from_vector_field(
            vf=vf,
            bandwidth=alg_params["bandwidth"],
            penalty_function=penalty_function,
            consider_action_distribution=True,
            obs_dim=env.obs_dim,
            act_dim=env.action_dim,
        )
        target_distribution = (target.p[0] + uniform_distribution) / 2
    elif excitation_objective == "dmpe_static_curv":
        vf = None  # implement curvature based vector field
        target = TargetDensity.from_vector_field(
            vf=vf,
            bandwidth=alg_params["bandwidth"],
            penalty_function=penalty_function,
            consider_action_distribution=True,
            obs_dim=env.obs_dim,
            act_dim=env.action_dim,
        )
        raise NotImplementedError()

    elif excitation_objective.startswith("dmpe_static_LOLA_"):
        if excitation_objective == "dmpe_static_LOLA_analytical":
            lola_vf = LolaVectorField.from_env(
                env,
                penalty_function,
                alg_params["points_per_dim"],
                low=-alg_params["grid_extend"],
                high=alg_params["grid_extend"],
                analytical_jac=True,
            )
        elif excitation_objective == "dmpe_static_LOLA_ls":
            lola_vf = LolaVectorField.from_env(
                env,
                penalty_function,
                alg_params["points_per_dim"],
                low=-alg_params["grid_extend"],
                high=alg_params["grid_extend"],
                analytical_jac=False,
            )

        lola_target = TargetDensity.from_vector_field(
            vf=lola_vf,
            bandwidth=0.1,
            penalty_function=penalty_function,
            consider_action_distribution=True,
            obs_dim=env.obs_dim,
            act_dim=env.action_dim,
        )
        # mix lola_target with uniform distribution to ensure general coverage
        target_distribution = (lola_target.p[0] + uniform_distribution) / 2

    if not alg_params["consider_action_distribution"]:

        # marginalize actions for the state-only case
        # first change shape to (points_per_dim, points_per_dim, ..., points_per_dim)
        target_distribution = target_distribution.reshape(
            (alg_params["points_per_dim"],) * (obs_dim + act_dim),
        )
        # then sum over all dimensions that belong to action dimensions
        target_distribution = target_distribution.sum(
            axis=tuple(range(obs_dim, obs_dim + act_dim)),
        ).reshape(-1, 1)

    return target_distribution


def setup_algorithm_parameterization(
    seed,
    system_name,
    excitation_objective,
    model_type,
    env,
    env_params,
    penalty_function,
    n_time_steps,
    featurize,
    model_class,
    *,
    alg_overrides=None,
    model_trainer_overrides=None,
    model_overrides=None,
):

    alg_params, model_trainer_params, model_params = fetch_parameterization(
        system_name, model_type, excitation_objective
    )

    alg_params = apply_overrides(alg_params, alg_overrides)

    # overwrite model_trainer_params to reflect overwrites in alg_params
    model_trainer_params["start_learning"] = alg_params["n_prediction_steps"]
    model_trainer_params["sequence_length"] = alg_params["n_prediction_steps"]

    model_trainer_params = apply_overrides(model_trainer_params, model_trainer_overrides)
    model_params = apply_overrides(model_params, model_overrides)

    params = dict(
        seed=seed,
        n_time_steps=n_time_steps,
        model_class=NeuralEulerODE if model_class is None else model_class,
        env_params=None,
        alg_params=alg_params,
        model_trainer_params=model_trainer_params,
        model_params=model_params,
    )

    key = jax.random.PRNGKey(seed=params["seed"])
    data_key, model_key, loader_key, expl_key, process_noise_key, key = jax.random.split(key, 6)

    data_keys = jax.random.split(data_key, env.action_dim)
    params["model_params"]["key"] = model_key

    # initial guess
    proposed_actions = jnp.hstack(
        [
            aprbs(alg_params["n_prediction_steps"], batch_size=1, t_min=1, t_max=10, key=data_key)[0]
            for data_key in data_keys
        ]
    )

    params["env_params"] = env_params
    alg_params = params["alg_params"]
    alg_params["penalty_function"] = penalty_function

    if model_type == "perfect":
        params["model_params"] = None
        params["model_trainer_params"] = None

    if excitation_objective.startswith("dmpe_"):
        alg_params["target_distribution"] = construct_target_distribution(
            alg_params,
            env,
            excitation_objective,
            penalty_function,
        )

    elif excitation_objective.startswith("gp_"):
        alg_params["loss_type"] = excitation_objective.removeprefix("gp_")
        alg_params["target_support_points"] = (
            build_grid(env.obs_dim + env.action_dim, -1.0, 1.0, alg_params["points_per_dim"])
            if excitation_objective == "gp_grid"
            else None
        )
    return params, proposed_actions, loader_key, expl_key, process_noise_key


def setup_environment(system_name):
    setup_env, model_class = _SYSTEM_SETUP[system_name]
    env, penalty_function, featurize, env_params = setup_env()

    return env, penalty_function, featurize, env_params, model_class


def setup_model(system_name, model_type, model_params):
    if model_type == "perfect":
        return None

    model_class = _SYSTEM_SETUP[system_name][-1]
    return model_class(**model_params)


def setup_estimator(excitation_objective, params):

    dim_obs_space = params["model_params"]["obs_dim"]
    dim_action_space = params["model_params"]["action_dim"]

    if params["alg_params"]["consider_action_distribution"]:
        dim = dim_obs_space + dim_action_space
    else:
        dim = dim_obs_space

    if excitation_objective.startswith("gp_"):
        # reconstruct the GP:
        gp_model = GPModel.initialize_empty(x_dim=dim, y_dim=dim_obs_space)
        n_train = params["n_time_steps"]

        estimator = update_gp_dataset(
            gp_model, k=n_train, observations=jnp.zeros(n_train + 1, dim_obs_space), actions=(n_train, dim_action_space)
        )

    else:
        # reconstruct the density estimate
        n_grid_points = params["alg_params"]["points_per_dim"] ** dim

        estimator = DensityEstimate(
            p=jnp.zeros([n_grid_points, 1]),
            z_g=build_grid(
                dim,
                low=-params["alg_params"]["grid_extend"],
                high=params["alg_params"]["grid_extend"],
                points_per_dim=params["alg_params"]["points_per_dim"],
            ),
            bandwidth=jnp.array([params["alg_params"]["bandwidth"]]),
            n_observations=jnp.array([0]),
        )

    return estimator


def setup_experiment(
    system_name: str,
    excitation_objective: str,
    model_type: str,
    seed: int,
    n_time_steps: int,
    *,
    alg_overrides=None,
    model_trainer_overrides=None,
    model_overrides=None,
):
    """Return environment, resolved parameters, initial actions, and three RNG keys.

    ``neural_euler`` selects the system-specific neural model and learns online;
    ``perfect`` uses the environment itself for predictions. The static target
    uses the true dynamics, even when predictions use a learned model.
    """
    validate_experiment(system_name, excitation_objective, model_type, n_time_steps)

    env, penalty_function, featurize, env_params, model_class = setup_environment(system_name)

    params, proposed_actions, loader_key, expl_key, process_noise_key = setup_algorithm_parameterization(
        seed,
        system_name,
        excitation_objective,
        model_type,
        env,
        env_params,
        penalty_function,
        n_time_steps,
        featurize,
        model_class,
        alg_overrides=alg_overrides,
        model_trainer_overrides=model_trainer_overrides,
        model_overrides=model_overrides,
    )

    return env, params, proposed_actions, loader_key, expl_key, process_noise_key
