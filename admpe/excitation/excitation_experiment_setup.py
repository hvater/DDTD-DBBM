"""Construct excitation components."""

from copy import deepcopy

import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax

import exciting_environments as excenvs


from dmpe.utils.density_estimation import build_grid, DensityEstimate, get_uniform_target_distribution
from dmpe.utils.signals import aprbs
from dmpe.utils.env_utils.cart_pole_utils import setup_env as setup_cart_pole
from dmpe.utils.env_utils.fluid_tank_utils import setup_env as setup_fluid_tank
from dmpe.utils.env_utils.pendulum_utils import setup_env as setup_pendulum
from admpe.pmsm import setup_env as setup_pmsm

from admpe.modeling.modeling_parameterization import get_base_model_type
from admpe.modeling.neural_euler import NeuralEulerODE, NeuralEulerODECartpole, NeuralEulerODEPendulum
from admpe.modeling.rnn import GRU, GRUPendulum, GRUCartpole
from admpe.modeling.gp_modeling import GPTransitionModel

from admpe.excitation.gp_excitation.gp_model import GPModel, update_gp_dataset
from admpe.adaptive_densities import TargetDensity
from admpe.vector_fields import DynamicsVectorField, CurvatureVectorField, LolaVectorField
from admpe.excitation.excitation_parameterization import fetch_parameterization
from admpe.pmsm import PMSMPlanningModel, FastNominalSpeedPMSM

SUPPORTED_SYSTEMS = ("fluid_tank", "pendulum", "cart_pole", "pmsm_nominal")
SUPPORTED_OBJECTIVES = (
    "dmpe_uniform",
    "dmpe_static_dynamics",  # mixture of dynamics based and uniform
    "dmpe_static_mixture_dynamics",
    "dmpe_static_curv",
    "dmpe_static_mixture_curv",
    "dmpe_static_LOLA_ls",
    "dmpe_static_LOLA_analytical",
    "gp_grid",
    "gp_trajectory",
)
SUPPORTED_MODELS = ("neural_euler", "perfect")

_SYSTEM_SETUP = {
    ("pmsm_nominal", "perfect"): (setup_pmsm, None),
    ("pmsm_nominal", "neural_euler"): (setup_pmsm, NeuralEulerODE),
    ("pmsm_nominal", "gru"): (setup_pmsm, GRU),
    ("pmsm_nominal", "gp_transition"): (setup_pmsm, GPTransitionModel),
    ("fluid_tank", "perfect"): (setup_fluid_tank, None),
    ("fluid_tank", "neural_euler"): (setup_fluid_tank, NeuralEulerODE),
    ("fluid_tank", "gru"): (setup_fluid_tank, GRU),
    ("fluid_tank", "gp_transition"): (setup_fluid_tank, GPTransitionModel),
    ("pendulum", "perfect"): (setup_pendulum, None),
    ("pendulum", "neural_euler"): (setup_pendulum, NeuralEulerODEPendulum),
    ("pendulum", "gru"): (setup_pendulum, GRUPendulum),
    ("pendulum", "gp_transition"): (setup_pendulum, GPTransitionModel),
    ("cart_pole", "perfect"): (setup_cart_pole, None),
    ("cart_pole", "neural_euler"): (setup_cart_pole, NeuralEulerODECartpole),
    ("cart_pole", "gru"): (setup_cart_pole, GRUCartpole),
    ("cart_pole", "gp_transition"): (setup_cart_pole, GPTransitionModel),
}


def _make_pmsm_curvature_dynamics(env, n_substeps=64):
    """Normalized PMSM change over tau using fixed internal substeps."""
    if not isinstance(n_substeps, int) or n_substeps < 1:
        raise ValueError("n_substeps must be a positive integer.")
    norm = env.env_properties.physical_normalizations
    anorm = env.env_properties.action_normalizations
    params = env.env_properties.static_params
    omega = 2 * jnp.pi * params.p * env.nominal_rpm / 60

    def transition(obs, action):
        y0 = (norm.i_d.denormalize(obs[0]), norm.i_q.denormalize(obs[1]), jnp.zeros((), dtype=obs.dtype))
        voltage = jnp.stack((anorm.u_d.denormalize(action[0]), anorm.u_q.denormalize(action[1])))
        term = diffrax.ODETerm(lambda t, y, args: env.nonlinear_ode(t, y, args, lambda _: voltage))
        solver = env._solver
        args = (params, omega)
        dt = jnp.asarray(env.tau / n_substeps, dtype=obs.dtype)
        zero = jnp.zeros((), dtype=obs.dtype)
        solver_state = solver.init(term, zero, dt, y0, args)

        def substep(carry, k):
            y, solver_state = carry
            t = k * dt
            y, _, _, solver_state, _ = solver.step(term, t, t + dt, y, args, solver_state, made_jump=False)
            return (y, solver_state), None

        (y, _), _ = jax.lax.scan(substep, (y0, solver_state), jnp.arange(n_substeps))
        return jnp.stack((norm.i_d.normalize(y[0]), norm.i_q.normalize(y[1]))) - obs

    return transition


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
    system_name: str,
    constrain_to_ses: bool = False,
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

    curvature_dynamics = None
    if system_name == "pmsm_nominal" and excitation_objective in ("dmpe_static_curv", "dmpe_static_mixture_curv"):
        # Record the target integration setting in saved experiment parameters.
        alg_params["target_integration_substeps"] = 64
        curvature_dynamics = _make_pmsm_curvature_dynamics(env, n_substeps=alg_params["target_integration_substeps"])

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
        vf = CurvatureVectorField.from_env(
            env,
            penalty_function,
            points_per_dim=alg_params["points_per_dim"],
            low=-alg_params["grid_extend"],
            high=alg_params["grid_extend"],
            use_discrete_dynamics=True,
            smooth_domain=(lambda q: (q[0] > -1.0)) if isinstance(env, excenvs.FluidTank) else None,
            discrete_dynamics_fn=curvature_dynamics,
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

    elif excitation_objective == "dmpe_static_mixture_curv":
        vf = CurvatureVectorField.from_env(
            env,
            penalty_function,
            points_per_dim=alg_params["points_per_dim"],
            low=-alg_params["grid_extend"],
            high=alg_params["grid_extend"],
            use_discrete_dynamics=True,
            smooth_domain=(lambda q: (q[0] > -1.0)) if isinstance(env, excenvs.FluidTank) else None,
            discrete_dynamics_fn=curvature_dynamics,
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

    if constrain_to_ses:
        from admpe.sets import SafelyExcitableSet

        safely_excitable_set, _ = SafelyExcitableSet.from_system_name(system_name)

        # regenerates grid for the target distribution as it is not returned
        joint_grid = build_grid(
            dim=obs_dim + act_dim,
            low=-alg_params["grid_extend"],
            high=alg_params["grid_extend"],
            points_per_dim=alg_params["points_per_dim"],
        )

        safe_mask = safely_excitable_set.contains(joint_grid)[:, None]
        target_distribution = jnp.where(safe_mask, target_distribution, 0.0)
        mass = target_distribution.sum()
        if not bool(jnp.isfinite(mass) & (mass > 0)):
            raise ValueError("Target has no finite positive mass inside the safely excitable set.")
        target_distribution = target_distribution / mass

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
    constrain_to_ses,
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
            system_name=system_name,
            constrain_to_ses=constrain_to_ses,
        )

    elif excitation_objective.startswith("gp_"):
        alg_params["loss_type"] = excitation_objective.removeprefix("gp_")
        alg_params["target_support_points"] = (
            build_grid(env.obs_dim + env.action_dim, -1.0, 1.0, alg_params["points_per_dim"])
            if excitation_objective == "gp_grid"
            else None
        )
    return params, proposed_actions, loader_key, expl_key, process_noise_key


def setup_environment(system_name: str, model_type: str, noise_variance: float | None = None):
    setup_env, model_class = _SYSTEM_SETUP[system_name, get_base_model_type(model_type)]
    env, penalty_function, featurize, env_params = setup_env()

    # After constructing the environment:
    if noise_variance is not None:
        if noise_variance < 0:
            raise ValueError("noise_variance must be nonnegative.")

        env = eqx.tree_at(
            lambda environment: environment.process_noise_variance,
            env,
            float(noise_variance),
        )

    env_params = env_params.copy()
    env_params["process_noise_variance"] = float(env.process_noise_variance)

    return env, penalty_function, featurize, env_params, model_class


def setup_model(system_name, model_type, model_params):
    if model_type == "perfect":
        return None

    setup_env, model_class = _SYSTEM_SETUP[system_name, get_base_model_type(model_type)]
    params = dict(model_params)

    if issubclass(model_class, (GRU, GPTransitionModel)):
        _, _, featurize, _ = setup_env()
        params["featurize"] = featurize
    elif issubclass(model_class, NeuralEulerODE):
        # if there is no featurize model parameter, the model has been created in the old framework
        # featurize = None reproduces the old behavior
        if params.get("featurize") is None:
            params["featurize"] = None
        else:
            _, _, featurize, _ = setup_env()
            params["featurize"] = featurize

    return model_class(**params)


def setup_estimator(excitation_objective, params, trajectory=None):

    if params["model_params"] is not None:
        dim_obs_space = params["model_params"]["obs_dim"]
        dim_action_space = params["model_params"]["action_dim"]
    else:
        dim_obs_space = trajectory["observations"].shape[-1]
        dim_action_space = trajectory["actions"].shape[-1]

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
    noise_variance: float,
    constrain_to_ses: bool,
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

    env, penalty_function, featurize, env_params, model_class = setup_environment(
        system_name, model_type, noise_variance
    )

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
        constrain_to_ses,
        alg_overrides=alg_overrides,
        model_trainer_overrides=model_trainer_overrides,
        model_overrides=model_overrides,
    )

    planning_model = None
    params["planning_solver"] = type(env._solver).__name__ if model_type == "perfect" else None

    if model_type == "perfect" and system_name == "pmsm_nominal" and excitation_objective.startswith("dmpe_"):
        planning_env = FastNominalSpeedPMSM(
            nominal_rpm=env.nominal_rpm,
            tau=env.tau,
            process_noise_variance=0.0,
            solver=diffrax.Heun(),
        )
        planning_model = PMSMPlanningModel(env=planning_env)
        params["planning_solver"] = type(planning_env._solver).__name__

    return env, params, proposed_actions, loader_key, expl_key, process_noise_key, planning_model
