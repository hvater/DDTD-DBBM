import math
from numbers import Integral

import jax
import jax.numpy as jnp

import exciting_environments as excenvs

from admpe.pmsm import NominalSpeedPMSM

SUPPORTED_SIGNALS = ("pi",)

PI_DEFAULTS = dict(
    n_targets=4,
    a=4.0,
    integral_scale=0.3,
    anti_windup_time=1e-3,
    voltage_margin=0.02,
    current_margin=0.01,
    min_target_distance=10.0,  # A
    target_steps_min=50,
    target_steps_max=200,
    max_sampling_attempts=1000,
)


class PIController:
    """dq current PI controller with feedforward and anti-windup.

    References are physical currents in A.
    Returned actions are normalized voltages.
    The integral contribution is passed explicitly, in V.
    """

    def __init__(
        self,
        env,
        *,
        a=4.0,
        integral_scale=0.3,
        anti_windup_time=1e-3,
        voltage_margin=0.02,
    ):
        self.env = env
        self.dt = float(env.tau)
        self.a = a
        self.integral_scale = integral_scale
        self.anti_windup_time = anti_windup_time

        p = env.env_properties.static_params
        self.omega = 2 * jnp.pi * p.p * env.nominal_rpm / 60
        self.voltage_limit = (1 - voltage_margin) * p.u_dc / jnp.sqrt(3.0)

    def reset(self, observation):
        """Return a zero integral contribution with matching shape and dtype."""
        return jnp.zeros_like(observation)

    def currents(self, observation):
        """Convert a normalized current observation to A."""
        n = self.env.env_properties.physical_normalizations
        return jnp.stack(
            (
                n.i_d.denormalize(observation[0]),
                n.i_q.denormalize(observation[1]),
            )
        )

    def normalize_voltage(self, voltage):
        a = self.env.env_properties.action_normalizations
        return jnp.stack(
            (
                a.u_d.normalize(voltage[0]),
                a.u_q.normalize(voltage[1]),
            )
        )

    def motor_terms(self, currents):
        """Return differential inductances and holding voltage."""
        p = self.env.env_properties.static_params
        lut = self.env._interpolate_saturation(currents[0], currents[1])

        # LUT order: L_dd, L_dq, L_qd, L_qq, Psi_d, Psi_q.
        inductances = jnp.stack((lut[0], lut[3]))
        holding_voltage = p.r_s * currents + self.omega * jnp.stack((-lut[5], lut[4]))
        return inductances, holding_voltage

    def __call__(self, state, target_currents, integral):
        """Compute the normalized action and next integral contribution."""
        i = jnp.stack(
            (
                state.physical_state.i_d,
                state.physical_state.i_q,
            )
        )
        error = target_currents - i

        inductances, feedforward = self.motor_terms(i)
        kp = inductances / (1.5 * self.a * self.dt)
        ki = self.integral_scale * kp / (1.5 * self.a**2 * self.dt)

        raw_voltage = feedforward + kp * error + integral

        # Radial projection onto the admissible voltage circle.
        voltage = raw_voltage * jnp.minimum(
            1.0,
            self.voltage_limit / jnp.maximum(jnp.linalg.norm(raw_voltage), 1e-12),
        )

        # Back-calculation anti-windup.
        next_integral = integral + self.dt * (ki * error + (voltage - raw_voltage) / self.anti_windup_time)

        return self.normalize_voltage(voltage), next_integral


def get_candidate_batch_function(
    env: excenvs.CoreEnvironment,
    params,
    penalty_function,
    data_key,
    safe_state_set,
    safely_excitable_set,
):
    if not isinstance(env, NominalSpeedPMSM):
        raise ValueError("The PI signal currently supports only pmsm_nominal.")
    if safe_state_set is None or safely_excitable_set is None:
        raise ValueError("PI generation requires use_safely_excitable_set=True.")
    if float(env.process_noise_variance) != 0.0:
        raise ValueError("Generate PI evaluation trajectories without process noise.")
    if params["amplitude"] != 1.0:
        raise ValueError("Use amplitude=1.0; configure PI targets through signal_overrides.")

    cfg = params["signal_params"]

    for name in (
        "n_targets",
        "target_steps_min",
        "target_steps_max",
        "max_sampling_attempts",
    ):
        value = cfg[name]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")

    for name in ("a", "integral_scale", "anti_windup_time"):
        if not math.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(f"{name} must be finite and positive.")

    for name in ("voltage_margin", "current_margin"):
        if not math.isfinite(cfg[name]) or not 0 <= cfg[name] < 1:
            raise ValueError(f"{name} must be in [0, 1).")

    if not math.isfinite(cfg["min_target_distance"]) or cfg["min_target_distance"] < 0:
        raise ValueError("min_target_distance must be finite and nonnegative.")

    if cfg["target_steps_min"] > cfg["target_steps_max"]:
        raise ValueError("target_steps_min must not exceed target_steps_max.")

    if cfg["n_targets"] * cfg["target_steps_max"] > params["n_steps"]:
        raise ValueError("Increase n_time_steps or reduce n_targets or target_steps_max.")

    if cfg["anti_windup_time"] < env.tau:
        raise ValueError("anti_windup_time must be at least env.tau.")

    n_targets = cfg["n_targets"]
    n_steps = params["n_steps"]

    n = env.env_properties.physical_normalizations
    lower = jnp.stack([axis[0] for axis in safe_state_set.axes])
    upper = jnp.stack([axis[-1] for axis in safe_state_set.axes])

    controller = PIController(
        env,
        a=cfg["a"],
        integral_scale=cfg["integral_scale"],
        anti_windup_time=cfg["anti_windup_time"],
        voltage_margin=cfg["voltage_margin"],
    )

    currents = controller.currents
    normalize_voltage = controller.normalize_voltage
    motor_terms = controller.motor_terms
    voltage_limit = controller.voltage_limit

    def operating_point_ok(obs):
        """Require a safe state with a feasible holding voltage."""
        i = currents(obs)
        _, voltage = motor_terms(i)

        radius_sq = (i[0] / jnp.abs(n.i_d.min)) ** 2 + (i[1] / n.i_q.max) ** 2
        pair = jnp.concatenate((obs, normalize_voltage(voltage)))

        return (
            safe_state_set.contains(obs)
            & safely_excitable_set.contains(pair)
            & jnp.all(jnp.abs(obs) <= 1.0)
            & (radius_sq <= (1 - cfg["current_margin"]) ** 2)
            & (jnp.linalg.norm(voltage) <= voltage_limit)
        )

    def sample_point(key, previous, min_distance):
        def condition(carry):
            _, _, accepted, attempts = carry
            return (~accepted) & (attempts < cfg["max_sampling_attempts"])

        def draw(carry):
            key, _, _, attempts = carry
            key, draw_key = jax.random.split(key)

            obs = jax.random.uniform(
                draw_key,
                previous.shape,
                dtype=previous.dtype,
                minval=lower,
                maxval=upper,
            )
            separated = jnp.linalg.norm(currents(obs) - currents(previous)) >= min_distance
            accepted = operating_point_ok(obs) & separated
            return key, obs, accepted, attempts + 1

        _, obs, accepted, _ = jax.lax.while_loop(
            condition,
            draw,
            (key, previous, jnp.array(False), jnp.int32(0)),
        )

        # Keep failed samples numerically usable, but reject the candidate.
        return jnp.where(accepted, obs, previous), accepted

    def generate_candidate(candidate_id):
        key = jax.random.fold_in(data_key, candidate_id)
        initial_key, state_key, target_key, hold_key = jax.random.split(key, 4)

        fallback_state = env.init_state(
            state_key,
            deterministic_state=True,
        )
        fallback_obs = env.generate_observation(fallback_state)

        if params["deterministic_initial_state"]:
            initial_obs = fallback_obs
            initial_ok = operating_point_ok(initial_obs)
        else:
            initial_obs, initial_ok = sample_point(initial_key, fallback_obs, 0.0)

        initial_state = env.generate_state_from_observation(
            initial_obs,
            key=state_key,
        )

        def draw_target(previous, key):
            target, ok = sample_point(
                key,
                previous,
                cfg["min_target_distance"],
            )
            return target, (target, ok)

        _, (targets, targets_ok) = jax.lax.scan(
            draw_target,
            initial_obs,
            jax.random.split(target_key, n_targets),
        )
        target_currents = jax.vmap(currents)(targets)

        target_steps = jax.random.randint(
            hold_key,
            (n_targets,),
            cfg["target_steps_min"],
            cfg["target_steps_max"] + 1,
        )

        def step(carry, _):
            state, integral, index, elapsed = carry

            # Continue applying the last reference after the schedule ends.
            active_index = jnp.minimum(index, n_targets - 1)
            target = target_currents[active_index]

            action, integral = controller(state, target, integral)
            obs, state = env.step(state, action)

            elapsed = elapsed + 1
            advance = (index < n_targets) & (elapsed >= target_steps[active_index])

            next_carry = (
                state,
                integral,
                index + advance.astype(jnp.int32),
                jnp.where(advance, 0, elapsed),
            )
            return next_carry, (obs, action, active_index)

        final, (observations, actions, target_indices) = jax.lax.scan(
            step,
            (
                initial_state,
                controller.reset(initial_obs),
                jnp.int32(0),
                jnp.int32(0),
            ),
            xs=None,
            length=n_steps,
        )

        observations = jnp.concatenate(
            (initial_obs[None], observations),
            axis=0,
        )
        penalty = jnp.max(penalty_function(x=observations, u=actions))

        finite = (
            jnp.all(jnp.isfinite(observations))
            & jnp.all(jnp.isfinite(actions))
            & jnp.all(jnp.isfinite(targets))
            & jnp.all(jnp.isfinite(final[1]))
            & jnp.isfinite(penalty)
        )
        valid = initial_ok & jnp.all(targets_ok) & finite & (penalty <= 0.0)

        return {
            "observations": observations,
            "actions": actions,
            "max_penalty": penalty,
            "target_observations": targets,
            "target_indices": target_indices,
            "target_steps": target_steps,
        }, valid

    return jax.jit(jax.vmap(generate_candidate))
