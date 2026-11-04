"""Adaptation of the PMSM from exciting_environments to fit into the A-DMPE framework."""

from dataclasses import fields
from functools import partial

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from exciting_environments import CoreEnvironment
from exciting_environments.pmsm import PMSM, MotorVariant


class NominalSpeedPMSM(PMSM):
    nominal_rpm: float = eqx.field(static=True)

    def __init__(
        self,
        nominal_rpm=4000.0,
        *,
        tau=1e-4,
        process_noise_variance=0.0,
    ):
        self.nominal_rpm = float(nominal_rpm)
        super().__init__(
            saturated=True,
            motor_variant=MotorVariant.BRUSA,
            static_params=dict(
                p=3,
                r_s=17.932e-3,
                l_d=jnp.nan,
                l_q=jnp.nan,
                psi_p=jnp.nan,
                u_dc=400.0,
                deadtime=0,
            ),
            solver=diffrax.Tsit5(),
            tau=tau,
            process_noise_variance=process_noise_variance,
        )

    def generate_observation(self, state):
        n = self.env_properties.physical_normalizations
        return jnp.stack(
            (
                n.i_d.normalize(state.physical_state.i_d),
                n.i_q.normalize(state.physical_state.i_q),
            ),
            axis=-1,
        )

    def generate_state_from_observation(self, obs, key=None):
        n = self.env_properties.physical_normalizations
        p = self.env_properties.static_params

        i_d = n.i_d.denormalize(obs[0])
        i_q = n.i_q.denormalize(obs[1])
        omega = 2 * jnp.pi * p.p * self.nominal_rpm / 60

        physical_state = self.PhysicalState(
            u_d_buffer=jnp.array(0.0),
            u_q_buffer=jnp.array(0.0),
            epsilon=jnp.array(0.0),
            i_d=i_d,
            i_q=i_q,
            torque=self.currents_to_torque_saturated(i_d, i_q),
            omega_el=omega,
        )

        term = diffrax.ODETerm(
            partial(
                self.nonlinear_ode,
                action=lambda t: jnp.zeros(2),
            )
        )
        solver_state = self._solver.init(
            term,
            0.0,
            self.tau,
            (i_d, i_q, physical_state.epsilon),
            (p, omega),
        )

        return self.State(
            physical_state=physical_state,
            prng_key=jnp.array(jnp.nan) if key is None else key,
            additions=self.Additions(
                solver_state=solver_state,
                active_solver_state=False,
            ),
            reference=self.PhysicalState(**{f.name: jnp.array(jnp.nan) for f in fields(self.PhysicalState)}),
        )

    def init_state(self, rng=None, deterministic_state=False):
        if rng is None or deterministic_state:
            obs, noise_key = jnp.zeros(2), rng
        else:
            sample_key, noise_key = jax.random.split(rng)
            sample = jax.random.ball(sample_key, 2)
            n = self.env_properties.physical_normalizations
            current_limit = jnp.abs(n.i_d.min)

            obs = jnp.stack(
                (
                    n.i_d.normalize(-jnp.abs(sample[0]) * current_limit),
                    n.i_q.normalize(sample[1] * current_limit),
                )
            )

        return self.generate_state_from_observation(obs, noise_key)

    # Use dq voltages directly; constraints belong to the penalty.
    step = CoreEnvironment.step

    def constraint_denormalization(self, action, state):
        return self.denormalize_action(action)

    def noise_state(self, state):
        key, noise_key = jax.random.split(state.prng_key)
        obs = self.generate_observation(state)
        obs += jnp.sqrt(self.process_noise_variance) * jax.random.normal(noise_key, obs.shape)
        return self.generate_state_from_observation(obs, key)

    @property
    def obs_description(self):
        return ["i_d", "i_q"]


def featurize_currents(x):
    return x


def pmsm_penalty(env, x, u):
    n = env.env_properties.physical_normalizations
    i_d = n.i_d.denormalize(x[..., 0])
    i_q = n.i_q.denormalize(x[..., 1])

    penalty = jnp.sum(jax.nn.relu(jnp.abs(x) - 1.0) ** 2)

    current_excess = jax.nn.relu((i_d / jnp.abs(n.i_d.min)) ** 2 + (i_q / n.i_q.max) ** 2 - 1.0)
    penalty += jnp.sum(current_excess**2)

    if u is not None:
        a = env.env_properties.action_normalizations
        u_dc = env.env_properties.static_params.u_dc
        u_d = a.u_d.denormalize(u[..., 0])
        u_q = a.u_q.denormalize(u[..., 1])

        voltage_excess = jax.nn.relu((u_d / u_dc) ** 2 + (u_q / u_dc) ** 2 - 1.0 / 3.0)
        penalty += jnp.sum(voltage_excess**2)

    return penalty


def setup_env():
    env_params = dict(
        nominal_rpm=4000.0,
        tau=1e-4,
        process_noise_variance=0.0,
    )
    env = NominalSpeedPMSM(**env_params)

    env_params.update(
        motor_variant="BRUSA",
        saturated=True,
        deadtime=0,
        voltage_model="ideal_dq",
        solver="Tsit5",
    )

    return (
        env,
        partial(pmsm_penalty, env),
        featurize_currents,
        env_params,
    )
