"""Adaptation of the PMSM from exciting_environments to fit into the A-DMPE framework."""

from dataclasses import fields
from functools import partial

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from exciting_environments import CoreEnvironment
from exciting_environments.pmsm import PMSM, MotorVariant

from dmpe.models.model_utils import simulate_ahead_with_env


class NominalSpeedPMSM(PMSM):
    nominal_rpm: float = eqx.field(static=True)
    _saturation_lut: jax.Array

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
        names = ("L_dd", "L_dq", "L_qd", "L_qq", "Psi_d", "Psi_q")
        self._saturation_lut = jnp.stack(
            [self.LUT_values[name] for name in names],
            axis=-1,
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

    def _interpolate_saturation(self, i_d, i_q):
        """Interpolate all six quantities using shared indices and weights."""
        grid_d, grid_q = self.LUT_grids["L_dd"]

        ix = jnp.clip(
            jnp.searchsorted(grid_d, i_d) - 1,
            0,
            grid_d.shape[0] - 2,
        )
        iy = jnp.clip(
            jnp.searchsorted(grid_q, i_q) - 1,
            0,
            grid_q.shape[0] - 2,
        )

        tx = (i_d - grid_d[ix]) / (grid_d[ix + 1] - grid_d[ix])
        ty = (i_q - grid_q[iy]) / (grid_q[iy + 1] - grid_q[iy])

        values = self._saturation_lut
        return (
            values[ix, iy] * (1 - tx) * (1 - ty)
            + values[ix + 1, iy] * tx * (1 - ty)
            + values[ix, iy + 1] * (1 - tx) * ty
            + values[ix + 1, iy + 1] * tx * ty
        )

    def nonlinear_ode(self, t, y, args, action):
        i_d, i_q, _ = y
        params, omega_el = args
        u_d, u_q = action(t)

        l_dd, l_dq, l_qd, l_qq, psi_d, psi_q = self._interpolate_saturation(i_d, i_q)

        # Combine the voltage, resistance, and rotational terms.
        rhs_d = u_d - params.r_s * i_d + omega_el * psi_q
        rhs_q = u_q - params.r_s * i_q - omega_el * psi_d

        # Explicit 2×2 solve: no general matrix inversion.
        determinant = l_dd * l_qq - l_dq * l_qd
        di_d = (l_qq * rhs_d - l_dq * rhs_q) / determinant
        di_q = (l_dd * rhs_q - l_qd * rhs_d) / determinant

        return di_d, di_q, omega_el

    def currents_to_torque_saturated(self, i_d, i_q):
        psi_d, psi_q = self._interpolate_saturation(i_d, i_q)[4:]
        return 1.5 * self.env_properties.static_params.p * (psi_d * i_q - psi_q * i_d)

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


def setup_env(*, fast: bool = False):
    env_params = dict(
        nominal_rpm=4000.0,
        tau=1e-4,
        process_noise_variance=0.0,
    )
    env_class = FastNominalSpeedPMSM if fast else NominalSpeedPMSM
    env = env_class(**env_params)

    env_params.update(
        motor_variant="BRUSA",
        saturated=True,
        deadtime=0,
        voltage_model="ideal_dq",
        solver=type(env._solver).__name__,
    )

    return (
        env,
        partial(pmsm_penalty, env),
        featurize_currents,
        env_params,
    )


class FastNominalSpeedPMSM(NominalSpeedPMSM):
    def __init__(
        self,
        nominal_rpm=4000.0,
        *,
        tau=1e-4,
        process_noise_variance=0.0,
        solver=None,
    ):
        super().__init__(
            nominal_rpm=nominal_rpm,
            tau=tau,
            process_noise_variance=process_noise_variance,
        )
        self._solver = diffrax.Heun() if solver is None else solver


class PMSMPlanningModel(eqx.Module):
    env: FastNominalSpeedPMSM

    def __call__(self, init_obs, actions, tau=None):
        # The physical sampling time is stored in self.env.tau.
        del tau

        initial_state = self.env.generate_state_from_observation(init_obs)

        observations, _ = simulate_ahead_with_env(
            env=self.env,
            init_obs=init_obs,
            init_state=initial_state,
            actions=actions,
        )
        return observations
