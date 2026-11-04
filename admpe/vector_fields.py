from typing import Callable
from functools import partial

import dataclasses

import jax
import jax.numpy as jnp
import equinox as eqx

import exciting_environments as excenvs
from dmpe.utils.density_estimation import build_grid

from admpe.utils.helper import get_state_difference_for_env


class VectorField(eqx.Module):
    values: jax.Array
    z: jax.Array
    _points_per_dim: int | None
    _ode: Callable | None
    _discrete_dynamics: Callable | None
    _labels_z: list[str] | None
    _labels_values: list[str] | None

    def visualize(
        self,
        dims: tuple[int, int] = (0, 1),
        aggregate: str = "slice",
        fixed_value: float = 0.0,
        scale: float | None = None,
        color: str = "b",
        alpha: float = 0.5,
    ):
        """Quiver plot of the vector field, projected onto two dims of `z`.

        aggregate="slice": fix all other dims to the grid value nearest
            `fixed_value`, and plot that 2D cross-section (exact, local).
        aggregate="mean": average over all other dims, requires `z` to be
            a regular grid with `points_per_dim` points per dim (global,
            smoothed view of how i,j relate to the field on average).

        Action dims (indices >= values.shape[-1]) always get a 0
        arrow-component, since actions have no eigendynamics.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        z = np.asarray(self.z)
        values = np.asarray(self.values)

        i, j = dims
        n_dims = z.shape[-1]
        n_value_dims = values.shape[-1]

        other_dims = [d for d in range(n_dims) if d not in (i, j)]

        if aggregate == "slice":
            mask = np.ones(z.shape[0], dtype=bool)
            for d in other_dims:
                col = z[:, d]
                nearest = col[np.argmin(np.abs(col - fixed_value))]
                mask &= np.isclose(col, nearest)

            x, y = z[mask, i], z[mask, j]

            def component(d):
                if d < n_value_dims:
                    return values[mask, d]
                return np.zeros(mask.sum())

            u, v = component(i), component(j)
        elif aggregate == "mean":
            if self._points_per_dim is None:
                points_per_dim = round(z.shape[0] ** (1 / n_dims))
                if points_per_dim**n_dims != z.shape[0]:
                    raise ValueError(
                        "Could not infer points_per_dim; z doesn't look like "
                        "a regular grid. Pass points_per_dim explicitly."
                    )
            else:
                points_per_dim = self._points_per_dim

            grid_shape = (points_per_dim,) * n_dims
            z_grid = z.reshape(*grid_shape, n_dims)

            x = z_grid.mean(axis=tuple(other_dims))[..., i]
            y = z_grid.mean(axis=tuple(other_dims))[..., j]

            def component(d):
                if d < n_value_dims:
                    v_grid = values.reshape(*grid_shape, n_value_dims)
                    return v_grid.mean(axis=tuple(other_dims))[..., d]
                return np.zeros_like(x)

            u, v = component(i), component(j)
            x, y, u, v = (arr.ravel() for arr in (x, y, u, v))

        else:
            raise ValueError(f"unknown aggregate mode: {aggregate!r}")

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)

        ax.quiver(x, y, u, v, scale=scale, color=color, alpha=alpha)
        ax.grid(True, alpha=0.5)

        if self._labels_z is None:
            labels = (rf"$z_{{{i}}}$", rf"$z_{{{j}}}$")
        else:
            labels = (self._labels_z[i], self._labels_z[j])

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

        return (fig, ax)


def normalized_ode(
    norm_state_array,
    norm_action,
    ode,
    params,
    physical_normalizations,
    action_normalizations,
):

    state_fields = [f.name for f in dataclasses.fields(physical_normalizations)]
    action_fields = [f.name for f in dataclasses.fields(action_normalizations)]
    state = jnp.stack(
        [getattr(physical_normalizations, f).denormalize(norm_state_array[i]) for i, f, in enumerate(state_fields)],
        axis=-1,
    )
    action = jnp.stack(
        [getattr(action_normalizations, f).denormalize(norm_action[i]) for i, f, in enumerate(action_fields)], axis=-1
    )
    action_fn = lambda t: action

    dxdt = ode(0.0, state, params, action_fn)

    dxdt = jnp.stack(dxdt)
    norm_dxdt = jnp.stack(
        [
            d * 2.0 / (getattr(physical_normalizations, f).max - getattr(physical_normalizations, f).min)
            for f, d in zip(state_fields, dxdt)
        ],
        axis=-1,
    )

    return norm_dxdt


def make_normalized_discrete_dynamics(
    env: excenvs.CoreEnvironment,
    state_difference: Callable | None = None,
):
    if state_difference is None:
        state_difference = get_state_difference_for_env(env)

    deterministic_env = eqx.tree_at(lambda e: e.process_noise_variance, env, 0.0)

    def discrete_dynamics(obs, action):
        state = deterministic_env.generate_state_from_observation(obs)
        next_obs, _ = deterministic_env.step(state, action)
        return state_difference(next_obs, obs)

    return discrete_dynamics


class DynamicsVectorField(VectorField):

    @classmethod
    def from_env(
        cls,
        env: excenvs.CoreEnvironment,
        penalty_function: Callable,
        points_per_dim: int,
        low: float = -1.0,
        high: float = 1.0,
        *,
        use_discrete_dynamics: bool = True,
    ):

        dim = env.obs_dim + env.action_dim

        z = build_grid(dim, low=low, high=high, points_per_dim=points_per_dim)

        if use_discrete_dynamics:
            ode = None
            discrete_dynamics = make_normalized_discrete_dynamics(env)
            evaluate = discrete_dynamics
        else:
            ode = partial(
                normalized_ode,
                ode=env._ode,
                params=env.env_properties.static_params,
                physical_normalizations=env.env_properties.physical_normalizations,
                action_normalizations=env.env_properties.action_normalizations,
            )
            discrete_dynamics = None
            evaluate = ode

        values = eqx.filter_vmap(evaluate, in_axes=(0, 0))(
            z[:, : env.obs_dim],
            z[:, env.obs_dim :],
        )

        d_x = env.obs_dim
        feasible = jax.vmap(lambda q: penalty_function(q[None, :d_x], q[None, d_x:]) == 0)(z)
        # overwrite all infeasible values with zero
        values = jnp.where(feasible[:, None], values, jnp.zeros_like(values))

        physical_labels = list(env.obs_description)
        action_labels = [f.name for f in dataclasses.fields(env.Action)]

        assert ode is not None or discrete_dynamics is not None

        return cls(
            values=values,
            z=z,
            _points_per_dim=points_per_dim,
            _ode=ode,
            _discrete_dynamics=discrete_dynamics,
            _labels_z=physical_labels + action_labels,
            _labels_values=physical_labels,
        )

    @classmethod
    def from_model(cls, model, points_per_dim, low, high, *, use_discrete_dynamics: bool = False):
        raise NotImplementedError()


class CurvatureVectorField(VectorField):

    @classmethod
    def from_env(
        cls,
        env,
        penalty_function,
        points_per_dim,
        low,
        high,
        *,
        use_discrete_dynamics: bool = True,
        smooth_domain=None,
    ):
        d_x = env.obs_dim
        dim = d_x + env.action_dim

        z = build_grid(
            dim,
            low=low,
            high=high,
            points_per_dim=points_per_dim,
        )

        if use_discrete_dynamics:
            ode = None
            discrete_dynamics = make_normalized_discrete_dynamics(env)
            dynamics = discrete_dynamics
        else:
            ode = partial(
                normalized_ode,
                ode=env._ode,
                params=env.env_properties.static_params,
                physical_normalizations=(env.env_properties.physical_normalizations),
                action_normalizations=(env.env_properties.action_normalizations),
            )
            discrete_dynamics = None
            dynamics = ode

        joint_dynamics = lambda q: dynamics(q[:d_x], q[d_x:])
        hessian = jax.hessian(joint_dynamics)

        def score(q):
            H = hessian(q)  # shape: (state_dim, dim, dim)
            return jnp.atleast_1d(jnp.sqrt(jnp.sum(H**2)))

        valid = jax.vmap(lambda q: penalty_function(q[None, :d_x], q[None, d_x:]) == 0)(z)
        if smooth_domain is not None:
            valid &= jax.vmap(smooth_domain)(z)

        safe_z = jnp.where(valid[:, None], z, jnp.zeros_like(z))
        values = eqx.filter_jit(eqx.filter_vmap(score))(safe_z)

        # Construction-time check: do not silently turn undefined
        # curvature at selected points into zero curvature.
        if bool(jnp.any(valid & ~jnp.isfinite(values[:, 0]))):
            raise ValueError(
                "Non-finite Hessian on the selected domain. " "Restrict smooth_domain or regularize the dynamics."
            )

        values = jnp.where(valid[:, None], values, 0.0)

        labels = list(env.obs_description) + [f.name for f in dataclasses.fields(env.Action)]

        return cls(
            values=values,
            z=z,
            _points_per_dim=points_per_dim,
            _ode=ode,
            _discrete_dynamics=discrete_dynamics,
            _labels_z=labels,
            _labels_values=["Hessian norm"],
        )

    @classmethod
    def from_model(cls, model, points_per_dim, low, high, *, use_discrete_dynamics: bool = False):
        raise NotImplementedError()


def get_full_neighbours(radius, dim):
    basis = jnp.eye(dim)
    i, j = jnp.triu_indices(dim, k=1)

    diagonals = jnp.concatenate(
        [
            basis[i] + basis[j],
            basis[i] - basis[j],
        ]
    ) / jnp.sqrt(2.0)

    directions = jnp.concatenate([basis, diagonals])
    offsets = radius * jnp.concatenate([directions, -directions])
    return offsets


class LolaVectorField(VectorField):
    """Measures the nonlinearity of the system dynamics based on the local linear approximation (LOLA) technique
    proposed in:

    ```
    @article{Crombecq2011,
        author = {Crombecq, Karel and Gorissen, Dirk and Deschrijver, Dirk and Dhaene, Tom},
        title = {A Novel Hybrid Sequential Design Strategy for Global Surrogate Modeling of Computer Experiments},
        journal = {SIAM Journal on Scientific Computing},
        volume = {33},
        number = {4},
        pages = {1948-1974},
        year = {2011},
        doi = {10.1137/090761811},
    }
    ```

    NOTE: The constructed field is a scalar field and not a vector field. However, it runs fine within this framework.
    Will likely be changed in the future.
    """

    @staticmethod
    def evaluate_nonlinearity(
        z,
        discrete_dynamics,
        offsets,
        env,
        penalty_function,
        analytical_jac: bool = True,
    ):
        d_x = env.obs_dim

        dynamics = lambda q: discrete_dynamics(q[:d_x], q[d_x:])
        feasible = lambda q: penalty_function(q[None, :d_x], q[None, d_x:]) == 0

        neighbor_points = z[None, :] + offsets

        valid = jax.vmap(feasible)(neighbor_points)
        count = valid.sum()

        # differences are forced to zero if a point is not safe by setting them to the same value as `z`
        safe_points = jnp.where(valid[:, None], neighbor_points, z)
        differences = jax.vmap(dynamics)(safe_points) - dynamics(z)

        if analytical_jac:
            jacobian = jax.jacrev(dynamics)(z)
            slopes = jacobian.T
        else:
            A = jnp.where(valid[:, None], offsets, 0.0)
            B = jnp.where(valid[:, None], differences, 0.0)
            slopes = jnp.linalg.lstsq(A, B, rcond=None)[0]

        lola_residuals = differences - offsets @ slopes
        reduced_residuals = jnp.linalg.norm(lola_residuals, axis=-1)

        # average over the valid results
        score = jnp.where(valid, reduced_residuals, 0.0).sum() / jnp.maximum(count, 1)
        return jnp.atleast_1d(score)

    @classmethod
    def from_env(
        cls,
        env,
        penalty_function,
        points_per_dim,
        low,
        high,
        *,
        use_discrete_dynamics: bool = True,
        analytical_jac: bool = True,
        full_neighbours: bool = False,
        radius: float = 0.05,
    ):

        dim = env.obs_dim + env.action_dim

        z = build_grid(dim, low=low, high=high, points_per_dim=points_per_dim)

        if use_discrete_dynamics:
            ode = None
            discrete_dynamics = make_normalized_discrete_dynamics(env)

            if full_neighbours:
                # computes neighbours on the axes and diagonals
                offsets = get_full_neighbours(radius, dim)
            else:
                # only computes neighbours on the full axes
                offsets = radius * jnp.concatenate([jnp.eye(dim), -jnp.eye(dim)], axis=0)

            # to ensure functionality at the border of the area
            offsets = jnp.concatenate([offsets, 0.5 * offsets], axis=0)

            evaluate = partial(
                cls.evaluate_nonlinearity,
                discrete_dynamics=discrete_dynamics,
                offsets=offsets,
                env=env,
                penalty_function=penalty_function,
                analytical_jac=analytical_jac,
            )
        else:
            raise NotImplementedError()

        values = eqx.filter_vmap(evaluate, in_axes=(0))(z)

        d_x = env.obs_dim
        feasible = jax.vmap(lambda q: penalty_function(q[None, :d_x], q[None, d_x:]) == 0)(z)
        # overwrite all infeasible values with zero
        values = jnp.where(feasible[:, None], values, jnp.zeros_like(values))

        physical_labels = list(env.obs_description)
        action_labels = [f.name for f in dataclasses.fields(env.Action)]

        assert ode is not None or discrete_dynamics is not None

        return cls(
            values=values,
            z=z,
            _points_per_dim=points_per_dim,
            _ode=ode,
            _discrete_dynamics=discrete_dynamics,
            _labels_z=physical_labels + action_labels,
            _labels_values=["LOLA residual"],
        )

    @classmethod
    def from_model(cls, model, points_per_dim, low, high, *, use_discrete_dynamics: bool = False):
        raise NotImplementedError()

    def from_data(cls, data, points_per_dim, low, high, *, use_discrete_dynamics: bool = False):
        raise NotImplementedError()
