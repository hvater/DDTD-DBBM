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
    template_state = deterministic_env.init_state()
    normalizations = deterministic_env.env_properties.physical_normalizations
    names = [f.name for f in dataclasses.fields(deterministic_env.PhysicalState)]

    def discrete_dynamics(norm_state_array, norm_action):
        physical_state = deterministic_env.PhysicalState(
            **{name: getattr(normalizations, name).denormalize(norm_state_array[i]) for i, name in enumerate(names)}
        )
        state = eqx.tree_at(
            lambda s: s.physical_state,
            template_state,
            physical_state,
        )
        _, next_state = deterministic_env.step(state, norm_action)

        next_norm_state_array = jnp.stack(
            [getattr(normalizations, name).normalize(getattr(next_state.physical_state, name)) for name in names]
        )

        return state_difference(next_norm_state_array, norm_state_array)

    return discrete_dynamics


class DynamicsVectorField(VectorField):

    @classmethod
    def from_env(
        cls,
        env: excenvs.CoreEnvironment,
        points_per_dim: int,
        low: float = -1.0,
        high: float = 1.0,
        *,
        use_discrete_dynamics: bool = False,
    ):

        dim = env.physical_state_dim + env.action_dim

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
            z[:, : env.physical_state_dim],
            z[:, env.physical_state_dim :],
        )

        physical_labels = [f.name for f in dataclasses.fields(env.PhysicalState)]
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
    def from_model(cls, model, low, high, points_per_dim):
        raise NotImplementedError()


class CurvatureVectorField(VectorField):

    @classmethod
    def from_env(cls, env, low, high, points_per_dim):
        pass

    @classmethod
    def from_model(cls, model, low, high, points_per_dim):
        raise NotImplementedError()


class LolaVectorField(VectorField):
    @classmethod
    def from_env(cls, env, low, high, points_per_dim):
        pass

    @classmethod
    def from_model(cls, model, low, high, points_per_dim):
        raise NotImplementedError()

    def from_data(cls, data, low, high, points_per_dim):
        raise NotImplementedError()
