from typing import Callable

import jax
import jax.numpy as jnp

from dmpe.utils.density_estimation import DensityEstimate

from admpe.vector_fields import VectorField


class TargetDensity(DensityEstimate):

    @classmethod
    def from_vector_field(
        cls,
        vf: VectorField,
        bandwidth: float,
        penalty_function: Callable,
        consider_action_distribution: bool,
        obs_dim: int,
        act_dim: int,
    ):

        z_g = vf.z

        if consider_action_distribution:
            constr_func = lambda z_g: penalty_function(z_g[..., None, :obs_dim], z_g[..., None, -act_dim:])
        else:
            constr_func = lambda z_g: penalty_function(z_g[..., None, :obs_dim], None)

        valid_grid_point = jax.vmap(constr_func, in_axes=0)(z_g) == 0
        constrained_data_points = z_g[jnp.where(valid_grid_point == True)]

        uniform_target_density = DensityEstimate.from_dataset(
            constrained_data_points[None],
            z_min=float(vf.z.min()),
            z_max=float(vf.z.max()),
            points_per_dim=vf._points_per_dim,
            bandwidth=bandwidth,
        )

        assert jnp.all(uniform_target_density.z_g == vf.z), "grids for vector_field and density differ"

        adapted_density_values = uniform_target_density.p * jnp.linalg.norm(vf.values, axis=-1)[:, None]
        adapted_density_values = adapted_density_values / jnp.sum(adapted_density_values)

        # uniform_density_values = uniform_target_density.p / jnp.sum(uniform_target_density.p)
        # adapted_density_values = (adapted_density_values + uniform_density_values) / 2

        adapted_target_density = DensityEstimate.from_estimate(
            p=adapted_density_values,
            n_additional_observations=0,
            density_estimate=uniform_target_density,
        )

        return adapted_target_density
