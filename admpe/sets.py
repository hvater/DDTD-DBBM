from typing import Callable, Self

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from admpe.data_management import SAFELY_EXCITABLE_SETS_ROOT


class CartesianGridSet(eqx.Module):

    axes: tuple[jax.Array, ...]
    mask: jax.Array

    def __init__(self, axes, mask):
        axes = tuple(np.asarray(axis, dtype=np.float32) for axis in axes)
        mask = np.asarray(mask)

        if not axes or mask.dtype != np.bool_:
            raise ValueError("Provide coordinate axes and a Boolean mask.")

        for axis in axes:
            if axis.ndim != 1 or axis.size == 0 or not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0):
                raise ValueError("Axes must be finite, nonempty, and strictly ascending.")

        expected_shape = tuple(axis.size for axis in axes)
        if mask.shape != expected_shape:
            raise ValueError("Mask shape must match the coordinate axes.")

        self.axes = tuple(jnp.asarray(axis) for axis in axes)
        self.mask = jnp.asarray(mask)

    @property
    def dim(self):
        return len(self.axes)

    @eqx.filter_jit
    def contains(self, points):
        """
        Check membership for points shaped (..., dim).

        Returns a Boolean array shaped (...,).
        """
        points = jnp.asarray(points)

        if points.ndim == 0 or points.shape[-1] != self.dim:
            raise ValueError(f"Expected points with final dimension {self.dim}.")

        finite = jnp.all(jnp.isfinite(points), axis=-1)
        values = jnp.where(jnp.isfinite(points), points, 0.0)

        indices = []

        for i, axis in enumerate(self.axes):
            x = values[..., i]

            right = jnp.searchsorted(axis, x, side="left")
            right = jnp.clip(right, 0, axis.size - 1)
            left = jnp.maximum(right - 1, 0)

            distance_left = jnp.abs(axis[left] - x)
            distance_right = jnp.abs(axis[right] - x)

            # Prefer the lower index at an exact midpoint.
            nearest = jnp.where(
                distance_right < distance_left,
                right,
                left,
            )
            indices.append(nearest)

        lower = jnp.stack([axis[0] for axis in self.axes])
        upper = jnp.stack([axis[-1] for axis in self.axes])

        excess = jnp.maximum(lower - values, 0.0) + jnp.maximum(values - upper, 0.0)

        # Retain DMPE's soft boundary tolerance for these [-1, 1] axes.
        penalty = jnp.sum(excess**2, axis=-1)
        in_bounds = jnp.isclose(penalty, 0.0)

        return finite & in_bounds & self.mask[tuple(indices)]

    def __and__(self, other):
        assert jnp.all(self.axes == other.axes), "The Sets have to be discretized on the same grid."

        assert self.mask.shape == other.mask.shape, ("The unflattened shape of the Sets must be the same.",)

        return self.__class__(
            axes=self.axes,
            mask=jnp.logical_and(self.mask, other.mask),
        )

    @classmethod
    def from_path(cls, path: Path | str) -> Self:
        with np.load(Path(path), allow_pickle=False) as archive:
            if archive["format_version"].item() != 1:
                raise ValueError("Unsupported set format version.")

            mask = archive["mask"]
            axes = tuple(archive[f"axis_{i}"] for i in range(mask.ndim))

        return cls(axes, mask)

    def sample(self, key, *, max_attempts=5000):
        return sample_from_set(key, self, max_attempts=max_attempts)

    def visualize(
        self,
        reduction_method: Callable = jnp.sum,
        labels: None | list[str] = None,
        use_contourf: bool = True,
        grid_spacing: float = 0.0,
    ):
        dim = self.dim
        coordinates = [np.asarray(axis) for axis in self.axes]

        if labels is None:
            labels = [str(i) for i in range(dim)]

        if len(labels) != dim:
            raise ValueError(f"Expected {dim} labels")

        if grid_spacing < 0:
            raise ValueError("grid_spacing must be nonnegative")

        def draw_plane(ax, x_dimension, y_dimension, values):
            x_coordinates = coordinates[x_dimension]
            y_coordinates = coordinates[y_dimension]
            values = np.asarray(values)

            if use_contourf:
                ax.contourf(
                    x_coordinates,
                    y_coordinates,
                    values,
                )
            else:
                ax.pcolormesh(
                    x_coordinates,
                    y_coordinates,
                    values,
                    shading="nearest",
                )

            ax.set_xlim(
                x_coordinates[0] - grid_spacing,
                x_coordinates[-1] + grid_spacing,
            )
            ax.set_ylim(
                y_coordinates[0] - grid_spacing,
                y_coordinates[-1] + grid_spacing,
            )
            ax.grid(True)

        if dim == 1:
            fig, ax = plt.subplots(figsize=(9, 9))

            ax.plot(coordinates[0], np.asarray(self.mask))
            ax.set_xlabel(labels[0])
            ax.set_ylabel("Membership")
            ax.grid(True)

            fig.tight_layout()
            return fig, ax

        if dim == 2:
            fig, ax = plt.subplots(figsize=(9, 9))

            draw_plane(ax, 0, 1, self.mask.T)

            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])

            fig.tight_layout()
            return fig, ax

        fig, axs = plt.subplots(
            nrows=dim,
            ncols=dim,
            figsize=(9, 9),
            sharex="col",
            sharey="row",
        )

        for row in range(dim):
            for column in range(dim):
                ax = axs[row, column]
                ax.grid(True)

                # Leave diagonal panels empty.
                if row == column:
                    continue

                reduction_axes = []

                for dimension in range(dim):
                    if dimension != column and dimension != row:
                        reduction_axes.append(dimension)

                values = reduction_method(
                    self.mask,
                    axis=tuple(reduction_axes),
                )

                # Reduction preserves the original dimension order.
                # Plotting requires [vertical_index, horizontal_index].
                if column < row:
                    values = values.T

                draw_plane(
                    ax,
                    x_dimension=column,
                    y_dimension=row,
                    values=values,
                )

        for dimension in range(dim):
            axs[-1, dimension].set_xlabel(labels[dimension])
            axs[dimension, 0].set_ylabel(labels[dimension])

        fig.tight_layout()
        return fig, axs


@eqx.filter_jit
def sample_from_set(
    key: jax.Array,
    set: CartesianGridSet,
    *,
    max_attempts: int = 5000,
):
    """Return a continuous sample and a Boolean indicating success."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")

    dtype = jnp.asarray(0.0).dtype
    lower = jnp.stack([axis[0] for axis in set.axes]).astype(dtype)
    upper = jnp.stack([axis[-1] for axis in set.axes]).astype(dtype)

    def condition(carry):
        _, _, accepted, attempts = carry
        return (~accepted) & (attempts < max_attempts)

    def draw(carry):
        key, _, _, attempts = carry
        key, draw_key = jax.random.split(key)

        point = jax.random.uniform(
            draw_key,
            shape=(set.dim,),
            dtype=dtype,
            minval=lower,
            maxval=upper,
        )

        return key, point, set.contains(point), attempts + 1

    _, point, accepted, _ = jax.lax.while_loop(
        condition,
        draw,
        (key, jnp.zeros_like(lower), jnp.array(False), jnp.array(0)),
    )

    return point, accepted


class SafelyExcitableSet(CartesianGridSet):

    @classmethod
    def from_system_name(cls, system_name: str) -> Self:
        filenames = {
            "fluid_tank": "fluid_tank_S_xu_666a769d-8c1b-4b.npz",
            "pendulum": "pendulum_S_xu_02430b86-ae0d-42.npz",
            "cart_pole": "cart_pole_S_xu_8744b5d5-30e6-4b.npz",
        }
        filename = filenames[system_name]

        path = SAFELY_EXCITABLE_SETS_ROOT / filename
        return cls.from_path(path), path


class SafeStateSet(CartesianGridSet):

    @classmethod
    def from_safely_excitable_set(
        cls,
        safely_excitable_set: SafelyExcitableSet,
        obs_dim: int,
    ) -> Self:

        if not 0 < obs_dim < safely_excitable_set.dim:
            raise ValueError("Expected state axes followed by action axes.")

        action_axes = tuple(range(obs_dim, safely_excitable_set.dim))

        state_mask = np.any(
            np.asarray(safely_excitable_set.mask),
            axis=action_axes,
        )

        return cls(
            axes=safely_excitable_set.axes[:obs_dim],
            mask=state_mask,
        )
