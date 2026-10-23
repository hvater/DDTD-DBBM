from typing import Callable

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx


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

        return CartesianGridSet(
            axes=self.axes,
            mask=jnp.logical_and(self.mask, other.mask),
        )

    @staticmethod
    def from_path(path):
        return load_safely_excitable_set(path)

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


def load_safely_excitable_set(path):
    """Load a compact archive and transfer its arrays to JAX."""
    with np.load(Path(path), allow_pickle=False) as archive:
        if archive["format_version"].item() != 1:
            raise ValueError("Unsupported set format version.")

        mask = archive["mask"]

        axes = tuple(archive[f"axis_{i}"] for i in range(mask.ndim))

    return CartesianGridSet(axes, mask)
