import pathlib
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx


def resolve_load_dtype(dtype):
    if dtype is None:
        return None
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be float32, float64, or None.")
    if dtype == np.dtype("float64") and not jax.config.jax_enable_x64:
        raise ValueError(
            "Loading float64 requires jax_enable_x64=True. "
            "Enable it in the notebook or experiment runner before loading."
        )
    return dtype


def _prepare_jax_array(array, dtype):
    array = np.asarray(array)

    if dtype is None:
        return array

    if np.issubdtype(array.dtype, np.floating):
        return array.astype(dtype, copy=False)

    if np.issubdtype(array.dtype, np.integer):
        target = np.dtype(jax.dtypes.canonicalize_dtype(array.dtype))

        if target != array.dtype:
            limits = np.iinfo(target)
            if np.any(array < limits.min) or np.any(array > limits.max):
                raise ValueError(f"Cannot convert {array.dtype} to {target}: " "integer value out of range.")
            return array.astype(target, copy=False)

    return array


def _check_jax_dtype_available(dtype):
    dtype = np.dtype(dtype)
    supported_dtype = np.dtype(jax.dtypes.canonicalize_dtype(dtype))
    if supported_dtype != dtype:
        raise ValueError(
            f"Cannot preserve {dtype} while jax_enable_x64=False. "
            "Enable x64, or request dtype=jnp.float32 for floating arrays."
        )


def load_jax_array(array, dtype):
    array = _prepare_jax_array(array, dtype)
    _check_jax_dtype_available(array.dtype)
    return jnp.asarray(array)


def _dtype_filter_spec(stream, leaf, *, dtype):
    """Equinox filter that converts floating arrays before JAX sees them."""
    if isinstance(leaf, (jax.Array, jax.ShapeDtypeStruct)):
        return load_jax_array(np.load(stream, allow_pickle=False), dtype)
    return eqx.default_deserialise_filter_spec(stream, leaf)


def load_eqx_with_dtype(path, template, dtype):
    """Align template dtypes, then use Equinox's public deserializer."""
    leaves, structure = jax.tree_util.tree_flatten(template)

    with pathlib.Path(path).open("rb") as stream:

        def align_leaf(leaf):
            if isinstance(leaf, (jax.Array, jax.ShapeDtypeStruct, np.ndarray)):
                saved = np.load(stream, allow_pickle=False)
                if isinstance(leaf, np.ndarray):
                    return np.empty(leaf.shape, dtype=saved.dtype)

                prepared = _prepare_jax_array(saved, dtype)

                _check_jax_dtype_available(prepared.dtype)
                return jax.ShapeDtypeStruct(leaf.shape, prepared.dtype)

            eqx.default_deserialise_filter_spec(stream, leaf)
            return leaf

        aligned = structure.unflatten([align_leaf(leaf) for leaf in leaves])
        if stream.read(1):
            raise ValueError(f"{path}: checkpoint contains extra leaves.")

        stream.seek(0)
        restored = eqx.tree_deserialise_leaves(
            stream,
            aligned,
            filter_spec=partial(_dtype_filter_spec, dtype=dtype),
        )

    return restored
