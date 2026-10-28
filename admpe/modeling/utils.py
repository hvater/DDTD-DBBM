from typing import Callable

import jax
import equinox as eqx


def count_array_parameters(model: eqx.Module, filter: Callable = eqx.is_array_like) -> int:
    """Returns the number of Parameters in the model by summing the sizes of the jax.Arrays.
    That is, all parameters of the model must be jax.Arrays for this function to work!
    """
    return sum([p.size if hasattr(p, "size") else 1 for p in jax.tree.leaves(eqx.filter(model, filter, None))])
