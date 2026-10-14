import numpy as np
import equinox as eqx


def convert_lists(x):
    if isinstance(x, dict):
        return {k: convert_lists(v) for k, v in x.items()}
    if isinstance(x, list):
        try:
            a = np.asarray(x)
            if np.issubdtype(a.dtype, np.number):
                return a
        except (TypeError, ValueError, OverflowError):
            return [convert_lists(v) for v in x]
    if isinstance(x, tuple):
        return tuple(convert_lists(v) for v in x)
    return x


def pprint_helper(x):
    return eqx.tree_pprint(convert_lists(x))
