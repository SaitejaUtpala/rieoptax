import importlib

from jax import grad, lax
from jax import numpy as jnp

from flax.traverse_util import (
    _get_params_dict,
    _sorted_items,
    empty_node,
    flatten_dict,
    unflatten_dict,
)

from typing_extensions import Protocol
from rieoptax.geometry.euclidean import Euclidean

OptState = Any




def straight_through_f(f: Callable) -> Callable:
    def _f(x: Array) -> Array:
        zero = x - lax.stop_gradient(x)
        return zero + lax.stop_gradient(f(x))

    return _f



