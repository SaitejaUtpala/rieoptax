from abc import ABC

from jax import grad, lax, jit
from typing import Callable 
from chex import Array


def straight_through_f(f: Callable) -> Callable:
    def _f(x: Array) -> Array:
        zero = x - lax.stop_gradient(x)
        return zero + lax.stop_gradient(f(x))

    return _f
