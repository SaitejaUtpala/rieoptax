import importlib
from typing import Callable, NamedTuple, Any, Dict, List
import flax

from chex import Array
from jax import grad, lax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
from typing_extensions import Protocol

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


@register_pytree_node_class
class ManifoldArray:
    """A lightweight wrapper for arrays constrained to mainfold.
    It combines the `value` (a JAX PyTree) with a corresponding 'manifold'
    (Manifold object).
    """

    def __init__(self, value: jnp.array, manifold=None) -> None:
        self.value = value
        self.manifold = manifold

    def __repr__(self) -> str:
        return f"ManifoldParameter(value={self.value}, " f"manifold={self.manifold})"

    def tree_flatten(self):
        children = (self.value,)
        aux_data = self.manifold
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], aux_data)


def rgrad(f: Callable) -> Callable:
    "Riemannian Gradient Operator"

    def _temp(*args, **kwargs):
        g = args[0].manifold.egrad_to_rgrad(args[0], grad(f)(*args, **kwargs))
        g = ManifoldArray(g, args[0].manifold)
        return g

    return _temp


def straight_through_f(f: Callable) -> Callable:
    def _f(x: Array) -> Array:
        zero = x - lax.stop_gradient(x)
        return zero + lax.stop_gradient(f(x))

    return _f


class TransformInitFn(Protocol):
    def __call__(self, params):
        "The `init` function"


class TransformUpdateFn(Protocol):
    def __call__(self, updates, state, params):
        """The `update` function."""


class EmptyState(NamedTuple):
    """An empty state for the simplest stateless transformations."""


class RiemannianGradientTransformation(NamedTuple):
    """Riemannian Gradient transformation."""

    init: TransformInitFn
    update: TransformUpdateFn
