from typing import NamedTuple

from jax import grad
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
from typing_extensions import Protocol


@register_pytree_node_class
class ManifoldArray:
    """A lightweight wrapper for arrays constrained to mainfold. 
    It combines the `value` (a JAX PyTree) with a corresponding 'manifold'
    (Manifold object).
    """
    def __init__(self, value : jnp.array, manifold = None) -> None:
        self.value = value
        self.manifold = manifold

    def __repr__(self) -> str:
        return f"ManifoldParameter(value={self.value}, " \
               f"manifold={self.manifold})" \

    def tree_flatten(self):
        children = (self.value,)
        aux_data = self.manifold
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], aux_data)


@register_pytree_node_class
class TangentArray():
    """A lightweight wrapper for arrays constrained to Tangent. 
    It combines the `value` (a JAX PyTree) with a corresponding 'manifold'
    (Manifold object).
    """

    def __init__(self, value : jnp.array, ba) -> None: 
        self.value = value
        self.ba = manifold

def rgrad(f):
    "Riemannian Gradient Operator"
    def _temp(*args, **kwargs):
        g = args[0].manifold.egrad_to_rgrad(args[0],grad(f)(*args, **kwargs))
        g = ManifoldArray(g, args[0].manifold)
        return g 
    return _temp


class TransformInitFn(Protocol):

    def __call__(self, params):
        "The `init` function"    
   
class TransformUpdateFn(Protocol):
    def __call__(self, updates, state, params) :
        """The `update` function."""


class EmptyState(NamedTuple):
    """An empty state for the simplest stateless transformations."""

class RiemannianGradientTransformation(NamedTuple):

    init: TransformInitFn
    update: TransformUpdateFn
