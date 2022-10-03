from jax import grad
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class


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

    def tree_flatten(self) -> tuple(jnp.array, str):
        children = (self.value,)
        aux_data = self.manifold
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], aux_data)


def rgrad(f):
    "Riemannian Gradient Operator"
    def _temp(mp):
        g = mp.manifold.egrad_to_rgrad(grad(f)(mp))
        return g 
    return _temp


class TransformInitFn(typing_extensions.Protocol):
  """A callable type for the `init` step of a `GradientTransformation`.
  The `init` step takes a tree of `params` and uses these to construct an
  arbitrary structured initial `state` for the gradient transformation. This
  may hold statistics of the past updates or any other non static information.
  """

  def __call__(self, params: Params) -> OptState:
    """The `init` function.
    Args:
      params: The initial value of the parameters.
    Returns:
      The initial state of the gradient transformation.
    """


class TransformUpdateFn(typing_extensions.Protocol):
  def __call__(
      self,
      updates: Updates,
      state: OptState,
      params: Optional[Params] = None
    ) -> Tuple[Updates, OptState]:
    """The `update` function.
    Args:
      updates: A tree of candidate updates.
      state: The state of the gradient transformation.
      params: (Optionally) the current value of the parameters.
    Returns:
      The transformed updates, and the updated state.
    """


class RiemannianGradientTransformation(NamedTuple):

  init: TransformInitFn
  update: TransformUpdateFn
