import importlib
from chex import Array
from typing import Callable, NamedTuple, Any, Dict, List
from typing_extensions import Protocol

from flax.core import FrozenDict
from flax.traverse_util import (_get_params_dict, _sorted_items, empty_node,
                                flatten_dict, unflatten_dict)
from rieoptax.geometry.euclidean import Euclidean


def construct_manifold_obj(param_name: str):
    if "@" not in param_name:
        return Euclidean()
    first_split = param_name.split("@")
    module_name = "rieoptax.geometry." + first_split[1]
    mo = first_split[2].split("(")
    manifold_name = mo[0]
    manifold_params = mo[1][:-1].split(",")
    return obj_from_str(module_name, manifold_name, manifold_params)


def obj_from_str(module_name: str, class_name: str, str_params: List[str]):

    m = importlib.import_module(module_name)
    c = getattr(m, class_name)
    obj = c.from_str(*str_params)
    return obj


def get_manifold_dict(inputs: Dict[str, Any]) -> Dict[str, Any]:
    params = _get_params_dict(inputs)
    flat_dict = flatten_dict(params, keep_empty_nodes=True)
    new_dict = {}
    for key, value in _sorted_items(flat_dict):
        if value is not empty_node:
            path = "/" + "/".join(key)
            name = key[-1]
            new_dict[key] = construct_manifold_obj(name)
    new_params = unflatten_dict(new_dict)

    if isinstance(inputs, flax.core.FrozenDict):
        return FrozenDict(new_params)
    else:
        return new_params

from flax.struct import PyTreeNode, field
from flax.core import FrozenDict
from rieoptax.core import RiemannianGradientTransformation
from typing import Optional, Any, Callable
from jax import tree_util 
OptState = Any

class RtxTrainState(PyTreeNode):
  
    step: int
    apply_fn: Callable = field(pytree_node=False)
    params: FrozenDict[str, Any]
    rtx: RiemannianGradientTransformation = field(pytree_node=False)
    opt_state: OptState
    manifold_dict : FrozenDict[str, Any] = field(pytree_node=False)
    #use_exp : bool = True

    def apply_gradients(self, *, egrads, **kwargs):
        rgrads = rgrad_from_egrad(self.params,egrads,self.manifold_dict)
        updates, new_opt_state = self.rtx.update(rgrads, self.opt_state, self.params)
        new_params = apply_updates(self.params, updates, self.manifold_dict)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, rtx, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = rtx.init(params)
        manifold_dict = get_manifold_dict(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            rtx=rtx,
            opt_state=opt_state,
            manifold_dict=manifold_dict,
            **kwargs,
        )

class TransformInitFn(Protocol):
    def __call__(self, params):
        "The `init` function"


class TransformUpdateFn(Protocol):
    def __call__(self, updates, state, params):
        """The `update` function."""


class RiemannianGradientTransformation(NamedTuple):
    """Riemannian Gradient transformation."""

    init: TransformInitFn
    update: TransformUpdateFn


class RiemannianEmptyState(NamedTuple):
  """An empty state for all Riemannian gradient transformations."""

  manifold_dict: FrozenDict[str, Any]



