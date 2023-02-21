import importlib
from chex import Array
from typing import Callable, NamedTuple, Any, Dict, List
from typing_extensions import Protocol
from jax import tree_util

from flax.core import FrozenDict
from flax.traverse_util import (
    _get_params_dict,
    _sorted_items,
    empty_node,
    flatten_dict,
    unflatten_dict,
)
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

    if isinstance(inputs, FrozenDict):
        return FrozenDict(new_params)
    else:
        return new_params


def rgrad_from_egrad(
    params: FrozenDict[str, Any],
    egrads: FrozenDict[str, Any],
    manifold_dict: FrozenDict[str, Any],
):
    rgrads = tree_util.tree_map(
        lambda param, egrad, manifold: manifold.egrad_to_rgrad(param, egrad),
        params,
        egrads,
        manifold_dict,
    )
    return rgrads


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
