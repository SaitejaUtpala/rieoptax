import importlib
import re
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from chex import Array, ArrayTree
from flax.core import FrozenDict
from flax.traverse_util import (
    _get_params_dict,
    _sorted_items,
    empty_node,
    flatten_dict,
    unflatten_dict,
)
from jax import tree_util
from rieoptax.geometry.euclidean import Euclidean
from typing_extensions import Protocol

PyTree = Any
Shape = Sequence[int]

OptState = ArrayTree
Params = ArrayTree
Updates = Params


def obj_from_str(
    module_name: str,
    class_name: str,
    str_params: str,
    module_name_base="rieoptax.geometry.",
):
    m = importlib.import_module(module_name_base + module_name)
    c = getattr(m, class_name)
    obj = c.from_str(*str_params.split(","))
    return obj


def get_product_manifold(
    nfold_module_name,
    nfold_class_name,
    manifold_obj,
    n,
    module_name_base="rieoptax.geometry.",
):
    m = importlib.import_module(module_name_base + nfold_module_name)
    c = getattr(m, nfold_class_name)
    product_manfiold_obj = c(manifold_obj, n)
    return product_manfiold_obj


def construct_manifold_obj(param_name: str):
    if "@" not in param_name:
        return Euclidean()
    
    regex = r"\b[\w,\-\s.]+\b"
    matches = re.findall(regex, param_name)
    if len(matches) == 7:
        # nfold manifold case
        nfold_module_name, nfold_class_name, nfold_n = (
            matches[1],
            matches[2],
            matches[-1],
        )
        manifold_module_name, manifold_class_name, manifold_params_name = (
            matches[3],
            matches[4],
            matches[5],
        )
        manifold_obj = obj_from_str(
            manifold_module_name, manifold_class_name, manifold_params_name
        )
        product_manifold = get_product_manifold(
            nfold_module_name, nfold_class_name, manifold_obj, int(nfold_n)
        )
        return product_manifold

    else:
        # normal case
        manifold_module_name, manifold_class_name, manifold_params_name = (
            matches[1],
            matches[2],
            matches[3],
        )
        manifold_obj = obj_from_str(
            manifold_module_name, manifold_class_name, manifold_params_name
        )
        return manifold_obj


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
    def __call__(self, params: Params) -> OptState:  # type: ignore
        "The `init` function"


class TransformUpdateFn(Protocol):
    def __call__(
        self, updates: Updates, state: Updates, params: Optional[Params] = None
    ) -> Tuple[Updates, OptState]:  # type: ignore
        """The `update` function."""


class RiemannianGradientTransformation(NamedTuple):
    """Riemannian Gradient transformation."""

    init: TransformInitFn
    update: TransformUpdateFn


class RiemannianEmptyState(NamedTuple):
    """An empty state for all Riemannian gradient transformations."""

    manifold_dict: FrozenDict[str, Any]
