from flax.struct import PyTreeNode, field
from flax.core import FrozenDict
from rieoptax.optimizers.base import (
    RiemannianGradientTransformation,
    get_manifold_dict,
    rgrad_from_egrad,
)
from rieoptax.optimizers.update import apply_updates
from typing import Optional, Any, Callable

OptState = Any


class RtxTrainState(PyTreeNode):
    step: int
    apply_fn: Callable = field(pytree_node=False)
    params: FrozenDict[str, Any]
    rtx: RiemannianGradientTransformation = field(pytree_node=False)
    opt_state: OptState
    manifold_dict: FrozenDict[str, Any] = field(pytree_node=False)
    update_fn: str = "exp"

    def apply_gradients(self, *, egrads, **kwargs):
        rgrads = rgrad_from_egrad(self.params, egrads, self.manifold_dict)
        updates, new_opt_state = self.rtx.update(rgrads, self.opt_state, self.params) 
        new_params = apply_updates(
            self.params, updates, self.manifold_dict, self.update_fn
        )
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
