import functools
from typing import Any, Callable, NamedTuple, Optional
from flax.core import FrozenDict
from chex import Array, ArrayTree

from jax import numpy as jnp
from jax import tree_util
from jax import jit

from rieoptax.optimizers.base import (
    RiemannianEmptyState,
    RiemannianGradientTransformation,
)


#still need to figure out it
class EmptyState(NamedTuple):
  """An empty state for the simplest stateless transformations."""

ScaleState = EmptyState

def scale(step_size):
    """Scale updates by some fixed scalar `step_size`."""

    def init_fn(params, manifold_dict=None):
        del params
        return ScaleState()

    def update_fn(updates, state, params=None):
        del params
        updates = step_size * updates
        return updates, state

    return RiemannianGradientTransformation(init_fn, update_fn)


def update_moment(updates, moments, decay):
    """Compute the exponential moving average of the first moment."""
    return tree_util.tree_map(
        lambda g, t: (1 - decay) * g + decay * t, updates, moments
    )


def update_moment_per_metric_norm(updates, moments, decay, manifold_dict, params):
    """Compute the exponential moving average of second moment of
    squared Riemannain metric norm."""

    def squared_metric_norm(g, m, p):
        """Calculates Riemmanian metric norm of tangent vector 'g' at base point 'p'."""
        return m.norm(p, g)

    return tree_util.tree_map(
        lambda g, t, m, p: (1 - decay) * squared_metric_norm(g, m, p) + decay * t,
        updates,
        moments,
        manifold_dict,
        params,
    )


@functools.partial(jit, inline=True)
def bias_correction(moment, decay, count):
    """Performs bias correction. It becomes a no-op as count goes to infinity."""
    bias_correction_ = 1 - decay**count

    # Perform division in the original precision.
    return tree_util.tree_map(lambda t: t / bias_correction_.astype(t.dtype), moment)


class ScaleByRadamState(NamedTuple):
    """State for the Riemannian Adam algorithm."""

    manifolds: FrozenDict[str, Any]
    count: Array  # shape=(), dtype=jnp.int32.
    mu: ArrayTree
    nu: ArrayTree
    params_prev: ArrayTree


# def scale_by_radam(
#     b1: float = 0.9,
#     b2: float = 0.99,
#     eps: float = 1e-8,
#     eps_root: float = 0.0,
#     ams_grad: bool = False,
# ) -> RiemannianGradientTransformation:
#     """Rescale updates according to the Adam algorithm.
#     References:
#       [Gary Becigneul et al, 2019](https://arxiv.org/abs/1810.00760)
#     Args:
#       b1: Decay rate for the exponentially weighted average of grads.
#       b2: Decay rate for the exponentially weighted average of squared
#         Riemannain metric norm of grads at base point.
#       eps: Term added to the denominator to improve numerical stability.
#       eps_root: Term added to the denominator inside the square-root to improve
#         numerical stability when backpropagating gradients through the rescaling.
#       ams_grad: Perform update according to ams_grad version of radam.
#         see [Sashank J. Reddi et al, 2019](https://arxiv.org/abs/1904.09237).
#     Returns:
#       A `RiemannianGradientTransformation` object.
#     """

#     def init_fn(params, manifolds):
#         mu = tree_util.tree_map(  # First moment
#             lambda t: jnp.zeros_like(t, dtype=mu_dtype), params
#         )
#         nu = tree_util.tree_map(jnp.zeros_like, params)  # Second moment
#         return ScaleByRadamState(
#             manifolds=manifolds,
#             count=jnp.zeros([], jnp.int32),
#             mu=mu,
#             nu=nu,
#             params_prev=params,
#         )

#     def update_fn(updates, state, params, manifolds):
#         # key difference between Riemannian and Eucliean is that
#         # first moment has to transportation to current_params.

#         tau = tree_util.tree_map(
#             lambda mo, m, p_prev, p_curr: mo.ptrans(p_prev, p_curr, m),
#             manifolds,
#             state.mu,
#             state.params_prev,
#             params,
#         )
#         mu = update_moment(updates, tau, b1)
#         nu = update_moment_per_metric_norm(updates, state.nu, b2, manifolds, params)
#         nu = jnp.max(nu, state.nu) if ams_grad else nu

#         count_inc = count_inc + jnp.array(1, dtype=jnp.int32)
#         mu = bias_correction(mu, b1, count_inc)
#         nu = bias_correction(nu, b2, count_inc)
#         updates = tree_util.tree_map(
#             lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu, nu
#         )
#         return updates, ScaleByRadamState(
#             count=count_inc, mu=mu, nu=nu, params_prev=params
#         )


def scale_by_learning_rate(learning_rate, flip_sign=True):
    m = -1 if flip_sign else 1
    return scale(m * learning_rate)
