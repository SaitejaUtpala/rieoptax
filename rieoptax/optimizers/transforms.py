import functools
from typing import Any, Callable, NamedTuple, Optional

import jax
from rieoptax.core import EmptyState, ManifoldArray, RiemannianGradientTransformation

ScaleState = EmptyState()




class TraceState(NamedTuple):
  """Holds an aggregation of past updates."""
  trace: ManifoldArray 




class EmptyState(NamedTuple):
  """An empty state for the simplest stateless transformations."""

ScaleState = EmptyState

def scale(step_size) :
  """Scale updates by some fixed scalar `step_size`."""

  def init_fn(params):
    del params 
    return ScaleState()

  def update_fn(updates, state, params=None, manifold_dict=None):
    del params
    del manifold_dict
    updates = step_size*updates.value
    return updates, state

  return RiemannianGradientTransformation(init_fn, update_fn)


def update_moment(updates, moments, decay):
    """Compute the exponential moving average of the `order`-th moment."""
    return jax.tree_util.tree_map(
        lambda g, t: (1 - decay) * g + decay * t, updates, moments
    )


def update_moment_per_metric_norm(
    updates, moments, decay, manifold_dict, params
):
    """Compute the exponential moving average of squared Riemannain metric norm."""

    def squared_metric_norm(g, m, p):
        """Calculates Riemmanian metric norm of tangent vector 'g' at base point 'p'."""
        return m.norm(p, g)

    return jax.tree_util.tree_map(
        lambda g, t, m, x: (1 - decay) * squared_metric_norm(g, m, p) + decay * t,
        updates,
        moments,
        manifold_dict,
        params,
    )



def scale_by_radam(
    b1: float = 0.9, b2: float = 0.99, eps: float = 1e-8, eps_root: float = 0.0
) -> RiemannianGradientTransformation:
    """Rescale updates according to the Adam algorithm.
    References:
      [Gary Becigneul et al, 2019](https://arxiv.org/abs/1810.00760)
    Args:
      b1: Decay rate for the exponentially weighted average of grads.
      b2: Decay rate for the exponentially weighted average of squared
        Riemannain metric norm of grads at base point.
      eps: Term added to the denominator to improve numerical stability.
      eps_root: Term added to the denominator inside the square-root to improve
        numerical stability when backpropagating gradients through the rescaling.
    Returns:
      A `RiemannianGradientTransformation` object.
    """

    def init_fn(params):
        mu = jax.tree_util.tree_map(  # First moment
            lambda t: jnp.zeros_like(t, dtype=mu_dtype), params
        )
        nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
        return ScaleByRadamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params, manifold_dict):
        mu = update_moment(updates, state.mu, b1)
        nu = update_moment_per_elem_norm(
            updates, state.nu, b2, manifold_dict, params
        )
        count_inc = numerics.safe_int32_increment(state.count)
        mu_hat = bias_correction(mu, b1, count_inc)
        nu_hat = bias_correction(nu, b2, count_inc)
        updates = jax.tree_util.tree_map(
            lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat
        )
        mu = utils.cast_tree(mu, mu_dtype)
        return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)





def scale_by_learning_rate(learning_rate, flip_sign=True):
    m = -1 if flip_sign else 1
    return scale(m * learning_rate)



  