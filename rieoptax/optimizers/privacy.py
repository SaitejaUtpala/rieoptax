import imp
from typing import NamedTuple

import jax
from rieoptax.core import RiemannianGradientTransformation
from jax import numpy as jnp


def per_example_global_norm_clip(params, grads, clip_norm):
    """Applied per-example gradient clipping."""
    manifold = params.manifold
    norms = manifold.norm(params, grads[0])
    divisors = jnp.maximum(norms/clip_norm, 1.0)
    num_clipped = jnp.greater(divisors, 1.0).sum()
    clipped_sum = (jnp.moveaxis(grads[0], 0, -1) / divisors).sum(-1)
    return [clipped_sum], num_clipped


class DifferentiallyPrivateAggregateState(NamedTuple):
  """State containing PRNGKey for `differentially_private_aggregate`."""
  rng_key: jnp.array



def differentially_private_aggregate(
    norm_clip: float,
    sigma: float,
    seed: int
) -> RiemannianGradientTransformation:
  """Aggregates gradients based on the DP_RSGD algorithm."""

  def init_fn(params):
    del params
    return DifferentiallyPrivateAggregateState(rng_key=jax.random.PRNGKey(seed))

  def update_fn(updates, state, params):
    manifold = params.manifold
    grads_flat, grads_treedef = jax.tree_util.tree_flatten(updates)
    bsize = grads_flat[0].shape[0]
    clipped, num_clipped = per_example_global_norm_clip(params, grads_flat, norm_clip)
    new_key, *rngs = jax.random.split(state.rng_key, len(grads_flat) + 1)
    noised = [(manifold.tangent_gaussian(r, params.value, g, sigma)) for g, r in zip(clipped, rngs)]
    return (noised[0], DifferentiallyPrivateAggregateState(rng_key=new_key))

  return RiemannianGradientTransformation(init_fn, update_fn)