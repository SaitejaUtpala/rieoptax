from jax import numpy as jnp


def per_example_gradient_clipping(params, grads, clip_norm):
    """Applied per-example gradient clipping."""
    manifold = params.manifold
    norms = manifold.norm(grads)
    divisors = jnp.maximum(norms / clip_norm, 1.0)
    num_clipped = jnp.greater(divisors, 1.0).sum()
    clipped_sum = (jnp.moveaxis(grads, 0, -1) / divisors).sum(-1) 
    return clipped_sum, num_clipped