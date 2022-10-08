import jax
from combine import chain
from core import ManifoldArray, RiemannianGradientTransformation
from jax import numpy as jnp
from privacy import differentially_private_aggregate





def _scale_by_learning_rate(learning_rate, flip_sign=True):
    m = -1 if flip_sign else 1
    return scale(m * learning_rate)


def rsgd(learning_rate):
    """Riemannian stochastic gradient descent."""
    return _scale_by_learning_rate(learning_rate)

def dp_rsgd(learning_rate, norm_clip, sigma, seed):
    "Differenitally private riemannian (stochastic) gradient descent."
    return chain(
      differentially_private_aggregate(norm_clip=norm_clip,sigma=sigma,seed=seed),
      _scale_by_learning_rate(learning_rate)
  )
    
    





