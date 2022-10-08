import jax
from core import ManifoldArray



def apply_update(params, updates):
    """Applies an update to the corresponding parameters by using
    Riemannian Exponential Map and returns updated parameters on manifold."""
    return ManifoldArray(params.manifold.retr(params.value,updates), params.manifold)
