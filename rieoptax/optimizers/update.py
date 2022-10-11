from rieoptax.core import ManifoldArray


def apply_updates(params, updates):
    """Applies an update to the corresponding parameters by using
    Riemannian Exponential Map and returns updated parameters on manifold."""
    return ManifoldArray(params.manifold.exp(params.value,updates), params.manifold)
