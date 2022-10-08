from combine import chain
from privacy import differentially_private_aggregate
from transforms import scale_by_learning_rate


def rsgd(learning_rate):
    """Riemannian stochastic gradient descent."""
    return scale_by_learning_rate(learning_rate)


def dp_rsgd(learning_rate, norm_clip, sigma, seed):
    "Differenitally private riemannian (stochastic) gradient descent."
    return chain(
        differentially_private_aggregate(norm_clip=norm_clip, sigma=sigma, seed=seed),
        scale_by_learning_rate(learning_rate),
    )
