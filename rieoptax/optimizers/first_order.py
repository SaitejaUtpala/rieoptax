from rieoptax.core import RiemannianGradientTransformation

from rieoptax.optimizers.combine import chain
from rieoptax.optimizers.privacy import differentially_private_aggregate
from rieoptax.optimizers.transforms import scale_by_learning_rate, scale_by_adam


def rsgd(learning_rate: float) -> RiemannianGradientTransformation:
    """Riemannian stochastic gradient descent."""
    return scale_by_learning_rate(learning_rate)

def radam(
    learning_rate: float,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    amgs_grad: bool = False,
) -> RiemannianGradientTransformation:
    """Riemannian adaptive moment estimation (ADAM)."""
    return chain(
        scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root, ams_grad=ams_grad),
        scale_by_learning_rate(learning_rate),
    )

def dp_rsgd(
    learning_rate: float, norm_clip: float, sigma: float, seed: int
) -> RiemannianGradientTransformation:
    "Differenitally private riemannian (stochastic) gradient descent."
    return chain(
        differentially_private_aggregate(norm_clip=norm_clip, sigma=sigma, seed=seed),
        scale_by_learning_rate(learning_rate),
    )
