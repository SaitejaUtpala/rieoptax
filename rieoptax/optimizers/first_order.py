from combine import chain
from privacy import differentially_private_aggregate
from transforms import scale_by_learning_rate, variance_reduction


def rsgd(learning_rate):
    """Riemannian stochastic gradient descent."""
    return scale_by_learning_rate(learning_rate)

def rasa(learning_rate, beta):
  """Riemannian adaptive stochastic algorithm."""
  return chain(variance_reduction(), scale_by_learning_rate(learning_rate))

def rsvrg(learning_rate):
    """Riemannain stochastic variance reduction gradient descent."""
    return chain(variance_reduction(), scale_by_learning_rate(learning_rate))

def dp_rsgd(learning_rate, norm_clip, sigma, seed):
    "Differenitally private riemannian (stochastic) gradient descent."
    return chain(
        differentially_private_aggregate(norm_clip=norm_clip, sigma=sigma, seed=seed),
        scale_by_learning_rate(learning_rate),
    )