from typing import Callable

from flax import linen as nn
from jax import numpy as jnp

from rieoptax.geometry.hyperbolic import PoincareBall


class PoincareDense(nn.Module):
    features: int
    curv: float = -1.0
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs):
        kernel = self.param(
            "kernel", self.kernel_init, (inputs.shape[-1], self.features)
        )
        bias = self.param("bias@PoincareBall", self.bias_init, (self.features,))
        y = PoincareBall(self.features, self.curv).mobius_matvec(kernel, inputs) + bias
        return y
