from typing import Callable

from flax import linen as nn
from jax import numpy as jnp


class PoincareDense(nn.Module):
    features: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs):
        kernel = self.param('kernel',self.kernel_init, (inputs.shape[-1], self.features))
        bias = self.param('bias@PoincareBall', self.bias_init, (self.features,))
        y = jnp.dot(kernel, bias) + bias
        return y