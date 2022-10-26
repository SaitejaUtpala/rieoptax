from typing import Callable

from flax import linen as nn
from jax import numpy as jnp

from rieoptax.geometry.hyperbolic import PoincareBall
from chex import Array 

class PoincareDense(nn.Module):
    """A poincare dense layer applied over the last dimension of the input.

    Attributes:
        features: the number of output features.
        curv: curvature of the poincare manifold.
        use_bias: whether to add a bias to the output.
        kernel_init: initializer function for the weight matrix.
        bias_init: initializer function for the bias.
    """
    features: int
    curv: float = -1.0
    use_bias: bool = True
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs : Array) -> Array:
        kernel = self.param(
            "kernel", self.kernel_init, (inputs.shape[-1], self.features)
        )
        bias = self.param("bias@PoincareBall", self.bias_init, (self.features,))
        y = PoincareBall(self.features, self.curv).mobius_matvec(kernel, inputs) 
        if self.use_bias:
            y = y + bias
        return y

class Hypergyroplanes(nn.Module):
    normal_init : Callable = nn.intializer.lecun_normal()
    point_init : Callable = nn.initializer.lecun_normal()
    curv : float = -1.0

    @nn.compact
    def __call__(self, inputs : Array) -> float:
        manifold = PoincareBall(inputs.shape[-1])
        normal = self.param(
            "normal", self.normal_init, (inputs.shape[-1], self.features)
        )
        point = self.param("point@PoincareBall", self.point_init, (self.features,)) 

        normal_at_point = manifold.pt(manifold.ref_point, point, normal)
        norm = jnp.norm(normal_at_point)
        sub = manifold.mobius_sub(point, normal_at_point)
        sc = manifold.abs_sqrt_curv    
        dist_nomin = 2 * sc * abs(jnp.inner(sub, normal_at_point)) 
        dist_denom =  (1 - self.curv * jnp.norm(sub)**2)  * norm
        dist = 1/sc * jnp.arcsinh(dist_nomin/dist_denom)
        return dist 

class PoincareMLR(nn.Module):
    num_classes : int 

    @nn.compact
    def __call__(self, inputs : Array) -> Array:
        x = inputs
        return jnp.hstack([Hypergyroplanes()(x) for _ in range(self.num_classes)])
        