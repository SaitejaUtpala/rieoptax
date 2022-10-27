from functools import partial
from typing import Any, Callable, Optional, Tuple

from chex import Array
from flax import linen as nn
from flax.linen.activation import sigmoid, tanh
from jax import numpy as jnp

from rieoptax.geometry.hyperbolic import PoincareBall

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
zeros = nn.initializers.zeros

class PoincareDense(nn.Module):
    """A poincare dense layer applied over the last dimension of the input.

    Attributes:
        features: the number of output features.
        curv: curvature of the poincare manifold.
        use_bias: whether to add a bias to the output.
        dtype: the dtype of the computation (default: infer from input and params).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        kernel_init: initializer function for the weight matrix.
        bias_init: initializer function for the bias.
    """

    features: int
    curv: float = -1.0
    use_bias: bool = True
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = zeros

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        kernel = self.param(
            "kernel", self.kernel_init, (inputs.shape[-1], self.features)
        )
        bias = self.param("bias@PoincareBall_{self.curv}", self.bias_init, (self.features,))
        y = PoincareBall(self.features, self.curv).mobius_matvec(kernel, inputs)
        if self.use_bias:
            y = y + bias
        return y


class Hypergyroplanes(nn.Module):
    """Single Hypergyroplane.
    
    Attributes:
        features: the number of output features.
        curv: curvature of the poincare manifold.
        use_bias: whether to add a bias to the output.
        dtype: the dtype of the computation (default: infer from input and params).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        kernel_init: initializer function for the weight matrix.
        bias_init: initializer function for the bias.
    """
    
    curv: float = -1.0
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    normal_init: Callable = nn.initializers.lecun_normal()
    point_init: Callable = zeros

    @nn.compact
    def __call__(self, inputs: Array) -> float:
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
        dist_denom = (1 - self.curv * jnp.norm(sub) ** 2) * norm
        dist = 1 / sc * jnp.arcsinh(dist_nomin / dist_denom)
        return dist


class PoincareMLR(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        x = inputs
        return jnp.hstack([Hypergyroplanes()(x) for _ in range(self.num_classes)])


class PoincareGRU(nn.Module):
    """Poincare GRU cell.

    Attributes:
        gate_fn: activation function used for gates (default: sigmoid)
        activation_fn: activation function used for output and memory update
        (default: tanh).
        kernel_init: initializer function for the kernels that transform
        the input (default: lecun_normal).
        recurrent_kernel_init: initializer function for the kernels that transform
        the hidden state (default: orthogonal).
        bias_init: initializer for the bias parameters (default: zeros)
    """

    gate_fn: Callable[..., Any] = sigmoid
    activation_fn: Callable[..., Any] = tanh
    kernel_init: Callable = nn.initializers.lecun_normal()
    recurrent_kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = zeros
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, carry, inputs):
        """Gated recurrent unit (GRU) cell.
        Args:
            carry: the hidden state of the LSTM cell,
                initialized using `GRUCell.initialize_carry`.
            inputs: an ndarray with the input for the current time step.
                All dimensions except the final are considered batch dimensions.
        Returns:
        A tuple with the new carry and the output.
        """

        h = carry
        hidden_features = h.shape[-1]
        dense_h = partial(
            PoincareDense,
            features=hidden_features,
            use_bias=False,
            kernel_init=self.recurrent_kernel_init,
            bias_init=self.bias_init,
        )
        dense_i = partial(
            PoincareDense,
            features=hidden_features,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        r = self.gate_fn(dense_i(name="ir")(inputs) + dense_h(name="hr")(h))
        z = self.gate_fn(dense_i(name="iz")(inputs) + dense_h(name="hz")(h))
        n = self.activation_fn(
            dense_i(name="in")(inputs) + r * dense_h(name="hn", use_bias=True)(h)
        )
        new_h = (1.0 - z) * n + z * h
        return new_h, new_h

    @staticmethod
    def initialize_carry(rng, batch_dims, size, init_fn=zeros):
        """Initialize the RNN cell carry.
        Args:
            rng: random number generator passed to the init_fn.
            batch_dims: a tuple providing the shape of the batch dimensions.
            size: the size or number of features of the memory.
            init_fn: initializer function for the carry.
            Returns:
            An initialized carry for the given RNN cell.
        """
        mem_shape = batch_dims + (size,)
        return init_fn(rng, mem_shape)