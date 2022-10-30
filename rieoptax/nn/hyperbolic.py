from functools import partial
from typing import Any, Callable, Optional, Tuple

from chex import Array
from flax import linen as nn
from flax.linen.activation import sigmoid, tanh
from flax.linen.dtypes import promote_dtype
from flax.linen.initializers import orthogonal
from flax.linen.linear import default_kernel_init
from jax import numpy as jnp
from jax import random, vmap

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
    in_radii: float = 1 - 1e-8
    out_radii: float = 1e-15
    use_bias: bool = True
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    kernel_init: Callable = default_kernel_init
    bias_init: Callable = zeros

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        manifold = PoincareBall(self.features, self.curv)
        regularize = vmap(manifold.regularize)
        mobius_matvec = vmap(manifold.mobius_matvec, in_axes=(None, 0))
        mobius_add = vmap(manifold.mobius_add, in_axes=(0, None))
        # (TODO) : make it consistent with flax kernel shape order i.e., transpose.
        kernel = self.param(
            "kernel",
            self.kernel_init,
            (self.features, inputs.shape[-1]),
            self.param_dtype,
        )
        if self.use_bias:
            bias = self.param(
                "bias@" + str(manifold),
                self.bias_init,
                (self.features,),
                self.param_dtype,
            )
        else:
            bias = None
        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
        inputs = regularize(inputs)
        y = mobius_matvec(kernel, inputs)
        if self.use_bias:
            y = mobius_add(regularize(y), bias)
        return regularize(y)


class _Hypergyroplane(nn.Module):
    """Single hypergyroplane Layer and computes logits.

    Attributes:
        curv: curvature of the poincare manifold.
        use_bias: whether to add a bias to the output.
        dtype: the dtype of the computation (default: infer from input and params).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        kernel_init: initializer function for the weight matrix.
        bias_init: initializer function for the bias.
    """

    curv: float = -1.0
    in_radii: float = 1 - 1e-8
    out_radii: float = 1e-15
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    normal_init: Callable = nn.initializers.lecun_normal()
    point_init: Callable = zeros

    @nn.compact
    def __call__(self, inputs: Array) -> float:
        input_shape = inputs.shape[-1]
        normal = self.param("normal", self.normal_init, (input_shape,))
        point = self.param("point@" + str(manifold), self.point_init, (input_shape,))
        manifold = PoincareBall(input_shape, self.curv)
        pt = manifold.pt 
        mobius_add = vmap(manifold.mobius_add, in_axes=(None,0))
        cf = manifold.cf


        sc = manifold.abs_sqrt_curv
        normal_at_point = pt(manifold.ref_point, point, normal)
        norm = jnp.linalg.norm(normal_at_point)
        add = mobius_add(point, inputs)
        dist_nomin = 2 * sc * jnp.inner(add, normal_at_point)
        dist_denom = (1 + self.curv * jnp.linalg.norm(add) ** 2) * norm
        dist = jnp.arcsinh(dist_nomin / dist_denom)
        logits =  (cf(point) * norm * dist)/sc 
        return logits


class PoincareMLR(nn.Module):
    """Poincare Multi logistiic regression layer.

    Attributes:
        curv: curvature of the poincare manifold.
        use_bias: whether to add a bias to the output.
        dtype: the dtype of the computation (default: infer from input and params).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        kernel_init: initializer function for the weight matrix.
        bias_init: initializer function for the bias.
    """
    num_classes: int
    curv: float = -1.0
    in_radii: float = 1 - 1e-8
    out_radii: float = 1e-15
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    normal_init: Callable = nn.initializers.lecun_normal()
    point_init: Callable = zeros

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        x = inputs
        return jnp.hstack(
            [
                _Hypergyroplane(
                    curv=self.curv,
                    in_radii=self.in_radii,
                    out_radii=self.out_radii,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    normal_init=self.normal_init,
                    point_int=self.point_init,
                )(x)
                for _ in range(self.num_classes)
            ]
        )


class PoincareRNNCell(nn.Module):
    """Poincare RNN cell.

    Attributes:
        curv: curvature of the poincare manifold.
        eps: eps to be added inputs, h while doing operations, done
            to increase numerical stability of operations (default: 1E-15).
        gate_fn: activation function used for gates (default: sigmoid).
        activation_fn: activation function used for output and memory update
            (default: tanh).
        kernel_init: initializer function for the kernels that transform
            the input (default: lecun_normal).
        recurrent_kernel_init: initializer function for the kernels that transform
            the hidden state (default: orthogonal).
        bias_init: initializer for the bias parameters (default: zeros)
    """

    curv: float = -1.0
    in_radii: float = 1 - 1e-8
    out_radii: float = 1e-15
    gate_fn: Callable[..., Any] = sigmoid
    activation_fn: Callable[..., Any] = tanh
    kernel_init: Callable = default_kernel_init
    recurrent_kernel_init: Callable = orthogonal()
    bias_init: Callable = zeros
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, carry: Array, inputs: Array) -> Tuple[Array, Array]:
        """Poincare RNN cell.

        Args:
            carry: the hidden state of the RNN cell,
                initialized using `PoincareRNNCell.initialize_carry`.
            inputs: an ndarray with the input for the current time step.
                All dimensions except the final are considered batch dimensions.

        Returns:
        A tuple with the new carry and the output.
        """
        h = carry + self.eps
        inputs = inputs + self.eps
        hidden_features = h.shape[-1]
        manifold = PoincareBall(hidden_features, self.curv)
        regularize = vmap(manifold.regularize)
        mobius_add = vmap(manifold.mobius_add, in_axes=(0, 0))
        mobius_gate_fn = vmap(manifold.mobius_f(self.gate_fn))

        dense = partial(
            PoincareDense,
            features=hidden_features,
            use_bias=True,
            kernel_init=self.recurrent_kernel_init,
            bias_init=self.bias_init,
        )
        dense_h = partial(dense, use_bias=False)
        dense_i = partial(dense, use_bias=True)
        new_h = mobius_gate_fn(
            regularize(mobius_add(dense_i(name="ih")(inputs), dense_h(name="hh")(h)))
        )
        return new_h, new_h

    @staticmethod
    def initialize_carry(
        rng: PRNGKey, batch_dims: Tuple[int, ...], size: int, init_fn: Array = zeros
    ) -> Array:
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


class PoincareGRUCell(nn.Module):
    """Poincare GRU cell.

    Attributes:
        curv: curvature of the poincare manifold (default: -1).
        eps: eps to be added inputs, h while doing operations, done
            to increase numerical stability of operations (default: 1E-15).
        gate_fn: activation function used for gates (default: sigmoid).
        activation_fn: activation function used for output and memory update
            (default: tanh).
        kernel_init: initializer function for the kernels that transform
            the input (default: lecun_normal).
        recurrent_kernel_init: initializer function for the kernels that transform
            the hidden state (default: orthogonal).
        bias_init: initializer for the bias parameters (default: zeros).
    """

    curv: float = -1.0
    in_radii: float = 1 - 1e-8
    out_radii: float = 1e-15
    gate_fn: Callable[..., Any] = sigmoid
    activation_fn: Callable[..., Any] = tanh
    kernel_init: Callable = default_kernel_init
    recurrent_kernel_init: Callable = orthogonal()
    bias_init: Callable = zeros
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, carry: Array, inputs: Array) -> Tuple[Array, Array]:
        """Poincare Gated recurrent unit (GRU) cell.
        Args:
            carry: the hidden state of the LSTM cell,
                initialized using `GRUCell.initialize_carry`.
            inputs: an ndarray with the input for the current time step.
                All dimensions except the final are considered batch dimensions.

        Returns:
            A tuple with the new carry and the output.
        """
        h = carry + self.eps
        inputs = inputs + self.eps
        hidden_features = h.shape[-1]
        manifold = PoincareBall(hidden_features, self.curv)
        mobius_add = vmap(manifold.mobius_add, in_axes=(0, 0))
        mobius_pw_prod = vmap(manifold.mobius_pw_prod, in_axes=(0, 0))
        mobius_gate_fn = vmap(manifold.mobius_f(self.gate_fn))
        mobius_activation_fn = vmap(manifold.mobius_f(self.activation_fn))
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
        r = mobius_gate_fn(
            mobius_add(dense_i(name="ir")(inputs), dense_h(name="hr")(h))
        )
        z = mobius_gate_fn(
            mobius_add(dense_i(name="iz")(inputs), dense_h(name="hz")(h))
        )
        n = mobius_activation_fn(
            mobius_add(
                dense_i(name="in")(inputs), mobius_pw_prod(r, dense_h(name="hn")(h))
            )
        )
        new_h = mobius_add(h, mobius_pw_prod(z, mobius_add(-1.0 * h, n)))
        return new_h, new_h

    @staticmethod
    def initialize_carry(
        rng: PRNGKey, batch_dims: Tuple[int, ...], size: int, init_fn: Array = zeros
    ) -> Array:
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


class LiftedPoincareGRUCell(nn.Module):
    """A minimalist poincare GRU cell which is ready to use."""

    @partial(
        nn.transforms.scan,
        variable_broadcast="params",
        in_axes=1,
        out_axes=1,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry: Array, inputs: Array) -> Tuple[Array, Array]:
        return PoincareGRUCell()(carry, inputs)

    @staticmethod
    def initialize_carry(batch_dims: Tuple[int, ...], size: int) -> Array:
        return PoincareGRUCell.initialize_carry(random.PRNGKey(0), batch_dims, size)
