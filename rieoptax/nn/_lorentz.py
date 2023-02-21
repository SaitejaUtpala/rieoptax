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

from rieoptax.geometry.hyperbolic import LorentzHyperboloid


class LorentzMultiHeadDotProductAttention(nn.Module):
    """Lorentz MultiHeadDotProductAttention by default
    employes

    Attributes:
        num_heads: number of attention heads. Features (i.e., inputs_q.shape[-1])
            should be divisible by the number of heads.
        dtype: the dtype of the computation.
            (default: infer from inputs and params)
        param_dtype: the dtype passed to parameter initializers (default: float32)
        qkv_features: dimension of the key, value, and value.
        out_featues: dimension of the last projection.
        broadcast_dropout: bool: use a broadcasted dropout along batch dims.
        dropout_rate: dropout rate.
        deterministic: if false, the attention weight is masked randomly
            using dropout, whereas if true, the attention weights
            are deterministic.
        precision: numerical precision of the computation see `jax.lax.precision`
            for the details.
        kernel_init: initializer for the kernel of the Dense layer.
        bias_init: initializer for the bias of the Dense layer.
    """

    num_heads: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    broadcast_dropout: bool = True
    dropout_rate: float = 0.0
    deterministic: Optional[bool] = None
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
    use_bias: bool = True
    attention_fn: Callable[..., Array] = dot_product_attention
    decode: bool = False

    @nn.compact
    def __call__(
        self,
        inputs_q: Array,
        inputs_kv: Array,
        mask: Optional[Array] = None,
        deterministic: Optional[bool] = None,
    ):
        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        assert (
            qkv_features % self.num_heads == 0
        ), "Memory dimension must be divisible by number of heads."
        head_dim = qkv_features // self.num_heads






class LorentzSelfAttention(LorentzMultiHeadDotProductAttention):
  """Self-attention special case of multi-head dot-product attention."""

  @nn.compact
  def __call__(self, inputs_q: Array, mask: Optional[Array] = None, # type: ignore
               deterministic: Optional[bool] = None):
    return super().__call__(inputs_q, inputs_q, mask,
                            deterministic=deterministic)

class LorentzMultiHeadDotProductAttention(Module):
  
  num_heads: int
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.
  deterministic: Optional[bool] = None
  precision: PrecisionLike = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  use_bias: bool = True
  attention_fn: Callable[..., Array] = dot_product_attention
  decode: bool = False

  @compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
               mask: Optional[Array] = None,
               deterministic: Optional[bool] = None):
    """Applies multi-head dot product attention on the input data.
    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.
    Args:
      inputs_q: input queries of shape
        `[batch_sizes..., length, features]`.
      inputs_kv: key/values of shape
        `[batch_sizes..., length, features]`.
      mask: attention mask of shape
        `[batch_sizes..., num_heads, query_length, key/value_length]`.
        Attention weights are masked out if their corresponding mask value
        is `False`.
      deterministic: if false, the attention weight is masked randomly
        using dropout, whereas if true, the attention weights
        are deterministic.
    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // self.num_heads

    dense = functools.partial(DenseGeneral,
                              axis=-1,
                              dtype=self.dtype,
                              param_dtype=self.param_dtype,
                              features=(self.num_heads, head_dim),
                              kernel_init=self.kernel_init,
                              bias_init=self.bias_init,
                              use_bias=self.use_bias,
                              precision=self.precision)
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    query, key, value = (dense(name='query')(inputs_q),
                         dense(name='key')(inputs_kv),
                         dense(name='value')(inputs_kv))

    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    if self.decode:
      # detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      cached_key = self.variable('cache', 'cached_key',
                                 jnp.zeros, key.shape, key.dtype)
      cached_value = self.variable('cache', 'cached_value',
                                   jnp.zeros, value.shape, value.dtype)
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.int32))
      if is_initialized:
        *batch_dims, max_length, num_heads, depth_per_head = (
            cached_key.value.shape)
        # shape check of cached keys against query input
        expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
        if expected_shape != query.shape:
          raise ValueError('Autoregressive cache shape error, '
                           'expected query shape %s instead got %s.' %
                           (expected_shape, query.shape))
        # update key, value caches with our new 1d spatial slices
        cur_index = cache_index.value
        indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
        key = lax.dynamic_update_slice(cached_key.value, key, indices)
        value = lax.dynamic_update_slice(cached_value.value, value, indices)
        cached_key.value = key
        cached_value.value = value
        cache_index.value = cache_index.value + 1
      
        mask = combine_masks(
            mask,
            jnp.broadcast_to(jnp.arange(max_length) <= cur_index,
                             tuple(batch_dims) + (1, 1, max_length)))

    dropout_rng = None
    if self.dropout_rate > 0.:  # Require `deterministic` only if using dropout.
      m_deterministic = merge_param('deterministic', self.deterministic,
                                    deterministic)
      if not m_deterministic:
        dropout_rng = self.make_rng('dropout')
    else:
      m_deterministic = True

    # apply attention
    x = self.attention_fn(
        query,
        key,
        value,
        mask=mask,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=m_deterministic,
        dtype=self.dtype,
        precision=self.precision)  # pytype: disable=wrong-keyword-args
    out = DenseGeneral(features=features,
                       axis=(-2, -1),
                       kernel_init=self.kernel_init,
                       bias_init=self.bias_init,
                       use_bias=self.use_bias,
                       dtype=self.dtype,
                       param_dtype=self.param_dtype,
                       precision=self.precision,
                       name='out')(x)
    return out