from dataclasses import field
from functools import partial
from typing import Any, Callable, Optional, Tuple, Union

from chex import Array
from flax import linen as nn
from flax.linen.activation import sigmoid, tanh
from flax.linen.dtypes import promote_dtype
from flax.linen.initializers import orthogonal
from flax.linen.linear import default_kernel_init
from jax import numpy as jnp
from jax import random, vmap
from chex import Array 
from jax import lax

from rieoptax.geometry.hyperbolic import PoincareBall

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str],
                      Tuple[lax.Precision, lax.Precision]]

class PoincareEmbed(nn.Module):
    """Embedding Module.
    A parameterized function from integers [0, n) to d-dimensional vectors.
    Attributes:
        num_embeddings: number of embeddings.
        features: number of feature dimensions for each embedding.
        dtype: the dtype of the embedding vectors (default: same as embedding).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        embedding_init: embedding initializer.
    """
    num_embeddings: int
    features: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    embedding_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_embed_init
    embedding: Array = field(init=False)

    def setup(self):
        self.embedding = self.param('embedding',
                                    self.embedding_init,
                                    (self.num_embeddings, self.features),
                                    self.param_dtype)

    def __call__(self, inputs: Array) -> Array: 
        """Embeds the inputs along the last dimension.
        Args:
            inputs: input data, all dimensions are considered batch dimensions.
        Returns:
            Output which is embedded input data.  The output shape follows the input,
            with an additional `features` dimension appended.
        """

        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError('Input type must be an integer or unsigned integer.')
        embedding, = promote_dtype(self.embedding, dtype=self.dtype, inexact=False)
        return jnp.take(embedding, inputs, axis=0)