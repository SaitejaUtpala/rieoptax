from abc import ABC, abstractmethod
from functools import partial

from jax import jit, vmap


class RiemannianMetric(ABC):
    def __init__(self):
        j = partial(jit, static_argnums=(0,))
        v = partial(vmap, in_axes=(0, None))
        self.exp = jit(v(self.exp))

