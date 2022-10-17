from abc import ABC, abstractmethod
from functools import partial

from jax import jit


class RiemannianManifold(ABC):
    def __init__(self):
        self.exp = jit(self.exp)
        self.log = jit(self.log)
