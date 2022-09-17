from abc import ABC, abstractmethod
from functools import partial

v2 = partial(vmap, in_axes=(0, 0))
v3 = partial(vmap, in_axes=(0, 0, 0))
from jax import jit, vmap


class RiemannianMetric(ABC):

    def __init__(self, k):
        self.k = k 
        self.norm = jit(v2(self._norm))
        self.dist = jit(v2(self._dist))
        self.exp  = jit(v2(self._exp))
        self.log  = jit(v2(self._log))  
   
    # def norm(self, tangent_vec, base_point):
    #     return 
        
    # def dist(self, point_a, point_b):

    # def exp(self, tangent_vec, base_point):
    #     pass 

    # def log(self, point, base_point):
    #     pass 

    # def retraction(self, tangent_vec, base_point):
    #     pass 

    # def parallel_transport(self, tangent_vec, start_point, end_point):
    #     pass 

    # def vector_transport(self, tangent_vec, start_point, end_point):
    #     pass

