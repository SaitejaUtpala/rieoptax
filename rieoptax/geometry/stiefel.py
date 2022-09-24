from base import RiemannianManifold


class StiefelManifold(RiemannianManifold):
    def __init__(self, m, r):
         if m < r or r < 1:
            raise ValueError(
                f"Need m >= r >= 1. Values supplied were m = {m} and r = {r}"
            )
        self.m = m 
        self.r = r


class StiefelEuclideanMetric(StiefelManifold):
    
    def inner_product(self, base_point, tangent_vec_a, tangent_vec_b):
        return jnp.tensordot(
            tangent_vector_a, tangent_vector_b, axes=2
        )

    def exp(self, base_point, tangent_vec):
        A = base_point.T @ tangent_vec 
        B = tangent_vec @ base_point.T
        exp = jnp.linalg.expm(B - B.T) @ base_point @ jnp.linalg.expm(-A)
        return exp

        
class StiefelCanonicalMetric(StiefelManifold):


    def inner_product(self, base_point, tangent_vec_a, tangent_vec_b):

    def exp(self, base_point, tangent_vec):
        pass 

    def tangent_gaussian(self, sigma):
        pass 


