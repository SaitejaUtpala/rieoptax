from base import RiemannianManifold
from jax import numpy as jnp


class SPDManifold(RiemannianManifold):  
    def sqrt_neg_sqrt(self, spd_matrix):
        eigval, eigvec = jnp.linalg.eigh(spd_matrix[None])
        pow_eigval = jnp.stack([jnp.power(eigval, 0.5), jnp.power(eigval, -0.5)])
        result = (pow_eigval * eigvec) @  eigvec.swapaxes(1,2)
        return result
    
class SPDAffineInvariant(SPDManifold):
    def exp(self, base_point, tangent_vec):
        powers = self.sqrt_neg_sqrt(base_point)
        eigval, eigvec = jnp.linalg.eigh(powers[1] @ tangent_vec @ powers[1])
        middle_exp = (jnp.exp(eigval).reshape(1,-1) * eigvec)@ eigvec.T
        exp = powers[0] @ middle_exp @ powers[0]
        return exp
    
    def log(self, base_point, point ):
        powers = self.sqrt_neg_sqrt(base_point)
        eigval, eigvec = jnp.linalg.eigh(powers[1] @ point @ powers[1])
        middle_log = (jnp.log(eigval).reshape(1,-1) * eigvec)@ eigvec.T
        exp = powers[0] @ middle_log @ powers[0]
        return exp

    def dist(self, point_a, point_b):
        eigval =  jnp.linalg.eigvals(jnp.linalg.inv(point_b) @ point_a)
        dist = jnp.linalg.norm(jnp.log(eigval))    
        return dist
    
    def parallel_transport(self, base_point, end_point, tangent_vec ):
        base_point

    def egrad_to_rgrad(self, egrad, base_point):
        return base_point @ egrad @ base_point.T

# class SPDBuresWasserstein(SPDManifold):
#     def exp(self, base_point, tangent_vec):
#        return 
    
#     def log(self, base_point, point ):
#         return 

#     def dist(self, point_a, point_b):
#         return 
    
#     def parallel_transport(self, base_point, end_point, tangent_vec ):
#         return 

#     def egrad_to_rgrad(self, egrad, base_point):
#         return base_point @ egrad @ base_point.T


# class SPDGeneralizedBuresWasserstein(SPDManifold):
#     def exp(self, base_point, tangent_vec):
#        return 
    
#     def log(self, base_point, point ):
#         return 

#     def dist(self, point_a, point_b):
#         return 
    
#     def parallel_transport(self, base_point, end_point, tangent_vec ):
#         return 

#     @staticmethod 
#     def egrad_to_rgrad(self, egrad, base_point):
#         return base_point @ egrad @ base_point.T