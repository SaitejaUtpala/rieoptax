from base import RiemannianMetric
from jax import numpy as jnp


class SPDMetric(RiemannianMetric):  
    def sqrt_neg_sqrt(self, spd_matrix):
        eigval, eigvec = jnp.linalg.eigh(spd_matrix[None])
        pow_eigval = jnp.stack([jnp.power(eigval, 0.5), jnp.power(eigval, -0.5)])
        result = (pow_eigval * eigvec) @  eigvec.swapaxes(1,2)
        return result
    

class SPDAffineInvariant(RiemannianMetric):
    def exp(self, tangent_vec, base_point):
        powers = self.sqrt_neg_sqrt(base_point)
        eigval, eigvec = jnp.linalg.eigh(powers[1] @ tangent_vec @ powers[1])
        middle_exp = (jnp.exp(eigval).reshape(1,-1) * eigvec)@ eigvec.T
        exp = powers[0] @ middle_exp @ powers[0]
        return exp
    
    def log(self, point, base_point ):
        powers = self.sqrt_neg_sqrt(base_point)
        eigval, eigvec = jnp.linalg.eigh(powers[1] @ point @ powers[1])
        middle_log = (jnp.log(eigval).reshape(1,-1) * eigvec)@ eigvec.T
        exp = powers[0] @ middle_log @ powers[0]
        return exp

    def dist(point_a, point_b):
        eigval =  jnp.linalg.eigvals(jnp.linalg.inv(point_b) @ point_a)
        dist = jnp.linalg.norm(jnp.log(eigval))    
        return dist







# @jax.tree_util.register_pytree_node_class
# class AffineInvariant(SPDMetric):

#     def __init__(self, k):
#         self.k = k 

#     def _norm(self, tangent_vec, base_point):
        
#     def _dist(self, point_a, point_b):

#     def _exp(self, tangent_vec, base_point):
#         powers = sqrt_neg_sqrt(base_point)
#         eigval, eigvec = jnp.linalg.eigh(powers[1] @ tangent_vec @ powers[1])
#         middle_exp = (jnp.exp(eigval).reshape(1,-1) * eigvec)@ eigvec.T
#         exp = powers[0] @ middle_exp @ powers[0]
#         return exp 
        
#     def exp(tangent_vec, base_point):
    

#     def _log(self, point, base_point):
#         pass 

#     def _retraction(self, tangent_vec, base_point):
#         pass 

#     def _parallel_transport(self, tangent_vec, start_point, end_point):
#         pass 

#     def _vector_transport(self, tangent_vec, start_point, end_point):
#         pass



# @jax.tree_util.register_pytree_node_class
# class LogEuclidean(SPDMetric):

#     def __init__(self, k):
#         self.k = k 

#     def _exp(self, tangent_vec, base_point):
#         pass 

#     def _log(self, point, base_point):
#         pass 

#     def _retraction(self, tangent_vec, base_point):
#         pass 

#     def _parallel_transport(self, tangent_vec, start_point, end_point):
#         pass 

#     def _vector_transport(self, tangent_vec, start_poitn, end_point):
#         pass 

# @jax.tree_util.register_pytree_node_class
# class LogCholesky(SPDMetric):
#     pass

# @jax.tree_util.register_pytree_node_class
# class BuresWasserstein(SPDMetric):
#     pass

# @jax.tree_util.register_pytree_node_class
# class GeneralizedBuresWasserstein(SPDMetric):
#     pass 

# @jax.tree_util.register_pytree_node_class
# class Euclidean(SPDMetric):

#     def __init__(self, k):
#         self.k = k 

#     def _exp(self, tangent_vec, base_point):
#         pass 

#     def _log(self, point, base_point):
#         pass 

#     def _retraction(self, tangent_vec, base_point):
#         pass 

#     def _parallel_transport(self, tangent_vec, start_point, end_point):
#         pass 

#     def _vector_transport(self, tangent_vec, start_poitn, end_point):
#         pass
