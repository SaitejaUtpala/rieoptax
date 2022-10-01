from base import RiemannianManifold
from jax import numpy as jnp


class SPDManifold(RiemannianManifold):  
    def sqrt_neg_sqrt(self, spd_matrix):
        eigval, eigvec = jnp.linalg.eigh(spd_matrix[None])
        pow_eigval = jnp.stack([jnp.power(eigval, 0.5), jnp.power(eigval, -0.5)])
        result = (pow_eigval * eigvec) @  eigvec.swapaxes(1,2)
        return result

    def inv_vecd(vector):
        dim = vector.shape[0]
        k = (int)((jnp.sqrt(8*dim +1) - 1)/2)
        diag_indices = jnp.diag_indices(k)
        triu_indices = jnp.triu_indices(k,1)
        spd = jnp.zeros((k,k))
        spd[diag_indices] = vector[:k]/2
        spd[triu_indices] = vector[k:]/jnp.sqrt(2)
        spd = spd + spd.T
        return spd
    
class SPDAffineInvariant(RiemannianManifold):
    
    def __init__(self, m):
        self.m = m
        super().__init__()
        
    def sqrt_neg_sqrt(self, spd_matrix):
        eigval, eigvec = jnp.linalg.eigh(spd_matrix[None])
        sqrt_eval = jnp.sqrt(eigval)
        neg_sqrt_eval = 1/sqrt_eval
        pow_eigval = jnp.stack([sqrt_eval, neg_sqrt_eval])
        result = (pow_eigval * eigvec) @  eigvec.swapaxes(1,2)
        return result
        
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
        log = powers[0] @ middle_log @ powers[0]
        return log

    def dist(self, point_a, point_b):
        eigval =  jnp.linalg.eigvals(jnp.linalg.inv(point_b) @ point_a)
        dist = jnp.linalg.norm(jnp.log(eigval))    
        return dist

    def egrad_to_rgrad(self, egrad, base_point):
        return base_point @ egrad @ base_point.T
   

