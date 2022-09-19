

from abc import ABC, abstractmethod
from functools import partial

from jax import jit, vmap

class RiemannianManifold(ABC):
    def __init__(self):
        j = partial(jit, static_argnums=(0,))
        v = partial(vmap, in_axes=(0, None))
        self.exp = jit(v(self.exp))
        self.log = jit(v(self.log))
        self.parallel_transport = jit(v())
    

class PoincareBall(Hyperbolic):
    
    def __init__(self,dim, curvature):
        self.dim = dim 
        self.curvature = curvature
        
    def mobius_addition(self, point_a, point_b):
        inp  = jnp.dot(point_a, point_b)
        b_norm = jnp.norm(point_b)**2
        a_norm = jnp.norm(point_a)**2
        
        numerator = (1-2*self.curvature*inp - self.curvature*b_norm)*point_a +  (1 + self.curvature * a_norm)*point_b
        denominator = (1 - 2*self.curvature + self.curvature**2 * b_norm a_norm)
        ma = numerator/denominator
        return ma 
        
    def mobius_subtraction(self, point_a, point_b):
        ms = self.mobius_addition(point_a, -1*point_b)
        return ms 
    
    def gyration_operator(self, point_a, point_b, vec):
        gb = self.mobius_addition(point_b, vec)
        ggb = self.mobius_addition(point_b, gb)
        gab = self.mobius_addition(point_a, point_b)
        return -1 * self.mobius_addition(gab, ggb)
    
    def conformal_factor(self, point):
        cp_norm = self.curvature * jnp.norm(point)**2
        cf = 2/(1+ cp_norm)
        return cf 
        
    def exp(self, tangent_vec, base_point):
        t = jnp.sqrt(jnp.abs(self.curvature)) * jnp.norm(tangent_vec)
        point = (jnp.tanh(t/2 * self.conformal_factor(base_point))/t) * t
        exp = self.mobius_addition(base_point, point)
        return exp 
        
    def log(self, base_point, point):
        ma = self.mobius_addition(-1*base_point, point)
        abs_sqrt_curv = jnp.sqrt(jnp.abs(self.curvature))
        norm_ma = jnp.norm(ma)
        mul = (2/(abs_sqrt_curv * self.conformal_factor(base_point))) *  jnp.arctanh(abs_sqrt_curv * norm_ma)
        log = mul * (ma/norm_ma)
        return log 
                 
    def metric(self, base_point, tangent_vec_a, tangent_vec_b) : 
        metric  = self.conformal_factor(base_point) * jnp.inp(tangent_vec_a, tangent_vec_b)
        return metric 
    
    def parallel_transport(self, start_point, end_point, tangent_vec):
        self.conformal_factor(start_point)
        self.conformal_facotr(end_point)
        pt = self.gyration_operator(end_point, -1*start_point, tangent_vec)
        return pt 