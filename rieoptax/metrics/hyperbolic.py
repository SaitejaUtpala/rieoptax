class Hyperbolic(RiemannianManifold):  
    

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
        point = jnp.tanh(t * self.conformal_factor(base_point))
        exp = self.mobius_addition(base_point, )
        
    def log(self, base_point, point):
        ma = self.mobius_addition(-1*base_point, point)
        abs_sqrt_curv = jnp.sqrt(jnp.abs(self.curvature))
        norm = jnp.norm(ma)
        log =2/(abs_sqrt_curv * base_point) jnp.arctanh(abs_sqrt_curv)
       
        mul = 2/(abs_sqrt_curv * self.conformal_factor()
        jnp.arctanh(jnp.sqrt())
    def parallel_transport(self, start_point, end_point, tangent_vec):
        self.conformal_factor(start_point)
        self.conformal_facotr(end_point)
        pt = self.gyration_operator(end_point, -1*start_point, tangent_vec)