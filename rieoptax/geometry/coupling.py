from chex import Array

from rieoptax.geometry.base import RiemannianManifold


class CouplingManifold(RiemannianManifold):
    def __init__(self, p, q):
        self.p = p 
        self.q = q 
        self.n = p.shape[0]
        self.m = q.shape[0]
        self.dim = (self.n-1) * (self.m-1)

    def extended_sinkhorn_knopp():
        pass 


class CouplingFisher(CouplingManifold):
    
    def inp(self, bpt : Array, tv_a : Array, tv_b : Array) -> float:
        mat = (tv_a * tv_b)/bpt
        return mat.sum()

    def ts_proj(self, bpt : Array, vec: Array) -> Array :
        """Orthogonal projection on tangent space.

        Args:
            bpt: base_point.
            pt: tangent_vec.

        Returns:
            returns orthogonal projection of 'vec' onto T_{bpt}M .
        """
        pass  

        
        
    def retr(self, bp, tv):
        pass 


class Multinomial(CouplingManifold):
    def __init__(self, p):
        self.p = p 

class SPDMultinomial(CouplingManifold):
    def __init__(self, p):
        self.p = p

class SYMMultinomial(CouplingManifold):
    def __init__(self, p):
        self.p = p

class DSMultinomial(CouplingManifold):
    def __init__(self, p):
        self.p = p


