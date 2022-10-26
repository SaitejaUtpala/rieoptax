from rieoptax.geometry.base import RiemannianManifold


class CouplingManifold(RiemannianManifold):
    def __init__(self, p, q):
        self.p = p 
        self.q = q 
        self.n = p.shape[0]
        self.m = q.shape[0]
        self.dim = (self.n-1) * (self.m-1)


class CouplingFisher(CouplingManifold):
    
    def inp(self, bp, tv_a, tv_b):
        pass 

    def retr(self, bp, tv):
        pass 


class Multinomial(CouplingManifold):
    def __init__(self, p):
        self.p = p 

class DSMultinomial(CouplingManifold):
    def __init__(self, p):
        self.p = p
