
class Stiefel(RiemannianManifold):
    def __init__(self, m, r):
         if m < r or r < 1:
            raise ValueError(
                f"Need n >= p >= 1. Values supplied were n = {n} and p = {p}"
            )


class StiefelEuclideanMetric(Stiefel):
    
    def inner_product(self, base_point, tangent_vec_a, tangent_vec_b):
        

    def exp(self, base_point, tangent_vec):
        pass 

        
class StiefelCanonicalMetric(Stiefel):


    def inner_product(self, base_point, tangent_vec_a, tangent_vec_b):

    def exp(self, base_point, tangent_vec):
        pass 


