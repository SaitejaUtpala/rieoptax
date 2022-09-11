import jax
from base import RiemannianMetric


@jax.tree_util.register_pytree_node_class
class SPDMetric(RiemannianMetric):
    pass 




@jax.tree_util.register_pytree_node_class
class AffineInvariant(SPDMetric):

    def __init__(self, k):
        self.k = k 

    def _norm(self, tangent_vec, base_point):
        
    def _dist(self, point_a, point_b):

    def _exp(self, tangent_vec, base_point):
        pass 

    def _log(self, point, base_point):
        pass 

    def _retraction(self, tangent_vec, base_point):
        pass 

    def _parallel_transport(self, tangent_vec, start_point, end_point):
        pass 

    def _vector_transport(self, tangent_vec, start_point, end_point):
        pass



@jax.tree_util.register_pytree_node_class
class LogEuclidean(SPDMetric):

    def __init__(self, k):
        self.k = k 

    def _exp(self, tangent_vec, base_point):
        pass 

    def _log(self, point, base_point):
        pass 

    def _retraction(self, tangent_vec, base_point):
        pass 

    def _parallel_transport(self, tangent_vec, start_point, end_point):
        pass 

    def _vector_transport(self, tangent_vec, start_poitn, end_point):
        pass 

@jax.tree_util.register_pytree_node_class
class LogCholesky(SPDMetric):
    pass

@jax.tree_util.register_pytree_node_class
class BuresWasserstein(SPDMetric):
    pass

@jax.tree_util.register_pytree_node_class
class GeneralizedBuresWasserstein(SPDMetric):
    pass 

@jax.tree_util.register_pytree_node_class
class Euclidean(SPDMetric):

    def __init__(self, k):
        self.k = k 

    def _exp(self, tangent_vec, base_point):
        pass 

    def _log(self, point, base_point):
        pass 

    def _retraction(self, tangent_vec, base_point):
        pass 

    def _parallel_transport(self, tangent_vec, start_point, end_point):
        pass 

    def _vector_transport(self, tangent_vec, start_poitn, end_point):
        pass
