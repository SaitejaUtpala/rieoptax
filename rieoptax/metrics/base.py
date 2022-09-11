from abc import ABC, abstractmethod


class RiemannianMetric(ABC):

    def __init__(self, k):
        self.k = k 
    
    def tangent_gaussian_sample(self, mean, sigma=1):



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

