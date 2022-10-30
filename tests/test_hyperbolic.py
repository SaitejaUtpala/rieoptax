
from absl.testing import absltest
from chex import assert_trees_all_close
from jax import numpy as jnp

from rieoptax.geometry.hyperbolic import PoincareBall


class TestPoincareBall(absltest.TestCase):
    manifold = PoincareBall(2)
    
    def test_mobius_matvec(self, M, pt):

    def test_mobius_add(self, pt_a, pt_b ):

    def test_exp(self, ):
        assert_trees_all_close(self.manifold.exp(bp, tv), expected)

    def test_log(self, ):
        assert_trees_all_close(self.manifold.log(bp, p), exp)

    def test_inp(self, ):
        assert_trees_all_close(self.manifold.inp(bp, tv_a, tv_b), exp)

    def test_dist(self, ):
        ssert_trees_all_close(self.manifold.dist(p_a, p_b), exp)

    def test_pt(self, ):
        assert_trees_all_close(self.manifold.pt(p_a, p_b, tv), exp)



   