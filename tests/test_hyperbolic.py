
from absl.testing import absltest
from chex import assert_trees_all_close
from jax import numpy as jnp

from rieoptax.geometry.hyperbolic import PoincareBall


class TestPoincareBall(absltest.TestCase):
    manifold = PoincareBall(2)
    sqrt2 = jnp.sqrt(2)
    bpt = jnp.array([0.1/sqrt2, 0.1/sqrt2])
    tv = jnp.array([1.0, 2.0])
    pt = jnp.array([0.2/sqrt2, 0.2/sqrt2])
    
    def test_mobius_matvec(self, M, pt):
        pass 

    def test_mobius_add(self):
        exptd = jnp.array([[0.07215376, 0.07215376]])
        assert_trees_all_close(self.manifold.mobius_add(self.bpt,self.pt), exptd)


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

    def test_sdist_to_gyroplanes(self):
        exptd = jnp.array(0.19441889)
        assert_trees_all_close(self.manifold.sdist_to_gyroplanes(self.bpt, self.tv, self.pt), exptd)




   