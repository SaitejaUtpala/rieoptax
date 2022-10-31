from absl.testing import absltest
from chex import assert_trees_all_close
from jax import numpy as jnp

from rieoptax.geometry.hyperbolic import PoincareBall


class TestPoincareBall(absltest.TestCase):
    manifold = PoincareBall(2)
    sqrt2 = jnp.sqrt(2)
    bpt = jnp.array([0.1 / sqrt2, 0.1 / sqrt2])
    tv = jnp.array([1.0, 2.0])
    pt = jnp.array([0.2 / sqrt2, 0.2 / sqrt2])

    def test_mobius_add(self):
        exptd = jnp.array([0.2079726, 0.2079726])
        assert_trees_all_close(self.manifold.mobius_add(self.bpt, self.pt), exptd)

    def test_dist(self):
        exptd = jnp.array(0.20479442)
        assert_trees_all_close(self.manifold.dist(self.bpt, self.pt), exptd)

    def test_ptrans(self):
        exptd = jnp.array([0.96969694, 1.9393936])
        assert_trees_all_close(self.manifold.ptrans(self.bpt, self.pt, self.tv), exptd)

    def test_sdist_to_gyroplanes(self):
        exptd = jnp.array(0.19441889)
        assert_trees_all_close(
            self.manifold.sdist_to_gyroplanes(self.bpt, self.tv, self.pt), exptd
        )

    def test_mobius_matvec(self):
        exptd = jnp.array([0.19429359, 0.45335174])
        M = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        assert_trees_all_close(self.manifold.mobius_matvec(M, self.bpt), exptd)

    def test_regularize(self):
        zero = jnp.array([0.0, 0.0])
        one = jnp.array([1.0, 0.0])
        exptd = jnp.array(True)

        assert_trees_all_close(
            jnp.linalg.norm(self.manifold.regularize(zero)) != 0.0, exptd
        )
        assert_trees_all_close(
            jnp.linalg.norm(self.manifold.regularize(one)) != 1.0, exptd
        )
