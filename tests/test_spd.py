
from absl.testing import absltest
from chex import assert_trees_all_close
from rieoptax.geometry.spd import SPDAffineInvariant

class TestSPDAffineInvarinat(absltest.TestCase):
    
    manifold = SPDAffineInvariant(2)
    def test_exp():
        manifold = SPDAffineInvariant()
        tv =jnp.array([[2.0, 0.0], [0.0, 2.0]])
        bp =jnp.array([[1.0, 0.0], [0.0, 1.0]])
        expected=jnp.array([[jnp.exp(2), 0.0], [0.0, jnp.exp(2)]])
        assert_trees_all_close(manifold.exp(bp, tv), expected)

    def test_log():
        p=jnp.array([[1.0, 0.0], [0.0, 1.0]])
        bp=jnp.array([[2.0, 0.0], [0.0, 2.0]])
        exp=jnp.array([[-2 * jnp.log(2), 0.0], [0.0, -2 * jnp.log(2)]])
        assert_trees_all_close(manifold.log(bp, p), exp)
    
    def test_inp():
        tv_a =jnp.array([[2.0, 0.0], [0.0, 2.0]])
        tv_b =jnp.array([[4.0, 0.0], [0.0, 4.0]])
        bp =jnp.array([[2.0, 0.0], [0.0, 2.0]])
        exp = jnp.array([4])
        assert_trees_all_close(manifold.inp(bp, tv_a, tv_b), exp)

    def test_dist():
        p_a =jnp.array([[1.0, 0.0], [0.0, 1.0]])
        p_b = jnp.array([[1.0, 0], [0.0, 2.0]])
        exp = jnp.array([4])
        assert_trees_all_close(manifold.dist(p_a, p_b), exp)

    def test_pt():
        p_a =jnp.array([[1.0, 0.0], [0.0, 1.0]])
        p_b = jnp.array([[2.0, 0], [0.0, 2.0]])
        tv = jnp.array([[0.0, 5], [5, 0.0]])
        exp = jnp.array([[0, 10], [10, 0]])
        assert_trees_all_close(manifold.pt(p_a, p_b, tv), exp)


    def test_egrad_to_rgrad():
        egrad = jnp.array([[2.0, 3.0], [3.0, 4.0]])
        bp = jnp.array([[2.0, 0.0], [0.0, 4.0]])
        exp = jnp.array([[8, 24], [24, 64]])
        assert_trees_all_close(manifold.egrad_to_rgrad(egrad, bp), exp)                
        
    