
from absl.testing import absltest
import chex
from rieoptax.geometry.spd import SPDAffineInvariant
class TestSPDAffineInvarinat(absltest.TestCase):
    
    manifold = SPDAffineInvariant(2,2)
    def test_exp():
        bp = jnp.array()
        tv = jnp.array()
        output = manifold.exp(bp, tv)
        expected = 
        chex.assert_trees_all_close(outptu, expected)
            
    def test_log():
        
    def test_pt():
        
    def test_inp():
        
    def test_
    