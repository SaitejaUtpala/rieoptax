from jax import grad


def rgrad(f):
    rgrad = f.egrad_to_rgrad(grad(f))
    

