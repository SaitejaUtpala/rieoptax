from jax import grad


def rgrad(f):
    rgrad = f.egrad_to_rgrad(grad(f))
    

def geometry(manifold):
    def wrapper(f):
        f.egrad_to_rgrad = manifold.egrad_to_rgrad
        return f
    return wrapper
