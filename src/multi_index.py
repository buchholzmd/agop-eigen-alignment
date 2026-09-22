import jax.numpy as jnp
import jax.random as jr

from jax import vmap
from linalg import stiefel

def poly(coeffs, orders, z):
    assert coeffs.shape == orders.shape
    if z.shape[0] > orders.shape[0]:
        coeffs = jnp.pad(coeffs, (0, z.shape[0] - orders.shape[0]))
        orders = jnp.pad(orders, (0, z.shape[0] - orders.shape[0]))
    return coeffs @ (z ** orders)

def generate_multi_index_data(g, r, X, rng, noise=0.1, normalize='none'):
    U = stiefel(X.shape[-1], r, rng)

    if normalize == 'rms':
        f = lambda x: g(U @ x)
        scale = jnp.sqrt(jnp.mean(vmap(f)(X)) ** 2)
    elif normalize == 'none':
        scale = 1.0
    f = lambda x: g(U @ x) / scale

    y = vmap(f)(X) + noise * jr.normal(rng.next(), (X.shape[0],))

    return f, U, y