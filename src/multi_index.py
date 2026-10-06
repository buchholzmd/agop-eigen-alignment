import jax.numpy as jnp
import jax.random as jr

from jax import vmap
from linalg import stiefel

def generate_multi_index_data(g, r, X, rng, noise=0.1, normalize='none'):
    U = stiefel(X.shape[-1], r, rng)

    f_raw = lambda x: g(U @ x)

    if normalize == 'rms':
        outputs = vmap(f_raw)(X)
        scale = jnp.sqrt(jnp.mean(outputs ** 2))
    elif normalize == 'none':
        scale = 1.0
    else:
        raise ValueError(f"Unknown normalization: {normalize}")

    f = lambda x: f_raw(x) / scale

    y = vmap(f)(X) + noise * jr.normal(rng.next(), (X.shape[0],))

    return f, U, y