import jax

import jax.numpy as jnp

def agop(f, X):
    """AGOP of ANY scalar f: R^d -> R on X: (n, d).  (1/n) sum_i grad f(x_i) grad f(x_i)^T.

    Deliberately NOT jitted: `f` would have to be a static arg and would hash by
    identity, recompiling on every call. jit at the call site instead.
    """
    J = jax.vmap(jax.grad(f))(X)
    return J.T @ J / X.shape[0]

@jax.jit
def nfm(W):
    """Neural Feature Matrix W^T W for first-layer weights W: (m, d)."""
    return W.T @ W