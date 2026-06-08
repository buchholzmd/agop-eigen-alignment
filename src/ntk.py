import jax

import jax.numpy as jnp

from functools import partial
from linalg import mvp_power_iteration

try:
    import neural_tangents as nt

    def entk(model, params, X):
        kernel_fn = nt.empirical_ntk_fn(
            lambda p, x: jnp.atleast_1d(model.apply_fn(p, x)),
            trace_axes=(),
            vmap_axes=0
        )

        return kernel_fn(X, X, params) / X.shape[0]
except ImportError:
    print("Neural Tangents library not found. Using only custom NTK implementation.") 

@partial(jax.jit, static_argnums=(1,))
def nvp(params, model, X, v):
    # NTK vector products
    _, vjp_fn = jax.vjp(lambda p: model.apply_fn(p, X).squeeze(), params) # J^T v
    u = vjp_fn(v)[0]

    _, Kv = jax.jvp(lambda p: model.apply_fn(p, X).squeeze(), (params,), (u,)) # J u

    return Kv / X.shape[0]

def compute_ntk_eigs(state, model, X, num_power_iters, rng, **kwargs):
    eigval, eigvec = mvp_power_iteration(
        lambda v: nvp(state.params, model, X, v),
        X.shape[0],
        num_power_iters,
        rng
    )
    return {"max_eigvals": eigval, "max_eigvecs": eigvec}