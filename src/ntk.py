import jax

import jax.numpy as jnp

from functools import partial

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
def nvp(params, model, X, v, precond=None):
    # NTK vector products, (1/n) J P J^T v with P = precond (a pytree like params, see
    # opt.lr_multipliers), or (1/n) J J^T v without it
    _, vjp_fn = jax.vjp(lambda p: model.apply_fn(p, X).squeeze(), params) # J^T v
    u = vjp_fn(v)[0]
    if precond is not None:
        u = jax.tree.map(lambda a, b: a * b, u, precond)

    _, Kv = jax.jvp(lambda p: model.apply_fn(p, X).squeeze(), (params,), (u,)) # J u

    return Kv / X.shape[0]
