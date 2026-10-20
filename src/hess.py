import jax

from functools import partial

from opt import batch_loss

@partial(jax.jit, static_argnames=("apply_fn", "loss_fn"))
def _hvp_flat(params, apply_fn, loss_fn, X, y, v_flat):
    """Flattened Hessian-vector product.

    `f` is built INSIDE the jitted function rather than passed in as a static
    argument: batch_loss() returns a fresh closure on every call, and static
    args hash by identity, so passing it would recompile on every power iteration.
    Traced once per (apply_fn, loss_fn) pair instead.
    """
    _, unravel = jax.flatten_util.ravel_pytree(params)
    f = batch_loss(apply_fn, loss_fn, (X, y))
    g = lambda p: f(p)[0]
    hv = jax.jvp(jax.grad(g), (params,), (unravel(v_flat),))[1]
    return jax.flatten_util.ravel_pytree(hv)[0]
