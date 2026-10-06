import jax

from functools import partial

from opt import batch_loss
from linalg import mvp_power_iteration

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

def compute_hessian_eigs(state, loss_fn, X, y, num_power_iters, rng, **kwargs):
    params_flat, _ = jax.flatten_util.ravel_pytree(state.params)

    eigval, eigvec = mvp_power_iteration(
            lambda v: _hvp_flat(
                state.params,
                apply_fn=state.apply_fn, 
                loss_fn=loss_fn,
                X=X,
                y=y,
                v_flat=v
            ),
            params_flat.shape[0],
            num_power_iters,
            rng
        )
    return {"max_eigvals": eigval, "max_eigvecs": eigvec}