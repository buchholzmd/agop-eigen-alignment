import jax

from functools import partial

from opt import batch_loss
from linalg import mvp_power_iteration

@partial(jax.jit, static_argnums=(0, 3))
def hvp(f, params, v, has_aux=False):
    fn = lambda p: f(p)[0] if has_aux else f
    return jax.jvp(
        jax.grad(fn),
        (params,),
        (v,)
    )[1]

def unravel_hvp(f, params, v_flat, unravel, has_aux=False):
    v_tree = unravel(v_flat)
    return jax.flatten_util.ravel_pytree(hvp(f, params, v_tree, has_aux=has_aux))[0]

def compute_hessian_eigs(state, loss_fn, X, y, num_power_iters, rng, **kwargs):
    has_aux = 'activations' in kwargs

    params_flat, unravel = jax.flatten_util.ravel_pytree(state.params)
    eigval, eigvec = mvp_power_iteration(
        lambda v: unravel_hvp(
            batch_loss(state.apply_fn, loss_fn, (X, y)),
            state.params,
            v,
            unravel,
            has_aux=has_aux
        ),
        params_flat.shape[0],
        num_power_iters,
        rng
    )
    return {"max_eigvals": eigval, "max_eigvecs": eigvec}