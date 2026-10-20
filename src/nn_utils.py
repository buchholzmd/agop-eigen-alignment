import jax
import jax.numpy as jnp
import jax.random as jr

from functools import partial

from agop import agop

@partial(jax.jit, static_argnums=(1,))
def get_activations(state, model, X):
    _, acts = jax.vmap(
        lambda x: model.apply_fn(state.params, x, return_activations=True)
    )(X)
    return acts

def sparsity(activations):
    return [jnp.mean(A <= 1e-8, axis=0) for A in activations[1:]]

def neuron_covariance(state, model=None, X=None, activations=None, **kwargs):
    if activations is None:
        activations = get_activations(state, model, X)
    return [A.T @ A / A.shape[0] for A in activations[1:]]

@jax.jit
def _cov_mvp(A, v):
    """v -> A^T A v / n.  `A` is an ARGUMENT, not a captured constant, so this
    compiles once per (shape, dtype) rather than once per call."""
    return A.T @ (A @ v) / A.shape[0]

def neuron_cov_mvp(acts):
    return [partial(_cov_mvp, A) for A in acts[1:]]

@partial(jax.jit, static_argnums=(1,), static_argnames=("layers",))
def nn_agop(params, model, X, activations=None, layers=(0,)):
    """Network wrapper: agop applied at the input of each layer in `layers`.

    layers=(0,) gives only the d x d input-space AGOP (all the NFA metrics use).
    layers=None reproduces the old all-layers behaviour.

    `layers` must be static: it is branched on at trace time.
    """
    if activations is None:
        _, activations = jax.vmap(
            lambda x: model.apply_fn(params, x, return_activations=True))(X)
    out = {}
    for i, A in enumerate(activations):
        if layers is not None and i not in layers:
            continue
        out[f"layer{i+1}_agop"] = agop(
            lambda a, _i=i: model.apply_fn(params, a, init_layer=_i).sum(), A)
    return out
