import jax
import jax.numpy as jnp
import jax.random as jr

from functools import partial

from linalg import mvp_power_iteration, intrinsic_dim, stable_rank

@partial(jax.jit, static_argnums=(1,))
def get_activations(state, model, X):
    _, acts = jax.vmap(
        lambda x: model.apply_fn(state.params, x, return_activations=True)
    )(X)
    return acts

def neuron_sparsity(state, model=None, X=None, activations=None, **kwargs):
    if activations is None:
        activations = get_activations(state, model, X)
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

def compute_neuron_cov_eigs(state, model, X, num_power_iters, rng, **kwargs):
    acts = kwargs.get("activations", get_activations(state, model, X))
    mvps = neuron_cov_mvp(acts)

    result = {}
    for i, (A, mvp) in enumerate(zip(acts[1:], mvps)):
        dim = A.shape[1]
        
        eigval, eigvec = mvp_power_iteration(mvp, dim, num_power_iters, rng)
        result[f"layer{i+1}_max_eigvals"] = eigval
        result[f"layer{i+1}_max_eigvecs"] = eigvec

    return result

def compute_neuron_cov_stats(state, model, X, num_power_iters, rng, dense_max=2048, **kwargs):
    acts = kwargs.get("activations", get_activations(state, model, X))
    neuron_cov = kwargs.get("neuron_cov", {})
    mvps = neuron_cov_mvp(acts)

    result = {}
    for i, (A, mvp) in enumerate(zip(acts[1:], mvps)):
        dim = A.shape[1]
        n, m = A.shape
        if min(n, m) <= dense_max:
            # A^T A and A A^T share their nonzero spectrum: form the smaller one and
            # read the stats off exactly, instead of Hutchinson-estimating the traces.
            M = (A @ A.T) if m > n else (A.T @ A)
            lam = jnp.linalg.eigvalsh(M / n)[::-1]
            result[f"layer{i+1}_intrinsic_dim"] = lam.sum() / lam[0]
            result[f"layer{i+1}_stable_rank"] = (lam ** 2).sum() / lam[0] ** 2
            continue
        max_eigval = neuron_cov.get(f"layer{i+1}_max_eigvals", None)
        result[f"layer{i+1}_intrinsic_dim"] = intrinsic_dim(
            mvp, dim, num_power_iters, rng, max_eigval=max_eigval
        )
        result[f"layer{i+1}_stable_rank"]   = stable_rank(
            mvp, dim, num_power_iters, rng, max_eigval=max_eigval
        )

    return result