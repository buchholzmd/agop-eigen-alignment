"""Metric wrappers.

Every function here has the dispatcher's signature -- fn(state, *args, **results,
**kwargs) -- and contains no mathematics of its own: it pulls arrays out of the
results dict and calls the implementation in the file that owns it.  The maths
lives in agop.py, nn_utils.py, ntk.py, hess.py and linalg.py.

Ordering matters.  compute_metrics (opt.py) accumulates results in insertion
order and forwards them as **results, so a wrapper can only see metrics declared
ABOVE it in metrics_config.
"""
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla

from agop import nfm as _nfm
from nn_utils import get_activations, nn_agop, sparsity, neuron_cov_mvp
from ntk import nvp
from hess import _hvp_flat
from linalg import (align_bases, matrix_cosine, topk_eigs, mvp_power_iteration,
                    intrinsic_dim, stable_rank, subspace_angles, wasserstein_1d,
                    nuclear)

def agop(state, model, X, layers=(0,), **kwargs):
    return nn_agop(state.params, model, X,
                   activations=kwargs.get("activations", None), layers=layers)

def agop_eigs(state, model, X, k, rng, num_power_iters=None, **kwargs):
    agops = kwargs.get("agop", agop(state, model, X, **kwargs))
    G = agops['layer1_agop']
    dim = G.shape[0]

    if num_power_iters is not None:
        eigvals, eigvecs = topk_eigs(lambda v: G @ v, dim, k, num_power_iters, rng)
    else:
        eigvals, eigvecs = jla.eigh(G)
        eigvals, eigvecs = eigvals[::-1][:k], eigvecs[:, ::-1][:, :k]

    return {"eigvals": eigvals, "eigvecs": eigvecs}

def nfm(state, **kwargs):
    return _nfm(state.params[0]['weights'])

def nfm_eigs(state, k, rng, num_power_iters=None, **kwargs):
    W = state.params[0]['weights']
    dim = W.shape[1]

    if num_power_iters is not None:
        eigvals, eigvecs = topk_eigs(lambda v: W.T @ (W @ v), dim, k, num_power_iters, rng)
    else:
        _, sv, Vh = jnp.linalg.svd(W, full_matrices=False)
        eigvals, eigvecs = sv[:k] ** 2, Vh[:k].T

    return {"eigvals": eigvals, "eigvecs": eigvecs}

def nfa_cosine(state, **kwargs):
    M = kwargs["nfm"]
    G = kwargs["agop"]['layer1_agop']
    return {"matrix_correlation": matrix_cosine(M, G)}

def nfa_alignment(state, **kwargs):
    M = kwargs["nfm_eigs"]
    G = kwargs["agop_eigs"]
    return align_bases(M["eigvecs"], G["eigvecs"])

def neuron_sparsity(state, model=None, X=None, activations=None, **kwargs):
    if activations is None:
        activations = get_activations(state, model, X)
    return sparsity(activations)

def neuron_cov_eigs(state, model, X, num_power_iters, rng, **kwargs):
    acts = kwargs.get("activations", get_activations(state, model, X))
    mvps = neuron_cov_mvp(acts)

    result = {}
    for i, (A, mvp) in enumerate(zip(acts[1:], mvps)):
        dim = A.shape[1]

        eigval, eigvec = mvp_power_iteration(mvp, dim, num_power_iters, rng)
        result[f"layer{i+1}_max_eigvals"] = eigval
        result[f"layer{i+1}_max_eigvecs"] = eigvec

    return result

def neuron_cov_stats(state, model, X, num_power_iters, rng, dense_max=2048, **kwargs):
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

def ntk_eigs(state, model, X, num_power_iters, rng, **kwargs):
    eigval, eigvec = mvp_power_iteration(
        lambda v: nvp(state.params, model, X, v),
        X.shape[0],
        num_power_iters,
        rng
    )
    return {"max_eigvals": eigval, "max_eigvecs": eigvec}

def hessian_eigs(state, loss_fn, X, y, num_power_iters, rng, **kwargs):
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

def target_alignment(state, target_agop, target_eigvals, target_eigvecs, U, **kwargs):
    """Learned AGOP and NFM referenced to the target AGOP, which does not move.

    target_agop / target_eigvals / target_eigvecs are computed once per run in
    run_one, not per step: the target is fixed, so re-decomposing it every step
    would be pure waste.

    Needs "agop", "nfm", "agop_eigs" and "nfm_eigs" above it in metrics_config.
    """
    learned_agop = kwargs["agop"]["layer1_agop"]
    learned_nfm  = kwargs["nfm"]
    agop_eigvals = kwargs["agop_eigs"]["eigvals"]
    agop_eigvecs = kwargs["agop_eigs"]["eigvecs"]
    nfm_eigvals  = kwargs["nfm_eigs"]["eigvals"]

    alignment      = align_bases(agop_eigvecs, target_eigvecs)
    rayleigh       = jnp.einsum("jd,de,je->j", U, learned_agop, U)
    agop_nuclear   = nuclear(learned_agop)
    target_nuclear = nuclear(target_agop)

    return {
        "agop_cosine":    matrix_cosine(learned_agop, target_agop),
        "nfm_cosine":     matrix_cosine(learned_nfm, target_agop),
        "eigvec_score":   alignment["score"],
        "eigvec_matched": alignment["matched"],
        "eigvec_perm":    alignment["perm"],
        "agop_nuclear":   agop_nuclear,
        "target_nuclear": target_nuclear,
        "nuclear_ratio":  agop_nuclear / (target_nuclear + 1e-12),
        "w1_agop_target": wasserstein_1d(agop_eigvals, target_eigvals),
        "w1_nfm_agop":    wasserstein_1d(nfm_eigvals, agop_eigvals),
        "rayleigh":       rayleigh,
        "rayleigh_ratio": rayleigh / (target_eigvals + 1e-12),
    }

def subspace_recovery(state, U, k, **kwargs):
    """Principal angles to the target subspace, measured from W1 and from the AGOP."""
    W = state.params[0]["weights"]
    agop_eigvecs = kwargs["agop_eigs"]["eigvecs"]

    result = {f"nfm_{name}":  value for name, value in subspace_angles(W, U, k).items()}
    result.update({f"agop_{name}": value
                   for name, value in subspace_angles(agop_eigvecs.T, U, k).items()})
    return result
