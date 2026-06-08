import jax

import jax.numpy as jnp
import jax.scipy.linalg as jla

from functools import partial

from linalg import align_bases, matrix_cosine, mvp_power_iteration, compute_topk_eigs

@partial(jax.jit, static_argnums=(1,))
def agop(params, model, X, activations=None):
    agops = {}

    def data_jacobian(x, acts):
        jacs = []
        for i, act in enumerate(acts):
            jacs.append(jax.grad(lambda a: model.apply_fn(params, a, init_layer=i).sum())(act))
        return jacs

    if activations is None:
        _, activations = jax.vmap(lambda x: model.apply_fn(params, x, return_activations=True))(X)

    jacs = jax.vmap(data_jacobian)(X, activations)

    for i, jac in enumerate(jacs):
        agops[f'layer{i+1}_agop'] = jac.T @ jac / X.shape[0]

    return agops

def compute_agop(state, model, X, **kwargs):
    acts = kwargs.get("activations", None)
    return agop(state.params, model, X, activations=acts)

def compute_agop_eigs(state, model, X, k, rng, num_power_iters=None, **kwargs):
    agops = kwargs.get("agop", compute_agop(state, model, X, **kwargs))
    G = agops['layer1_agop']
    dim = G.shape[0]

    if num_power_iters is not None:
        eigvals, eigvecs = compute_topk_eigs(lambda v: G @ v, dim, k, num_power_iters, rng)
    else:
        eigvals, eigvecs = jla.eigh(G)
        eigvals, eigvecs = eigvals[::-1][:k], eigvecs[:, ::-1][:, :k]

    return {"eigvals": eigvals, "eigvecs": eigvecs}

@jax.jit
def compute_nfm(state, **kwargs):
    W = state.params[0]['weights']
    dim = W.shape[1]
    return W.T @ W / dim

def compute_nfm_eigs(state, k, rng, num_power_iters=None, **kwargs):
    if num_power_iters is not None:
        W = state.params[0]['weights']
        dim = W.shape[1]
        eigvals, eigvecs = compute_topk_eigs(lambda v: W.T @ (W @ v) / dim, dim, k, num_power_iters, rng)
    else:
        nfm = kwargs.get("nfm", compute_nfm(state, **kwargs))
        eigvals, eigvecs = jla.eigh(nfm)
        eigvals, eigvecs = eigvals[::-1][:k], eigvecs[:, ::-1][:, :k]

    return {"eigvals": eigvals, "eigvecs": eigvecs}

def compute_nfa_correlation(state, **kwargs):
    nfm  = kwargs["nfm"]
    agop = kwargs["agop"]['layer1_agop']
    return {"matrix_correlation": matrix_cosine(nfm, agop)}

def compute_nfa_eigen_alignment(state, **kwargs):
    nfm  = kwargs["nfm_eigs"]
    agop = kwargs["agop_eigs"]
    return align_bases(nfm["eigvecs"], agop["eigvecs"])