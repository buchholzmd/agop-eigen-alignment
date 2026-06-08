import jax
import optax

import jax.numpy as jnp
import jax.random as jr

from functools import partial

def stiefel(m, n, rng, shape=()):
    if m is None:
        _m = n
    else:
        _m = m
    
    z = jr.normal(rng.next(), (*shape, max(n, _m), min(n, _m)))

    q, r = jnp.linalg.qr(z)
    d = jnp.diagonal(r, offset=0, axis1=0, axis2=1) 

    x = q * jnp.expand_dims(jnp.sign(d), -2)

    if n < _m:
        return x.mT
    else:
        return x

def mvp_power_iteration(matrix_fn, dim, num_iters, rng):
    b_k = jr.uniform(rng.next(), shape=(dim,))

    for _ in range(num_iters):
        b_k1 = matrix_fn(b_k)

        b_k = b_k1 / jnp.linalg.norm(b_k1)

    mu = b_k.dot(matrix_fn(b_k)) / jnp.linalg.norm(b_k)**2
    return mu, b_k

def compute_topk_eigs(mvp, dim, k, num_iters, rng):
    vecs = []
    vals = []

    for _ in range(k):
        val, vec = mvp_power_iteration(mvp, dim, num_iters, rng)

        vecs.append(vec)
        vals.append(val)

        # deflation
        def mvp_deflated(v, _mvp=mvp, _val=val, _vec=vec):
            return _mvp(v) - _val * (_vec @ v) * _vec

        mvp = mvp_deflated

    return jnp.array(vals), jnp.stack(vecs, axis=1)

def hutchinson_trace_estimate(mvp, dim, num_iters, rng):
    trace_estimate = 0.0
    for _ in range(num_iters):
        v = jr.rademacher(rng.next(), (dim,))
        trace_estimate += v @ mvp(v)
    return trace_estimate / num_iters

def intrinsic_dim(mvp, dim, num_iters, rng, max_eigval=None):
    trace = hutchinson_trace_estimate(
            mvp, 
            dim, 
            num_iters,
            rng
    )
    if max_eigval is None:
        max_eigval, _ = mvp_power_iteration(
                mvp,
                dim,
                num_iters,
                rng
        )
    return trace / max_eigval

def stable_rank(mvp, dim, num_iters, rng, max_eigval=None):
    frob_sq = hutchinson_trace_estimate(
            lambda v: mvp(mvp(v)), 
            dim, 
            num_iters,
            rng
    )

    if max_eigval is None:
        max_eigval, _ = mvp_power_iteration(
                mvp,
                dim,
                num_iters,
                rng
        )
    return frob_sq / (max_eigval ** 2)

@jax.jit
def matrix_cosine(A, B):
    return jnp.sum(A * B) / (jnp.linalg.norm(A) * jnp.linalg.norm(B) + 1e-12)

@jax.jit
def align_bases(QA, QB):
    """
    QA, QB: (d, k) matrices with orthonormal columns
    """
    W = (QA.T @ QB) ** 2

    # Hungarian solves min cost → negate for max
    costs = -W

    row_ind, col_ind = optax.assignment.hungarian_algorithm(costs)

    matched = W[row_ind, col_ind]
    score = jnp.mean(matched)

    # permutation vector: maps i -> col_ind[i]
    perm = jnp.zeros(W.shape[0], dtype=jnp.int32)
    perm = perm.at[row_ind].set(col_ind)

    return {
        "score": score,
        "perm": perm,
        "matched": matched,
        "overlap_matrix": W,
    }

@jax.jit
def perm_hamming(perm1, perm2):
    assert perm1.shape == perm2.shape
    return jnp.mean(perm1 != perm2)

@jax.jit
def kendall_tau(perm):
    k = perm.shape[0]
    count = 0
    total = k * (k - 1) // 2
    for i in range(k):
        for j in range(i+1, k):
            count += (perm[i] > perm[j])
    return count / total

@jax.jit
def spectral_l2(lam1, lam2):
    lam1 = lam1 / lam1.sum()
    lam2 = lam2 / lam2.sum()
    return jnp.linalg.norm(lam1 - lam2)

@jax.jit
def spectral_kl(lam1, lam2, eps=1e-8):
    p = lam1 / lam1.sum()
    q = lam2 / lam2.sum()
    return jnp.sum(p * jnp.log((p + eps) / (q + eps)))

@jax.jit
def wasserstein_1d(lam1, lam2):
    p = lam1 / lam1.sum()
    q = lam2 / lam2.sum()

    cdf_p = jnp.cumsum(p)
    cdf_q = jnp.cumsum(q)

    return jnp.linalg.norm(cdf_p - cdf_q)