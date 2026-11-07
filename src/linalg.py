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
    d = jnp.diagonal(r, offset=0, axis1=-2, axis2=-1)

    x = q * jnp.expand_dims(jnp.sign(d), -2)

    if n < _m:
        return x.mT
    else:
        return x

def mvp_orthogonal_iteration(matrix_fn, dim, k, num_iters, rng):
    '''
    Compute the top-k eigenvalues and eigenvectors of a matrix using orthogonal iteration.
    '''
    if k >= dim:
        num_iters = 0   # Q spans R^dim already; the Rayleigh-Ritz step below is exact

    Q_k = stiefel(dim, k, rng).T
    matrix_fn = jax.vmap(matrix_fn, in_axes=1, out_axes=1)

    def body (_, Q_k):
        Q_k1, _ = jnp.linalg.qr(matrix_fn(Q_k))
        return Q_k1

    Q = jax.lax.fori_loop(0, num_iters, body, Q_k)

    M = Q.T @ matrix_fn(Q)
    eigvals, eigvecs = jnp.linalg.eigh((M + M.T) / 2)
    return eigvals[::-1], Q @ eigvecs[:, ::-1]

def mvp_power_iteration(matrix_fn, dim, num_iters, rng):
    b_k = jr.uniform(rng.next(), shape=(dim,))

    for _ in range(num_iters):
        b_k1 = matrix_fn(b_k)

        b_k = b_k1 / jnp.linalg.norm(b_k1)

    mu = b_k.dot(matrix_fn(b_k)) / jnp.linalg.norm(b_k)**2
    return mu, b_k

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

def nuclear(M):
    """||M||_* for PSD M, which equals tr(M): the nuclear norm is sum|lambda_i|
    and a Gram matrix has no negative eigenvalues. O(d) against O(d^3) for the
    SVD that jnp.linalg.norm(M, 'nuc') would run, and bit-identical on PSD input.

    Not jitted: one op, so dispatch costs more than the work. It inlines into
    whatever jit calls it.
    """
    return jnp.trace(M)

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

def _to_prob(lam, k=None, eps=1e-12):
    """Normalize a spectrum to a probability vector.

    Restricts to the top-k first: the AGOP has rank <= min(n, r) and the NFM
    rank <= m, so a full-length spectrum is mostly numerical noise around zero,
    and distances computed over it measure the regularizer rather than the data.
    """
    lam = jnp.sort(lam)[::-1]
    if k is not None:
        lam = lam[:k]
    lam = jnp.clip(lam, 0.0)
    return lam / (lam.sum() + eps)

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
    return 1 - 2 * count / total

@jax.jit
def wasserstein_1d(lam1, lam2):  # the actual W1
    p, q = _to_prob(lam1), _to_prob(lam2)
    return jnp.sum(jnp.abs(jnp.cumsum(p) - jnp.cumsum(q)))

@jax.jit
def subspace_angles(A, B):
    """Principal angles between span(A) and span(B).

    A : (d, k) orthonormal COLUMNS spanning the first subspace
    B : (d, r) orthonormal COLUMNS spanning the second

    Column convention for both arguments, align_bases uses the same convention

    C = A^T B is the cosine matrix: its singular values are the cosines of the
    principal angles, and Align(A, b_j) = ||A^T b_j||^2 is the j-th column sum
    of C^2.

    subspace_dist = ||sin(theta)||_F is the chordal metric on the Grassmannian.
    ||A^T B_perp||_F = (1/sqrt2)||P_A - P_B||_F only when k <= r, so
    it is computed from the sines directly.
    """
    assert A.shape[0] == B.shape[0], "A and B must share the ambient dimension"

    C = A.T @ B
    alignment = (C ** 2).sum(axis=0)

    cosines = jnp.linalg.svd(C, compute_uv=False)
    sines   = jnp.sqrt(jnp.clip(1.0 - cosines ** 2, 0.0))

    return {
        "alignment":         alignment,
        "mean_alignment":    alignment.mean(),
        "cos_max":           cosines[0],
        "sin_max":           sines[-1],
        "subspace_dist_sq":  jnp.linalg.norm(sines) ** 2, # = r * (1 - mean_alignment) for k <= r
    }
