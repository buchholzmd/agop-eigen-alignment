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

def topk_eigs(matrix_fn, dim, k, num_iters, rng):
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

def nuclear(G):
    """||G||_* for PSD G. Equals tr(G), since nuclear norm is sum|lambda_i|

    Not jitted: one op, so the dispatch overhead exceeds the work. It inlines
    into whatever jit calls it.
    """
    return jnp.trace(G)

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

@partial(jax.jit, static_argnums=(2,))
def subspace_angles(W, U, k):
    """Principal angles between the top-k right singular subspace of W and row(U).

    W : (m, d)  first layer weights, or any frame whose row space we want
    U : (r, d)  orthonormal target directions, one per row
    k : number of right singular vectors of W to keep

    A is the top-k right singular subspace of W, with orthonormal rows. It reduces
    to the polar factor G_W^{-1/2} W when k = m <= d, but stays well defined when
    m >= d, where row(W) is all of R^d and every alignment is trivially 1.

    C = A U^T is the cosine matrix: its singular values are the cosines of the
    principal angles, and Align(W, u_j) = ||A u_j||^2 is the j-th column sum of C^2.

    sin_max is sqrt(1 - cos_min^2), NOT the top singular value of A U_perp^T: when
    k > r that matrix has singular values equal to 1 for directions of row(W) lying
    entirely outside row(U), which says nothing about the principal angles.
    """
    _, _, Vh = jnp.linalg.svd(W, full_matrices=False)
    A = Vh[:k]
    C = A @ U.T
    cosines = jnp.linalg.svd(C, compute_uv=False)
    alignment = (C ** 2).sum(axis=0)

    return {
        "alignment":        alignment,
        "mean_alignment":   alignment.mean(),
        "cos_max":          cosines[0],
        "cos_min":          cosines[-1],
        "sin_max":          jnp.sqrt(jnp.clip(1.0 - cosines[-1] ** 2, 0.0)),
        "projection_error": jnp.linalg.norm(A.T @ A - U.T @ U) / jnp.sqrt(2.0 * U.shape[0]),
    }
