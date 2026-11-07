import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np

from functools import partial


def _to_orthonormal_hermite(k, basis, normalize):
    """P_k(z) in the basis's own convention, as [(j, w)] with P_k = sum w h_j."""
    if basis == "hermite":
        return [(k, 1.0 if normalize else math.sqrt(math.factorial(k)))]   # He_k = sqrt(k!) h_k
    if basis == "monomial":
        # z^k = sum_p k! / (2^p p! (k-2p)!) He_{k-2p}, and He_j = sqrt(j!) h_j
        return [(k - 2 * p, math.factorial(k)
                 / (2 ** p * math.factorial(p) * math.sqrt(math.factorial(k - 2 * p))))
                for p in range(k // 2 + 1)]
    raise ValueError(f"Unknown basis: {basis!r}")


def hermite_expansion(alpha, coeffs, basis="monomial", normalize=False):
    """g's coefficients in the orthonormal Hermite basis h_beta(z) = prod_j h_{beta_j}(z_j).

    -> {beta: coefficient}, beta an r-tuple. Exact, for every basis poly accepts.
    """
    out = {}
    for a, c in zip(np.asarray(alpha), np.asarray(coeffs, dtype=np.float64)):
        per_coord = [_to_orthonormal_hermite(int(k), basis, normalize) for k in a]
        for combo in itertools.product(*per_coord):
            beta = tuple(j for j, _ in combo)
            out[beta] = out.get(beta, 0.0) + c * math.prod(w for _, w in combo)
    return out


def gradient_gram(alpha, coeffs, basis="monomial", normalize=False):
    """E_z[grad g(z) grad g(z)^T] for z ~ N(0, I_r), in closed form: an (r, r) array.

    In the orthonormal Hermite basis d/dz_i h_beta = sqrt(beta_i) h_{beta - e_i}, so
    d_i g = sum_beta v_beta[i] h_beta with v_beta[i] = c_{beta + e_i} sqrt(beta_i + 1),
    and orthonormality gives E[d_i g d_k g] = sum_beta v_beta[i] v_beta[k]. Derivation:
    README, "Target EGOP". alpha holds the multi-indices (the README's beta), named as
    in poly.
    """
    r = np.asarray(alpha).shape[1]
    rows = {}
    for beta, c in hermite_expansion(alpha, coeffs, basis, normalize).items():
        for i in np.flatnonzero(beta):
            lowered = list(beta)
            lowered[i] -= 1
            rows.setdefault(tuple(lowered), np.zeros(r))[i] += c * math.sqrt(beta[i])
    if not rows:
        return np.zeros((r, r))
    V = np.stack(list(rows.values()))
    return V.T @ V

def poly(alpha, coeffs, z, basis="monomial", normalize=False, max_order=None):
    """g(z) = sum_m coeffs[m] * prod_j P_{alphas[m,j]}(z_j).

    alpha  : (M, r) int   multi-indices
    coeffs : (M,)   float coefficients in the chosen basis
    """
    alpha = jnp.asarray(alpha, dtype=jnp.int32)
    coeffs = jnp.asarray(coeffs)
    K = int(max_order) if max_order is not None else int(jnp.max(alpha))
    z = jnp.asarray(z)
    if basis == "monomial":
        B = z[None, :] ** jnp.arange(K + 1)[:, None]
    elif basis == "hermite":
        B = hermite(K, z, normalize=normalize)      # already (K+1, *z.shape)
    else:
        raise ValueError(f"Unknown basis: {basis!r}")                     # (K+1, r)
    terms = B[alpha, jnp.arange(alpha.shape[1])[None, :]]        # (M, r)
    return coeffs @ jnp.prod(terms, axis=1)

@partial(jax.jit, static_argnums=(0,), static_argnames=("normalize",))
def hermite(k, x, normalize=False):
    """Probabilists' Hermite basis evaluated elementwise on `z`.

    Returns B with shape (K + 1, *z.shape):

        B[k] = He_k(z)                       if normalize is False
        B[k] = He_k(z) / sqrt(k!) =: h_k(z)  if normalize is True

    `K` must be a static Python int (pass it positionally).  One lax.scan:
    O(K) sequential steps, O(K * z.size) flops, O(K * z.size) memory.
    """
    k = int(k)
    if k < 0:
        raise ValueError(f"K must be >= 0, got {k}")

    x = jnp.asarray(x)
    x = x.astype(jnp.result_type(x.dtype, jnp.float32))   # never recurse in int
    one = jnp.ones_like(x)

    if k == 0:
        return one[None]

    if normalize:
        # h_n = (x h_{n-1} - sqrt(n-1) h_{n-2}) / sqrt(n)
        def step(carry, n):
            hm2, hm1 = carry
            hn = (x * hm1 - jnp.sqrt(n - 1.0) * hm2) * jax.lax.rsqrt(n)
            return (hm1, hn), hn
    else:
        # He_n = x He_{n-1} - (n-1) He_{n-2}
        def step(carry, n):
            hm2, hm1 = carry
            hn = x * hm1 - (n - 1.0) * hm2
            return (hm1, hn), hn

    ns = jnp.arange(2, k + 1, dtype=x.dtype)   # empty array when k == 1
    _, He = jax.lax.scan(step, (one, x), ns)   # (k - 1, *x.shape)

    return jnp.concatenate([one[None], x[None], He], axis=0)