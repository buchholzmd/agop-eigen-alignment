import jax.numpy as jnp
import jax.random as jr
import numpy as np

from jax import vmap
from linalg import stiefel
from poly import gradient_gram


def multi_index_set(terms, r, alpha_star):
    """target.terms and target.alpha -> (beta, coeffs) for g(z) = sum_m coeffs[m] prod_j P_{beta[m, j]}(z_j).

    beta   : (M, r) int32, the multi-index set Lambda = {beta^(m)} (README notation),
             which poly takes as its alpha argument
    coeffs : (M,) float32

    Coefficients are built from alpha_star (target.alpha), not listed: c_k ~ k^-alpha_star
    over all r directions, then normalized so ||c||_2 = 1 and E[f*^2] = 1 exactly at every
    r. alpha_star = 0 IS the constant-coefficient target, so one parameter covers both
    families and it is sweepable.

    A term maps a RELATIVE direction offset to a Hermite degree. Two forms:

      flat   [{0: 2}]                  broadcast: EVERY direction k gets h_2(z_k)
             [{0: 1}, {0: 2}]          Ben Arous, h_1(z_k) + h_2(z_k) for every k
             [{0: 2}, {0: 1, 1: 1}]    h_2(z_k) + h_1(z_k) h_1(z_{k+1}), interacting

      nested [[{0: 4}], [{0: 3}], [{0: 2}]]
                                       cycled: direction k gets group k mod n_groups,
                                       i.e. Lambda = {4e_1, 3e_2, 2e_3, 4e_4, ...}

    Both are r-independent, so an r sweep regenerates Lambda consistently.
    Each term carries 1/sqrt(terms in its group) so every direction's link has
    unit L2 norm and ||c||_2 = 1 still gives E[f*^2] = 1.
    """
    coeffs = jnp.arange(1, r + 1, dtype=jnp.float32) ** -float(alpha_star)
    coeffs = coeffs / jnp.linalg.norm(coeffs)

    groups = terms if isinstance(terms[0], list) else [terms]
    rows, weights = [], []
    for k in range(r):
        group = groups[k % len(groups)]
        for term in group:
            row = np.zeros(r, dtype=np.int32)
            for offset, degree in term.items():
                row[(k + int(offset)) % r] = int(degree)
            rows.append(row)
            weights.append(coeffs[k] / np.sqrt(len(group)))
    beta   = jnp.asarray(np.stack(rows), dtype=jnp.int32)
    coeffs = jnp.asarray(np.asarray(weights, dtype=np.float32))
    return beta, coeffs


def gaussian_egop(U, beta, coeffs, basis="hermite", normalize=True, scale=1.0):
    """E_x[grad f(x) grad f(x)^T], x ~ N(0, I_d), for f(x) = g(Ux) / scale, in closed form.

    U : (r, d) with orthonormal rows; beta, coeffs : g's multi-indices and coefficients
    in poly's convention (basis, normalize). grad f(x) = U^T grad g(Ux) / scale and
    Ux ~ N(0, I_r), so the EGOP is U^T Gamma U / scale^2 with Gamma = E_z[grad g grad g^T]
    from gradient_gram. Derivation: README, "Target EGOP".
    """
    Gamma = gradient_gram(beta, coeffs, basis=basis, normalize=normalize)
    return U.T @ jnp.asarray(Gamma, dtype=U.dtype) @ U / scale ** 2


def generate_multi_index_data(g, r, X, rng, noise=0.1, normalize='none', return_scale=False):
    """-> f, U, y; with return_scale, also the scale f = g(Ux) / scale divides by."""
    U = stiefel(X.shape[-1], r, rng)

    f_raw = lambda x: g(U @ x)

    if normalize == 'rms':
        outputs = vmap(f_raw)(X)
        scale = jnp.sqrt(jnp.mean(outputs ** 2))
    elif normalize == 'none':
        scale = 1.0
    else:
        raise ValueError(f"Unknown normalization: {normalize}")

    f = lambda x: f_raw(x) / scale

    y = vmap(f)(X) + noise * jr.normal(rng.next(), (X.shape[0],))

    if return_scale:
        return f, U, y, scale
    return f, U, y
