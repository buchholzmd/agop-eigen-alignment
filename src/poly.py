import jax
import jax.numpy as jnp

from functools import partial

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