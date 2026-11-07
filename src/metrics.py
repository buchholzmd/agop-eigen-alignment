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
import numpy as np
import jax.scipy.linalg as jla

from agop import nfm as _nfm
from functools import partial

from nn_utils import get_activations, nn_agop, sparsity, _cov_mvp
from ntk import nvp
from hess import _hvp_flat
from linalg import (align_bases, matrix_cosine, mvp_orthogonal_iteration, mvp_power_iteration,
                    hutchinson_trace_estimate, intrinsic_dim, stable_rank, subspace_angles,
                    wasserstein_1d, nuclear)

def agop(state, model, X, layers=(0,), **kwargs):
    return nn_agop(state.params, model, X,
                   activations=kwargs.get("activations", None), layers=layers)

def agop_eigs(state, model, X, k, rng, num_power_iters=None, **kwargs):
    agops = kwargs.get("agop", agop(state, model, X, **kwargs))
    G = agops['layer1_agop']
    dim = G.shape[0]

    if num_power_iters is not None:
        eigvals, eigvecs = mvp_orthogonal_iteration(lambda v: G @ v, dim, k, num_power_iters, rng)
    else:
        spectrum, vecs = jla.eigh(G)
        spectrum, vecs = spectrum[::-1], vecs[:, ::-1]
        return {"eigvals": spectrum[:k], "eigvecs": vecs[:, :k], "spectrum": spectrum}

    return {"eigvals": eigvals, "eigvecs": eigvecs, "spectrum": eigvals}

def nfm(state, **kwargs):
    return _nfm(state.params[0]['weights'])

def nfm_eigs(state, k, rng, num_power_iters=None, **kwargs):
    W = state.params[0]['weights']
    dim = W.shape[1]

    if num_power_iters is not None:
        eigvals, eigvecs = mvp_orthogonal_iteration(lambda v: W.T @ (W @ v), dim, k, num_power_iters, rng)
    else:
        _, sv, Vh = jnp.linalg.svd(W, full_matrices=False)
        spectrum = sv ** 2
        return {"eigvals": spectrum[:k], "eigvecs": Vh[:k].T, "spectrum": spectrum}

    return {"eigvals": eigvals, "eigvecs": eigvecs, "spectrum": eigvals}

def nfa_cosine(state, **kwargs):
    M = kwargs["nfm"]
    G = kwargs["agop"]['layer1_agop']
    return {"matrix_correlation": matrix_cosine(M, G)}

def nfa_alignment(state, **kwargs):
    """The NFA at the level of eigenvectors and eigenspaces: top-r NFM vs top-r AGOP.

    align_bases matches individual eigenvectors, which is unstable when
    eigenvalues are tied. The nfm_agop_ entries are the subspace_recovery suite
    (principal angles) with the NFM eigenspace in place of span(U) as the frame
    and the AGOP eigenvectors as the directions: nfm_agop_alignment[X] is the
    fraction of AGOP eigenvector X inside the NFM's top-r eigenspace.
    """
    M = kwargs["nfm_eigs"]
    G = kwargs["agop_eigs"]
    result = align_bases(M["eigvecs"], G["eigvecs"])
    result.update({f"nfm_agop_{name}": value
                   for name, value in subspace_angles(M["eigvecs"], G["eigvecs"]).items()})
    return result

def neuron_sparsity(state, model=None, X=None, activations=None, **kwargs):
    if activations is None:
        activations = get_activations(state, model, X)
    return sparsity(activations)

def rank_stats(X, num_power_iters=None, rng=None, dense_max=2048):
    """Spectral rank statistics of a PSD matrix, from whichever form you hold.

    X : (k,)   a spectrum already in hand, used directly; or
        (n, m) a factor, whose Gram X^T X / n is the matrix of interest.

    For a factor this dispatches on shape. If min(n, m) <= dense_max the Gram is
    formed and the spectrum read off exactly -- and since X^T X and X X^T share
    their nonzero spectrum, the SMALLER side is formed, which is what makes
    m >> n affordable. lam[0] is then the max eigenvalue, so no power iteration
    runs. Above dense_max the matrix cannot be formed and the traces are
    Hutchinson-estimated matrix-free instead.

    All three ranks equal the true rank on a flat spectrum, which is what makes
    them ranks, and they are ordered stable_rank <= intrinsic_dim <=
    participation_ratio because (lam/lam_max)^2 <= lam/lam_max termwise:

        stable_rank          ||lam||_2^2 / ||lam||_inf^2   dominated by the top
        intrinsic_dim        ||lam||_1   / ||lam||_inf     carries more tail weight
        participation_ratio  ||lam||_1^2 / ||lam||_2^2     how many dominant
                                                           dimensions are
                                                           contributing to the
                                                           spectrum
    """
    if X.ndim == 1:
        lam = jnp.clip(X, 0.0)
        max_eigval = lam.max() + 1e-30
        intrinsic  = lam.sum() / max_eigval
        stable     = (lam ** 2).sum() / max_eigval ** 2
    else:
        n, m = X.shape
        if min(n, m) <= dense_max:
            M = (X @ X.T) if m > n else (X.T @ X)
            lam = jnp.linalg.eigvalsh(M / n)[::-1]
            max_eigval = lam[0]
            intrinsic  = lam.sum() / max_eigval
            stable     = (lam ** 2).sum() / max_eigval ** 2
        else:
            mvp = partial(_cov_mvp, X)
            max_eigval, _ = mvp_power_iteration(mvp, m, num_power_iters, rng)
            intrinsic = intrinsic_dim(mvp, m, num_power_iters, rng, max_eigval=max_eigval)
            stable    = stable_rank(mvp, m, num_power_iters, rng, max_eigval=max_eigval)

    return {
        "max_eigval":          max_eigval,
        "stable_rank":         stable,
        "intrinsic_dim":       intrinsic,
        "participation_ratio": intrinsic ** 2 / (stable + 1e-30),
    }

def neuron_cov_stats(state, model, X, num_power_iters, rng, dense_max=2048, **kwargs):
    acts = kwargs.get("activations", get_activations(state, model, X))

    result = {}
    for i, A in enumerate(acts[1:]):
        for name, value in rank_stats(A, num_power_iters, rng, dense_max).items():
            result[f"layer{i+1}_{name}"] = value

    return result

def val_loss(state, loss_fn, X_val, y_val, **kwargs):
    """Loss on the held-out set. Evaluated at the same theta as every other
    metric, so it is directly comparable to the training loss at that step."""
    outputs = jax.vmap(lambda x: state.apply_fn(state.params, x))(X_val)
    return {"value": loss_fn(outputs, y_val).mean()}

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

def matched_alignment(learned_eigvecs, reference_eigvecs):
    """Hungarian-matched eigenvector overlaps, keyed by the REFERENCE eigenvector X.

    -> (mean, align, rank): align[X] = <v_rank[X], ref_X>^2 and rank[X] = the learned
    eigenvector matched to reference eigenvector X. align_bases keys perm by the
    LEARNED eigenvector (perm[i] = reference matched to learned i); inverting it keys
    both series by the reference, so a series never changes meaning when the
    matching does.
    """
    alignment = align_bases(learned_eigvecs, reference_eigvecs)
    rank  = jnp.argsort(alignment["perm"])
    align = alignment["overlap_matrix"][rank, jnp.arange(rank.shape[0])]
    return alignment["score"], align, rank

def target_alignment(state, target_agop, target_eigvals, target_eigvecs, target_dirs, **kwargs):
    """Learned AGOP and NFM referenced to the target AGOP, which does not move.

    target_dirs : (d, r) the target directions as COLUMNS, i.e. U.T.
    target_agop : the target's AGOP on the training inputs.

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

    rayleigh       = jnp.einsum("dj,de,ej->j", target_dirs, learned_agop, target_dirs)
    agop_nuclear   = nuclear(learned_agop)
    target_nuclear = nuclear(target_agop)
    agop_rank      = rank_stats(kwargs["agop_eigs"]["spectrum"])
    target_rank    = rank_stats(target_eigvals)
    mean_align_true, align_true, rank_true = matched_alignment(agop_eigvecs, target_eigvecs)

    return {
        "agop_cosine":     matrix_cosine(learned_agop, target_agop),
        "nfm_cosine":      matrix_cosine(learned_nfm, target_agop),
        "mean_align_true": mean_align_true,
        "align_true":      align_true,
        "rank_true":       rank_true,
        "agop_nuclear":    agop_nuclear,
        "target_nuclear":  target_nuclear,
        "nuclear_ratio":   agop_nuclear / (target_nuclear + 1e-12),
        "wasserstein1_agop_target": wasserstein_1d(agop_eigvals, target_eigvals),
        "wasserstein1_nfm_agop":    wasserstein_1d(nfm_eigvals, agop_eigvals),
        "rayleigh":       rayleigh,
        "rayleigh_ratio": rayleigh / (target_eigvals + 1e-12),
        **{f"agop_{name}": value for name, value in agop_rank.items()},
        **{f"{name}_ratio": value / (target_rank[name] + 1e-12)
           for name, value in agop_rank.items()},
    }

def egop_alignment(state, target_egop, egop_eigvals, egop_eigvecs, num_power_iters, rng,
                   dense_max=2048, **kwargs):
    """Learned AGOP and NFM against the target's population EGOP, which does not move.

    target_egop : (d, d) the target's EGOP, from multi_index.gaussian_egop.
    egop_eigvals / egop_eigvecs : its top r eigenpairs, computed once per run in run_one.

    The distances are relative, ||C|| / ||EGOP|| with C = G - EGOP, a d x d matrix.
    As in rank_stats, d <= dense_max reads both norms exactly off one eigvalsh(C):
    ||C||_F = ||lambda||_2 and ||C||_2 = max |lambda|. Above dense_max they are
    matrix-free, from the PSD operator C^T C = C^2 (C is symmetric) as in stable_rank:
    ||C||_F^2 = Tr(C^T C) is a Hutchinson estimate and ||C||_2^2 = lambda_max(C^T C) a
    power iteration, num_power_iters each. Power iteration on C itself returns a
    Rayleigh quotient, which stalls when C has eigenvalues of both signs and similar
    size; on C^2 it cannot. The EGOP has rank <= r, so its own norms are exact from
    its r eigenvalues.

    The top-r eigenspace of the EGOP is span(U) (it is U^T Gamma U), so eigenspace-level
    alignment to it is subspace_recovery's agop_ entries; only the eigenvector-level
    matching is computed here.

    Needs "agop", "nfm" and "agop_eigs" above it in metrics_config.
    """
    learned_agop = kwargs["agop"]["layer1_agop"]
    learned_nfm  = kwargs["nfm"]
    mean_align, align, rank = matched_alignment(kwargs["agop_eigs"]["eigvecs"], egop_eigvecs)

    dim = learned_agop.shape[0]
    if dim <= dense_max:
        lam = jnp.linalg.eigvalsh(learned_agop - target_egop)
        frobenius, spectral = jnp.linalg.norm(lam), jnp.abs(lam).max()
    else:
        C   = lambda v: learned_agop @ v - target_egop @ v
        CtC = lambda v: C(C(v))
        frobenius = jnp.sqrt(hutchinson_trace_estimate(CtC, dim, num_power_iters, rng))
        spectral  = jnp.sqrt(mvp_power_iteration(CtC, dim, num_power_iters, rng)[0])

    return {
        "agop_egop_cosine":         matrix_cosine(learned_agop, target_egop),
        "nfm_egop_cosine":          matrix_cosine(learned_nfm, target_egop),
        "egop_nuclear_ratio":       nuclear(learned_agop) / (nuclear(target_egop) + 1e-12),
        "agop_egop_frobenius_dist": frobenius / (jnp.linalg.norm(egop_eigvals) + 1e-12),
        "agop_egop_spectral_dist":  spectral / (egop_eigvals[0] + 1e-12),
        "mean_align_egop":          mean_align,
        "align_egop":               align,
        "rank_egop":                rank,
    }

def subspace_recovery(state, target_dirs, **kwargs):
    """Principal angles to the target subspace, from the NFM and from the AGOP.

    target_dirs : (d, r) the target directions as COLUMNS, i.e. U.T.

    Both frames come straight out of nfm_eigs / agop_eigs, which are already
    column-orthonormal and must sit above this entry in metrics_config.
    """
    nfm_eigvecs  = kwargs["nfm_eigs"]["eigvecs"]
    agop_eigvecs = kwargs["agop_eigs"]["eigvecs"]

    result = {f"nfm_{name}":  value
              for name, value in subspace_angles(nfm_eigvecs, target_dirs).items()}
    result.update({f"agop_{name}": value
                   for name, value in subspace_angles(agop_eigvecs, target_dirs).items()})
    return result


def fit_power_law(x, y, min_points=5, r2_min=0.98):
    """Least-squares fit of y ~ C x^beta in log-log, over the longest linear range.

    Ben Arous et al. identify the fitting range "by visual inspection". This does
    it by scan instead: every contiguous window of at least min_points is fit,
    and the LONGEST window clearing r2_min wins, ties broken by R^2. The exponent
    is then reproducible rather than eyeballed, and the range it was fit over is
    reported alongside it so the claim can be checked.

    Window fits use prefix sums, so the scan is O(n^2) rather than O(n^3).

    Returns exponent = nan when no window clears r2_min -- that is a real answer
    (no power law here), not a failure.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    lx, ly = np.log(x[keep]), np.log(y[keep])
    order = np.argsort(lx)
    lx, ly = lx[order], ly[order]
    n = lx.size
    nan = {"exponent": np.nan, "intercept": np.nan, "r2": np.nan,
           "x_lo": np.nan, "x_hi": np.nan, "n_points": 0}
    if n < min_points:
        return nan

    z  = lambda a: np.concatenate([[0.0], np.cumsum(a)])
    cx, cy, cxx, cyy, cxy = z(lx), z(ly), z(lx*lx), z(ly*ly), z(lx*ly)

    best = None
    for i in range(n - min_points + 1):
        js = np.arange(i + min_points, n + 1)
        k  = (js - i).astype(float)
        sx, sy   = cx[js]-cx[i], cy[js]-cy[i]
        sxx, syy = cxx[js]-cxx[i], cyy[js]-cyy[i]
        sxy      = cxy[js]-cxy[i]
        varx = sxx - sx*sx/k
        vary = syy - sy*sy/k
        cov  = sxy - sx*sy/k
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = np.where((varx > 0) & (vary > 0), cov*cov/(varx*vary), 0.0)
        ok = np.flatnonzero(r2 >= r2_min)
        if ok.size == 0:
            continue
        pick = ok[np.lexsort((r2[ok], k[ok]))[-1]]
        key = (k[pick], r2[pick])
        if best is None or key > best[0]:
            beta = cov[pick] / varx[pick]
            best = (key, beta, (sy[pick] - beta*sx[pick])/k[pick], r2[pick], i, js[pick])

    if best is None:
        return nan
    _, beta, c, r2, i, j = best
    return {"exponent": float(beta), "intercept": float(c), "r2": float(r2),
            "x_lo": float(np.exp(lx[i])), "x_hi": float(np.exp(lx[j-1])),
            "n_points": int(j - i)}


def classify_regime(loss, blowup=100.0, tol=1e-9):
    """monotonic | catapult | diverged.

    A catapult needs both a rise AND a recovery below the pre-rise loss. Without
    the recovery condition every diverging run also looks like a catapult.
    """
    loss = np.asarray(loss, dtype=float).ravel()
    if not np.all(np.isfinite(loss)) or loss[-1] > blowup * np.nanmin(loss):
        return "diverged"
    rises = np.where(np.diff(loss) > tol * np.maximum(1.0, np.abs(loss[:-1])))[0]
    if len(rises) and loss[-1] < loss[rises[0]]:
        return "catapult"
    return "monotonic"


def run_summary(metrics, lr):
    """Run-level scalars for wandb.summary -- one number per run, not a series.

    These become sortable columns in the W&B run table and usable axes in
    scatter plots, which is where a fitted exponent or an edge-of-stability
    crossing belongs. wandb.log() would make them degenerate time series.
    """
    loss = np.asarray([float(v) for v in metrics["loss"]])
    sharp = np.asarray([float(v) for v in metrics["hessian"]["max_eigvals"]]) * lr / 2.0

    crossed = np.flatnonzero(sharp >= 1.0)
    fit = fit_power_law(np.arange(1, loss.size + 1), loss)

    out = {
        "regime":              classify_regime(loss),
        "loss_final":          float(loss[-1]),
        "loss_min":            float(np.nanmin(loss)),
        "loss_exponent":       fit["exponent"],      # loss ~ step^beta
        "loss_exponent_r2":    fit["r2"],
        "loss_fit_lo":         fit["x_lo"],
        "loss_fit_hi":         fit["x_hi"],
        "sharpness_ratio_max": float(np.nanmax(sharp)),
        "eos_first_step":      int(crossed[0]) if crossed.size else -1,
        "eos_fraction":        float((sharp >= 1.0).mean()),
    }
    for group, keys in (("subspace", ("nfm_mean_alignment", "nfm_subspace_dist_sq",
                                      "agop_mean_alignment", "agop_subspace_dist_sq")),
                        ("target",   ("agop_cosine", "mean_align_true", "nuclear_ratio",
                                      "intrinsic_dim_ratio", "stable_rank_ratio"))):
        for key in keys:
            if key in metrics.get(group, {}):
                out[f"final_{key}"] = float(jnp.stack(metrics[group][key])[-1])
    return out
