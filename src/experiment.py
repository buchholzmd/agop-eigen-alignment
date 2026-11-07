"""Experiment driver, lifted verbatim from notebooks/nfa_phase_sweep.ipynb.

Keeping this importable is what lets the same code run in a notebook and under sbatch.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import jax.random as jr

from jax.flatten_util import ravel_pytree
from optax import l2_loss

from utils import RNGKey
from networks import MLP
from opt import TrainState, gd_update, train
from ntk import nvp
from hess import _hvp_flat
from linalg import matrix_cosine, mvp_power_iteration, nuclear
from poly import poly
from multi_index import gaussian_egop, generate_multi_index_data, multi_index_set
from agop import agop
import metrics
from config import get_in, run_config, run_name, group_name, run_tags

def build_data(cfg, rng):
    '''-> X, y, X_val, y_val, U, target_fn, G_star, G_egop.  Sources: multi-index.ipynb cell 4.'''
    n, d, r = get_in(cfg, "data.n"), get_in(cfg, "data.d"), get_in(cfg, "data.r")
    X = jr.normal(rng.next(), (n, d))

    # The multi-index set and coefficients from target.terms and target.alpha; see
    # multi_index_set for the term syntax.
    beta, coeffs = multi_index_set(get_in(cfg, "target.terms"), r, get_in(cfg, "target.alpha"))
    basis, normalize = get_in(cfg, "target.basis"), get_in(cfg, "target.normalize")

    g = lambda z: poly(beta, coeffs, z, basis=basis, normalize=normalize)

    X_val = jr.normal(RNGKey(get_in(cfg, "data.val_seed")).next(),
                      (get_in(cfg, "data.val_n"), d))

    target_fn, U, y, scale = generate_multi_index_data(
        g, r=r, X=X, rng=rng,
        noise=get_in(cfg, "data.noise"),
        normalize=get_in(cfg, "data.normalize"),
        return_scale=True,
    )
    y_val = jax.vmap(target_fn)(X_val)      # noiseless: val_loss -> 0 iff f is recovered

    # G_star is the target's AGOP on the training inputs; G_egop is its population
    # EGOP E_x[grad f* grad f*^T], x ~ N(0, I_d), in closed form: grad f*(x) =
    # U^T grad g(Ux) / scale with Ux ~ N(0, I_r). Both are supported on span(U^T).
    G_egop = gaussian_egop(U, beta, coeffs, basis=basis, normalize=normalize, scale=scale)
    return X, y, X_val, y_val, U, target_fn, agop(target_fn, X), G_egop

def build_model(cfg, rng):
    '''-> model, state, params0.  Sources: cells 7, 8, 9.'''
    d, m = get_in(cfg, "data.d"), get_in(cfg, "model.width")
    layer_shapes = [d] + [m] * (get_in(cfg, "model.depth") - 1) + [1]
    pz = get_in(cfg, "model.parameterization")
    model = MLP(layer_shapes, parameterization=pz, rng=rng)
    state = TrainState.create(apply_fn=model.apply_fn, params=model.params,
                              update_fn=gd_update, parameterization=pz)
    return model, state, state.params


def critical_lr(state, model, X, y, loss_fn, rng, iters=30):
    '''-> ntk_lambda0, hess_lambda0, eta_crit.  Sources: cells 13, 19, 21.

    eta_crit = 2/lambda_max(H) is the GD stability limit. Note 2/lambda_max(NTK) is TWICE
    that, because loss_fn = 2*l2_loss makes the Gauss-Newton Hessian (2/n)J^T J while
    ntk.nvp returns (1/n)J J^T.
    '''
    pf, _ = ravel_pytree(state.params)
    ntk_lambda0, _ = mvp_power_iteration(
        lambda v: nvp(state.params, model, X, v), X.shape[0], iters, rng)
    hess_lambda0, _ = mvp_power_iteration(
        lambda v: _hvp_flat(state.params, state.apply_fn, loss_fn, X, y, v),
        pf.shape[0], iters, rng)
    return float(ntk_lambda0), float(hess_lambda0), 2.0 / float(hess_lambda0)

def make_metrics_config(cfg, model, X, y, loss_fn, rng, X_val=None, y_val=None,
                        U=None, target_agop=None, target_eigvals=None, target_eigvecs=None,
                        target_egop=None, egop_eigvals=None, egop_eigvecs=None):
    '''Verbatim from multi-index.ipynb cell 34, wrapped so each run builds its own.'''
    r = get_in(cfg, 'data.r')
    num_power_iters = get_in(cfg, 'metrics.num_power_iters')
    
    metrics_config = {
        "val":                 {"fn": metrics.val_loss,          "args": (loss_fn, X_val, y_val)},
        "ntk":                 {"fn": metrics.ntk_eigs,          "args": (model, X, num_power_iters, rng)},
        "hessian":             {"fn": metrics.hessian_eigs,      "args": (loss_fn, X, y, num_power_iters, rng)},
        "sparsity":            {"fn": metrics.neuron_sparsity,   "args": (model, X)},
        "neuron_cov":          {"fn": metrics.neuron_cov_stats,  "args": (model, X, 50*num_power_iters, rng)},
        "agop":                {"fn": metrics.agop,              "args": (model, X)},
        "nfm":                 {"fn": metrics.nfm},
        "agop_eigs":           {"fn": metrics.agop_eigs,         "args": (model, X, r, rng)},
        "nfm_eigs":            {"fn": metrics.nfm_eigs,          "args": (r, rng)},
        "nfa_correlation":     {"fn": metrics.nfa_cosine},
        "nfa_eigen_alignment": {"fn": metrics.nfa_alignment},
    }
    if U is not None:
        metrics_config["target"]   = {"fn": metrics.target_alignment,
                                      "args": (target_agop, target_eigvals, target_eigvecs, U.T)}
        metrics_config["subspace"] = {"fn": metrics.subspace_recovery, "args": (U.T,)}
    if target_egop is not None:
        metrics_config["egop"] = {"fn": metrics.egop_alignment,
                                  "args": (target_egop, egop_eigvals, egop_eigvecs,
                                           50*num_power_iters, rng)}
    return metrics_config

def run_one(cfg, log_wandb=None, run=None):
    '''Execute one configuration end to end. -> dict with traces, constants and artifacts.

    run: a W&B run that already exists (a sweep point's); the results are logged to it.
    '''
    rng = RNGKey(get_in(cfg, "data.seed"))
    X, y, X_val, y_val, U, target_fn, G_star, G_egop = build_data(cfg, rng)
    model, state, params0 = build_model(cfg, rng)
    loss_fn = lambda outputs, targets: 2 * l2_loss(outputs, targets)

    ntk_lambda0, hess_lambda0, eta_crit = critical_lr(state, model, X, y, loss_fn, rng)
    lr = get_in(cfg, "optim.lr")
    if get_in(cfg, "optim.lr_mode") == "relative":
        lr = lr * eta_crit

    r = get_in(cfg, "data.r")
    target_eigvals, target_eigvecs = jnp.linalg.eigh(G_star)
    target_eigvals = target_eigvals[::-1][:r]
    target_eigvecs = target_eigvecs[:, ::-1][:, :r]
    egop_eigvals, egop_eigvecs = jnp.linalg.eigh(G_egop)
    egop_eigvals = egop_eigvals[::-1][:r]
    egop_eigvecs = egop_eigvecs[:, ::-1][:, :r]

    mc = make_metrics_config(cfg, model, X, y, loss_fn, rng, X_val=X_val, y_val=y_val,
                             U=U, target_agop=G_star,
                             target_eigvals=target_eigvals, target_eigvecs=target_eigvecs,
                             target_egop=G_egop, egop_eigvals=egop_eigvals, egop_eigvecs=egop_eigvecs)
    state, traj, metrics = train(state, loss_fn, X, y,
                                 lr=lr, steps=get_in(cfg, "optim.steps"),
                                 metrics_config=mc)

    out = dict(cfg=cfg, X=X, y=y, U=U, G_star=G_star, G_egop=G_egop, target_fn=target_fn,
               target_eigvals=target_eigvals, metrics=metrics, traj=traj,
               state=state, model=model, lr=lr, eta_crit=eta_crit,
               ntk_lambda0=ntk_lambda0, hess_lambda0=hess_lambda0, r=r)

    if run is not None or (log_wandb if log_wandb is not None else get_in(cfg, "wandb.enabled")):
        log_run_to_wandb(out, run)
    return out

def log_run_to_wandb(out, run=None):
    import wandb
    cfg = out["cfg"]
    d, r = get_in(cfg, "data.d"), get_in(cfg, "data.r")
    n, m = get_in(cfg, "data.n"), get_in(cfg, "model.width")

    # The config is the search index. Alongside the raw settings, log the derived
    # ratios the sweep is actually organized around, so they can be filtered and
    # plotted against directly instead of being recomputed in the dashboard.
    config = run_config(cfg)
    config.update(
        eta_crit     = out["eta_crit"],
        lr_absolute  = out["lr"],
        lr_relative  = out["lr"] / out["eta_crit"],
        ntk_lambda0  = out["ntk_lambda0"],
        hess_lambda0 = out["hess_lambda0"],
        phi          = m / r,          # width per target direction
        capacity     = n / d,          # samples per ambient dimension
        n_over_r2    = n / r ** 2,     # samples against the target AGOP floor
        n_params     = d * m + 2 * m + 1,
        # How far the target's AGOP on the n training inputs is from its population
        # EGOP: fixed per run, so config rather than a logged series.
        target_agop_egop_cosine        = float(matrix_cosine(out["G_star"], out["G_egop"])),
        target_agop_egop_nuclear_ratio = float(nuclear(out["G_star"]) / nuclear(out["G_egop"])),
    )

    if run is None:
        run = wandb.init(entity=get_in(cfg, "wandb.entity"), project=get_in(cfg, "wandb.project"),
                         name=cfg.get("_wandb_name") or run_name(cfg),
                         group=group_name(cfg),
                         job_type=get_in(cfg, "model.parameterization"),
                         tags=run_tags(cfg),
                         config=config,
                         mode=get_in(cfg, "wandb.mode"),
                         notes=get_in(cfg, "wandb.notes") or None,
                         reinit=True)
    else:
        # A sweep point's run: run_sweep.py has named and tagged it, and the sweep's
        # parameters in its config are locked, so only add what is missing.
        run.config.update({k: v for k, v in config.items() if k not in run.config})
    m, r = out["metrics"], out["r"]
    lr = out["lr"]

    # Sections are split by what is compared, and no section logs a function of
    # another section's metrics. nfa/ compares the NFM M with the AGOPs (learned G,
    # target G*) in any respect; agop/ holds G alone and against G*; subspace/
    # compares the top-r eigenspaces of M and G with span(U).
    for t in range(len(m["loss"])):
        hessian_max = float(m["hessian"]["max_eigvals"][t])
        row = {
            "loss": float(m["loss"][t]),
            "val_loss": float(m["val"]["value"][t]),

            # Loss geometry. sharpness_ratio crosses 1.0 exactly at the edge of
            # stability, so it IS the EOS plot -- wandb cannot draw a reference
            # line on another panel.
            "geometry/hessian_max_eigval": hessian_max,
            "geometry/ntk_max_eigval_x2":  2.0 * float(m["ntk"]["max_eigvals"][t]),
            "geometry/sharpness_ratio":    hessian_max * lr / 2.0,
            "geometry/eos_threshold":      2.0 / lr,
            "geometry/eos_gap":            hessian_max - 2.0 / lr,

            # Neuron statistics
            "neuron/cov_max_eigval":  float(m["neuron_cov"]["layer1_max_eigval"][t]),
            "neuron/intrinsic_dim":   float(m["neuron_cov"]["layer1_intrinsic_dim"][t]),
            "neuron/stable_rank":     float(m["neuron_cov"]["layer1_stable_rank"][t]),
            "neuron/participation_ratio": float(m["neuron_cov"]["layer1_participation_ratio"][t]),
            "neuron/sparsity_layer1": float(m["sparsity"]["value"][t][0].mean()),
            "neuron/frac_dead":       float((m["sparsity"]["value"][t][0] == 1.0).mean()),

            # The NFM M against the AGOPs. nfa_cosine = cos(M, G) is the literal
            # Neural Feature Ansatz; the rest test it by eigenvectors, eigenspaces
            # and spectra; nfm_cosine and nfm_egop_cosine compare M with the target's
            # AGOP G* on the training inputs and with its population EGOP.
            "nfa/nfa_cosine":                float(m["nfa_correlation"]["matrix_correlation"][t]),
            "nfa/nfm_cosine":                float(m["target"]["nfm_cosine"][t]),
            "nfa/nfm_egop_cosine":           float(m["egop"]["nfm_egop_cosine"][t]),
            "nfa/mean_matched_alignment":    float(m["nfa_eigen_alignment"]["score"][t]),
            "nfa/nfm_agop_mean_alignment":   float(m["nfa_eigen_alignment"]["nfm_agop_mean_alignment"][t]),
            "nfa/nfm_agop_cos_max":          float(m["nfa_eigen_alignment"]["nfm_agop_cos_max"][t]),
            "nfa/nfm_agop_sin_max":          float(m["nfa_eigen_alignment"]["nfm_agop_sin_max"][t]),
            "nfa/nfm_agop_subspace_dist_sq": float(m["nfa_eigen_alignment"]["nfm_agop_subspace_dist_sq"][t]),
            "nfa/wasserstein1_nfm_agop":     float(m["target"]["wasserstein1_nfm_agop"][t]),

            # The learned AGOP G alone, against the target's AGOP G* on the training
            # inputs, and (egop in the name) against the target's population EGOP
            # (rayleighX, rayleigh_ratioX, align_trueX, rank_trueX, align_egopX and
            # rank_egopX are added below).
            "agop/agop_nuclear":  float(m["target"]["agop_nuclear"][t]),
            "agop/nuclear_ratio": float(m["target"]["nuclear_ratio"][t]),
            "agop/egop_nuclear_ratio": float(m["egop"]["egop_nuclear_ratio"][t]),
            "agop/agop_stable_rank":         float(m["target"]["agop_stable_rank"][t]),
            "agop/agop_intrinsic_dim":       float(m["target"]["agop_intrinsic_dim"][t]),
            "agop/agop_participation_ratio": float(m["target"]["agop_participation_ratio"][t]),
            "agop/stable_rank_ratio":         float(m["target"]["stable_rank_ratio"][t]),
            "agop/intrinsic_dim_ratio":       float(m["target"]["intrinsic_dim_ratio"][t]),
            "agop/participation_ratio_ratio": float(m["target"]["participation_ratio_ratio"][t]),
            "agop/wasserstein1_agop_target":  float(m["target"]["wasserstein1_agop_target"][t]),
            "agop/agop_cosine":               float(m["target"]["agop_cosine"][t]),
            "agop/agop_egop_cosine":          float(m["egop"]["agop_egop_cosine"][t]),
            "agop/agop_egop_frobenius_dist":  float(m["egop"]["agop_egop_frobenius_dist"][t]),
            "agop/agop_egop_spectral_dist":   float(m["egop"]["agop_egop_spectral_dist"][t]),
            "agop/mean_align_true":           float(m["target"]["mean_align_true"][t]),
            "agop/mean_align_egop":           float(m["egop"]["mean_align_egop"][t]),

            # Top-r eigenspaces of M and G against span(U) (align_dirX is added below).
            "subspace/nfm_mean_alignment":   float(m["subspace"]["nfm_mean_alignment"][t]),
            "subspace/nfm_cos_max":          float(m["subspace"]["nfm_cos_max"][t]),
            "subspace/nfm_sin_max":          float(m["subspace"]["nfm_sin_max"][t]),
            "subspace/nfm_subspace_dist_sq": float(m["subspace"]["nfm_subspace_dist_sq"][t]),
            "subspace/agop_mean_alignment":  float(m["subspace"]["agop_mean_alignment"][t]),
            "subspace/agop_sin_max":         float(m["subspace"]["agop_sin_max"][t]),
            "subspace/agop_subspace_dist_sq": float(m["subspace"]["agop_subspace_dist_sq"][t]),
        }
        # Per index X, one metric at a time so each family's panels stay together.
        # X is the target direction u_X for rayleigh / align_dir, the target AGOP
        # eigenvector v*_X for align_true / rank_true, the target EGOP's eigenvector
        # for align_egop / rank_egop, and the learned AGOP eigenvector v_X for
        # nfm_agop_align_dir.
        per_index = (
            ("nfa/nfm_agop_align_dir",  m["nfa_eigen_alignment"]["nfm_agop_alignment"], float),
            ("agop/rayleigh",           m["target"]["rayleigh"],        float),
            ("agop/rayleigh_ratio",     m["target"]["rayleigh_ratio"],  float),
            ("agop/align_true",         m["target"]["align_true"],      float),
            ("agop/rank_true",          m["target"]["rank_true"],       int),
            ("agop/align_egop",         m["egop"]["align_egop"],        float),
            ("agop/rank_egop",          m["egop"]["rank_egop"],         int),
            ("subspace/align_dir",      m["subspace"]["nfm_alignment"], float),
        )
        for key, series, cast in per_index:
            for j in range(r):
                row[f"{key}{j}"] = cast(series[t][j])
        wandb.log(row, step=t)

    run.summary.update(metrics.run_summary(m, lr))
    log_directional_slices(out)
    run.finish()
    return run

def log_directional_slices(out):
    '''Target and network sliced along the top-r rows of U, logged as an image.'''
    import wandb
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    U, r = out["U"], out["r"]
    t = jnp.linspace(-3.0, 3.0, 201)
    network_fn = lambda x: out["model"].apply_fn(out["state"].params, x)

    fig, axs = plt.subplots(1, r, figsize=(3.2 * r, 3.0), squeeze=False)
    for j in range(r):
        Z = t[:, None] * U[j][None, :]
        axs[0, j].plot(t, jax.vmap(out["target_fn"])(Z), label="target", lw=2)
        axs[0, j].plot(t, jax.vmap(network_fn)(Z), label="network", ls="--")
        axs[0, j].set_title(f"direction {j}")
        axs[0, j].set_xlabel("t")
    axs[0, 0].set_ylabel("f(t u_j)")
    axs[0, 0].legend(fontsize=8)
    fig.tight_layout()
    wandb.log({"slices/directional": wandb.Image(fig)})
    plt.close(fig)
