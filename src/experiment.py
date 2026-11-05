"""Experiment driver, lifted verbatim from notebooks/nfa_phase_sweep.ipynb.

Keeping this importable is what lets the same code run in a notebook and under sbatch.
"""
import os

import jax
import jax.numpy as jnp
import jax.random as jr

from jax.flatten_util import ravel_pytree
from optax import l2_loss

from utils import RNGKey
from networks import MLP
from opt import TrainState, gd_update, train
from ntk import nvp
from hess import _hvp_flat
from linalg import mvp_power_iteration
from poly import poly
from multi_index import generate_multi_index_data
from agop import agop
import metrics
from config import get_in, run_config, run_name, group_name, run_tags

def build_data(cfg, rng):
    '''-> X, y, U, target_fn, G_star.  Sources: multi-index.ipynb cell 4.'''
    n, d, r = get_in(cfg, "data.n"), get_in(cfg, "data.d"), get_in(cfg, "data.r")
    X = jr.normal(rng.next(), (n, d))

    coeffs = jnp.asarray(get_in(cfg, "target.coeffs"), dtype=jnp.float32)
    orders = jnp.asarray(get_in(cfg, "target.orders"), dtype=jnp.int32)
    # Span all r latent directions: cycle the exponent pattern, and continue the
    # coefficient list's own power law k^-s past its length.  Without this the target
    # silently ignores coordinates len(orders)..r-1, so every swept r trains an r=3
    # target.  Renormalizing to ||c||_2 = 1 makes E[f*^2] = 1 analytically at every r.
    if r > coeffs.shape[0]:
        s = -jnp.log(coeffs[1] / coeffs[0]) / jnp.log(2.0)
        coeffs = coeffs[0] * jnp.arange(1, r + 1, dtype=coeffs.dtype) ** -s
    coeffs = coeffs[:r] / jnp.linalg.norm(coeffs[:r])
    orders = orders[jnp.arange(r) % orders.shape[0]]
    alpha  = jnp.diag(orders)                      # 1-D orders == one term per coordinate
    g = lambda z: poly(alpha, coeffs, z,
                       basis=get_in(cfg, "target.basis"),
                       normalize=get_in(cfg, "target.normalize"))

    target_fn, U, y = generate_multi_index_data(
        g, r=r, X=X, rng=rng,
        noise=get_in(cfg, "data.noise"),
        normalize=get_in(cfg, "data.normalize"),
    )
    return X, y, U, target_fn, agop(target_fn, X)

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

def make_metrics_config(cfg, model, X, y, loss_fn, rng,
                        U=None, target_agop=None, target_eigvals=None, target_eigvecs=None):
    '''Verbatim from multi-index.ipynb cell 34, wrapped so each run builds its own.'''
    r = get_in(cfg, 'data.r')
    num_power_iters = get_in(cfg, 'metrics.num_power_iters')
    
    metrics_config = {
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
    return metrics_config

def run_one(cfg, log_wandb=None):
    '''Execute one configuration end to end. -> dict with traces, constants and artifacts.'''
    rng = RNGKey(get_in(cfg, "data.seed"))
    X, y, U, target_fn, G_star = build_data(cfg, rng)
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

    mc = make_metrics_config(cfg, model, X, y, loss_fn, rng, U=U, target_agop=G_star,
                             target_eigvals=target_eigvals, target_eigvecs=target_eigvecs)
    state, traj, metrics = train(state, loss_fn, X, y,
                                 lr=lr, steps=get_in(cfg, "optim.steps"),
                                 metrics_config=mc)

    out = dict(cfg=cfg, X=X, y=y, U=U, G_star=G_star, target_fn=target_fn,
               target_eigvals=target_eigvals, metrics=metrics, traj=traj,
               state=state, model=model, lr=lr, eta_crit=eta_crit,
               ntk_lambda0=ntk_lambda0, hess_lambda0=hess_lambda0, r=r)

    if (log_wandb if log_wandb is not None else get_in(cfg, "wandb.enabled")):
        log_run_to_wandb(out)
    return out

def log_run_to_wandb(out):
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
    )

    run = wandb.init(entity=get_in(cfg, "wandb.entity"), project=get_in(cfg, "wandb.project"),
                     name=cfg.get("_wandb_name") or run_name(cfg),
                     group=group_name(cfg),
                     job_type=get_in(cfg, "model.parameterization"),
                     tags=run_tags(cfg),
                     config=config,
                     mode=get_in(cfg, "wandb.mode"),
                     notes=get_in(cfg, "wandb.notes") or None,
                     reinit=True)
    m, r = out["metrics"], out["r"]
    lr = out["lr"]
    matched = jnp.stack(m["nfa_eigen_alignment"]["matched"])

    for t in range(len(m["loss"])):
        hessian_max = float(m["hessian"]["max_eigvals"][t])
        row = {
            "loss": float(m["loss"][t]),

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

            # NFM, against the AGOP (the literal Neural Feature Ansatz)
            "nfm/cosine_to_agop": float(m["nfa_correlation"]["matrix_correlation"][t]),
            "nfm/mean_alignment": float(matched[t].mean()),

            # Spectrum, against the target AGOP
            "spectrum/agop_cosine":    float(m["target"]["agop_cosine"][t]),
            "spectrum/nfm_cosine":     float(m["target"]["nfm_cosine"][t]),
            "spectrum/eigvec_score":   float(m["target"]["eigvec_score"][t]),
            "spectrum/agop_nuclear":   float(m["target"]["agop_nuclear"][t]),
            "spectrum/nuclear_ratio":  float(m["target"]["nuclear_ratio"][t]),
            "spectrum/w1_agop_target": float(m["target"]["w1_agop_target"][t]),
            "spectrum/w1_nfm_agop":    float(m["target"]["w1_nfm_agop"][t]),
            "spectrum/agop_stable_rank":         float(m["target"]["agop_stable_rank"][t]),
            "spectrum/agop_intrinsic_dim":       float(m["target"]["agop_intrinsic_dim"][t]),
            "spectrum/agop_participation_ratio": float(m["target"]["agop_participation_ratio"][t]),
            "spectrum/stable_rank_ratio":         float(m["target"]["stable_rank_ratio"][t]),
            "spectrum/intrinsic_dim_ratio":       float(m["target"]["intrinsic_dim_ratio"][t]),
            "spectrum/participation_ratio_ratio": float(m["target"]["participation_ratio_ratio"][t]),

            # Subspace recovery
            "subspace/nfm_mean_alignment":   float(m["subspace"]["nfm_mean_alignment"][t]),
            "subspace/nfm_cos_max":          float(m["subspace"]["nfm_cos_max"][t]),
            "subspace/nfm_sin_max":          float(m["subspace"]["nfm_sin_max"][t]),
            "subspace/nfm_subspace_dist":    float(m["subspace"]["nfm_subspace_dist"][t]),
            "subspace/agop_mean_alignment":  float(m["subspace"]["agop_mean_alignment"][t]),
            "subspace/agop_sin_max":         float(m["subspace"]["agop_sin_max"][t]),
            "subspace/agop_subspace_dist":   float(m["subspace"]["agop_subspace_dist"][t]),
        }
        # Per direction, keyed by the TRUE direction j so the series never moves.
        # The matched eigenvector index is logged as its own series instead.
        for j in range(r):
            row[f"nfa/align_true{j}"]         = float(m["target"]["eigvec_matched"][t][j])
            row[f"nfa/rank_true{j}"]          = int(m["target"]["eigvec_perm"][t][j])
            row[f"subspace/align_dir{j}"]     = float(m["subspace"]["nfm_alignment"][t][j])
            row[f"spectrum/rayleigh{j}"]      = float(m["target"]["rayleigh"][t][j])
            row[f"spectrum/rayleigh_ratio{j}"]= float(m["target"]["rayleigh_ratio"][t][j])
        wandb.log(row, step=t)

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
