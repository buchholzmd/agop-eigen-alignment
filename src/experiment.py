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
from agop import (compute_agop, compute_nfm, compute_agop_eigs, compute_nfm_eigs,
                  compute_nfa_correlation, compute_nfa_eigen_alignment)
from nn_utils import neuron_sparsity, compute_neuron_cov_eigs, compute_neuron_cov_stats
from ntk import compute_ntk_eigs
from hess import compute_hessian_eigs
from config import get_in, flatten, run_name, group_name

def target_agop(f, X):
    '''(1/n) sum_i grad f(x_i) grad f(x_i)^T -- same estimator as agop(), same X.'''
    J = jax.vmap(jax.grad(f))(X)
    return J.T @ J / X.shape[0]


def build_data(cfg, rng):
    '''-> X, y, U, target_fn, G_star.  Sources: multi-index.ipynb cell 4.'''
    n, d, r = get_in(cfg, "data.n"), get_in(cfg, "data.d"), get_in(cfg, "data.r")
    X = jr.normal(rng.next(), (n, d))

    coeffs = jnp.asarray(get_in(cfg, "target.coeffs"))
    orders = jnp.asarray(get_in(cfg, "target.orders"))
    alpha  = jnp.diag(orders)                      # 1-D orders == one term per coordinate
    g = lambda z: poly(alpha, coeffs, z,
                       basis=get_in(cfg, "target.basis"),
                       normalize=get_in(cfg, "target.normalize"))

    target_fn, U, y = generate_multi_index_data(
        g, r=r, X=X, rng=rng,
        noise=get_in(cfg, "data.noise"),
        normalize=get_in(cfg, "data.normalize"),
    )
    return X, y, U, target_fn, target_agop(target_fn, X)

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

def make_metrics_config(cfg, model, X, y, loss_fn, rng):
    '''Verbatim from multi-index.ipynb cell 34, wrapped so each run builds its own.'''
    r = get_in(cfg, 'data.r')
    num_power_iters = get_in(cfg, 'metrics.num_power_iters')
    
    metrics_config = {
        "ntk":                 {"fn": compute_ntk_eigs,                 "args": (model, X, num_power_iters, rng)},
        "hessian":             {"fn": compute_hessian_eigs,             "args": (loss_fn, X, y, num_power_iters, rng)},
        "sparsity":            {"fn": neuron_sparsity,                  "args": (model, X)},
        "neuron_cov":          {"fn": compute_neuron_cov_eigs,          "args": (model, X, 10*num_power_iters, rng)},
        "neuron_cov_int_dim":  {"fn": compute_neuron_cov_stats,         "args": (model, X, 50*num_power_iters, rng)},
        "agop":                {"fn": compute_agop,                     "args": (model, X)},
        "nfm":                 {"fn": compute_nfm},
        "agop_eigs":           {"fn": compute_agop_eigs,                "args": (model, X, r, rng, 10*num_power_iters)},
        "nfm_eigs":            {"fn": compute_nfm_eigs,                 "args": (r, rng, 10*num_power_iters)},
        "nfa_correlation":     {"fn": compute_nfa_correlation},
        "nfa_eigen_alignment": {"fn": compute_nfa_eigen_alignment},
    }
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

    mc = make_metrics_config(cfg, model, X, y, loss_fn, rng)
    state, traj, metrics = train(state, loss_fn, X, y,
                                 lr=lr, steps=get_in(cfg, "optim.steps"),
                                 metrics_config=mc)

    out = dict(cfg=cfg, X=X, y=y, U=U, G_star=G_star, metrics=metrics, traj=traj,
               state=state, model=model, lr=lr, eta_crit=eta_crit,
               ntk_lambda0=ntk_lambda0, hess_lambda0=hess_lambda0, r=get_in(cfg, "data.r"))

    if (log_wandb if log_wandb is not None else get_in(cfg, "wandb.enabled")):
        log_run_to_wandb(out)
    return out

def log_run_to_wandb(out):
    import wandb
    cfg = out["cfg"]
    cfgf = flatten(cfg)
    cfgf.update(eta_crit=out["eta_crit"], lr_absolute=out["lr"],
                ntk_lambda0=out["ntk_lambda0"], hess_lambda0=out["hess_lambda0"])
    run = wandb.init(entity=get_in(cfg, "wandb.entity"), project=get_in(cfg, "wandb.project"),
                     name=cfg.get("_wandb_name") or run_name(cfg), group=group_name(cfg), config=cfgf,
                     mode=get_in(cfg, "wandb.mode"), reinit=True)
    m, r = out["metrics"], out["r"]
    for t in range(len(m["loss"])):
        row = {
            "loss": float(m["loss"][t]),
            "ntk/max_eigval":     float(m["ntk"]["max_eigvals"][t]),
            "hessian/max_eigval": float(m["hessian"]["max_eigvals"][t]),
            "sharpness_ratio":    float(m["hessian"]["max_eigvals"][t]) * out["lr"] / 2.0,
            "nfa/cosine":         float(m["nfa_correlation"]["matrix_correlation"][t]),
            "nfa/mean_alignment": float(jnp.stack(m["nfa_eigen_alignment"]["matched"])[t].mean()),
            "sparsity/layer1":    float(m["sparsity"]["value"][t][0].mean()),
            "neuron_cov/max_eigval":     float(m["neuron_cov"]["layer1_max_eigvals"][t]),
            "neuron_cov/intrinsic_dim":  float(m["neuron_cov_int_dim"]["layer1_intrinsic_dim"][t]),
            "neuron_cov/stable_rank":    float(m["neuron_cov_int_dim"]["layer1_stable_rank"][t]),
        }
        for i in range(r):   # per-direction, so the dashboard can grid them
            row[f"nfa/alignment_dir{i}"] = float(jnp.stack(m["nfa_eigen_alignment"]["matched"])[t, i])
        wandb.log(row, step=t)
    run.finish()
    return run
