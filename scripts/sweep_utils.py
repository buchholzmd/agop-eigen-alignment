"""Grid expansion, feasibility filtering, the lr early-exit scan, and W&B sweeps."""
import copy
import json
import math

from config import get_in, set_in

# W&B sweep values must be scalars, so an axis whose values are not (target.terms
# is a list of dicts) is swept by INDEX, under its own name so the index never
# overwrites the real value that the run config logs under the axis name.
INDEX_SUFFIX = "_index"


def _is_scalar(v):
    return v is None or isinstance(v, (bool, int, float, str))


def wandb_sweep_config(cfg, config_name):
    """W&B grid-sweep config for `cfg`: one parameter per `sweep` axis, plus optim.lr.

    The lr values are the config's lr_scan grid, or its `sweep.optim.lr` list when it
    sweeps lr directly. W&B's grid has no early exit and no constraints, so it serves
    every combination, feasible or not; scripts/trial.py closes infeasible ones at once.
    """
    axes = dict(cfg.get("sweep") or {})
    lrs = axes.pop("optim.lr", None)
    if lrs is None:
        lrs = lr_grid(cfg)
    if not isinstance(lrs, list):
        raise ValueError(f"{config_name}: sweep.optim.lr must be a list, got {lrs!r}")

    params = {}
    for key, values in axes.items():
        if all(_is_scalar(v) for v in values):
            params[key] = {"values": list(values)}
        else:
            params[key + INDEX_SUFFIX] = {"values": list(range(len(values)))}
    params["optim.lr"] = {"values": [float(v) for v in lrs]}

    return {
        "name": cfg.get("name", "sweep"),
        "method": "grid",
        "metric": dict(get_in(cfg, "wandb.sweep_metric")),
        "program": "scripts/trial.py",
        "command": ["${env}", "${interpreter}", "${program}",
                    "--config", config_name, "--params", "${args_json_file}"],
        "parameters": params,
    }


def grid_size(sweep_config):
    return math.prod(len(p["values"]) for p in sweep_config["parameters"].values())


def trial_config(base, params):
    """The inverse of wandb_sweep_config: the full config for one (cell, lr) trial.

    params is the flat {parameter: value} dict W&B hands the trial. Values are looked
    up in the config's own lists, so each keeps its YAML type rather than whatever
    the JSON round-trip made of it.
    """
    cfg = copy.deepcopy(base)
    sweep = cfg.pop("sweep", None) or {}
    axes, index = {}, {}
    for key, values in sweep.items():
        if key == "optim.lr":
            continue
        if key in params:
            i = values.index(params[key])
        elif key + INDEX_SUFFIX in params:
            i = int(params[key + INDEX_SUFFIX])
        else:
            raise KeyError(f"no sweep parameter for axis {key!r}; was the config "
                           f"edited after the sweep was created?")
        set_in(cfg, key, copy.deepcopy(values[i]))
        axes[key], index[key] = values[i], i
    set_in(cfg, "optim.lr", float(params["optim.lr"]))
    cfg["_sweep_axes"], cfg["_sweep_index"] = axes, index
    return expand_dr(cfg)


def expand_dr(cfg):
    """`data.dr: "512,32"` -> data.d=512, data.r=32. Keeps r <= d by construction."""
    dr = cfg.get("data", {}).pop("dr", None)
    if dr is not None:
        d, r = (int(x) for x in str(dr).split(","))
        cfg["data"]["d"], cfg["data"]["r"] = d, r
    return cfg


def lr_grid(cfg):
    s = cfg.get("lr_scan", {})
    lo, hi, num = s.get("lo", 1e-4), s.get("hi", 1.0), s.get("num", 21)
    if s.get("values", "logspace") == "linspace":
        return [lo + (hi - lo) * i / (num - 1) for i in range(num)]
    # log spacing puts the resolution where the regime transitions are: a linear grid
    # over [1e-4, 1] spends ~97% of its points in the diverged regime.
    lg = math.log10
    return [10 ** (lg(lo) + (lg(hi) - lg(lo)) * i / (num - 1)) for i in range(num)]


def feasible(cfg):
    """-> (ok, reason). Checked BEFORE allocating anything."""
    L = cfg.get("limits", {})
    d, n, m = cfg["data"]["d"], cfg["data"]["n"], cfg["model"]["width"]
    P = d * m + 2 * m + 1
    if P > L.get("max_params", math.inf):
        return False, f"params {P:,} > {L['max_params']:,}"
    if n * m * 4 > L.get("max_act_bytes", math.inf):
        return False, f"activations {n*m*4/1e9:.1f}GB > {L['max_act_bytes']/1e9:.1f}GB"
    if n * m > L.get("max_n_times_m", math.inf):
        return False, f"n*m {n*m:,} > {L['max_n_times_m']:,}"
    cap = n / d                      # samples per input dimension
    if cap < L.get("min_capacity", 0):
        return False, f"n/d {cap:.3g} < {L['min_capacity']} (subspace not identifiable)"
    if cap > L.get("max_capacity", math.inf):
        return False, f"n/d {cap:.3g} > {L['max_capacity']} (past saturation)"
    return True, ""


def is_diverged(metrics, factor=100.0):
    """A run diverged if the loss went non-finite or ended far above its own minimum."""
    import numpy as np
    L = np.asarray([float(v) for v in metrics["loss"]])
    if not np.all(np.isfinite(L)):
        return True
    return float(L[-1]) > factor * float(np.nanmin(L))
