"""Grid expansion, feasibility filtering and the lr early-exit scan."""
import json
import math


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
