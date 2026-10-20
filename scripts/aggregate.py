"""Aggregate a finished sweep into videos and regime plots.

For each (parameterization, width) cell:
  * an mp4 stepping through the learning-rate grid, six panels
  * a three-panel split of the losses into monotonic / catapult / diverged

    python scripts/aggregate.py --config subspace_sweep.yaml
"""
import argparse
import os
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, FFMpegWriter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]

from config import load_config, get_in

# (metric key, axis label, log scale)
PANELS = [
    ("loss",                       "loss",                    True),
    ("nfm/cosine_to_agop",         "NFA cosine",              False),
    ("subspace/nfm_mean_alignment","subspace alignment",      False),
    ("spectrum/agop_nuclear",      "AGOP nuclear norm",       True),
    ("neuron/sparsity_layer1",     "neuron sparsity",         False),
    ("geometry/sharpness_ratio",   "sharpness / (2/lr)",      True),
]


def load_runs(entity, project, group, cache_dir):
    """Pull every run in `group`, caching full histories to parquet.

    scan_history, not history: history is downsampled by default and would smooth
    away the catapult spikes this whole sweep exists to measure.
    """
    import wandb
    import pandas as pd

    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for run in wandb.Api().runs(f"{entity}/{project}", filters={"group": group}):
        cache = cache_dir / f"{run.id}.parquet"
        if cache.exists():
            history = pd.read_parquet(cache)
        else:
            history = pd.DataFrame(list(run.scan_history()))
            if len(history):
                history.to_parquet(cache)
        runs.append((dict(run.config), history, run.name))
    return runs


def classify_regime(loss, blowup=100.0, tol=1e-9):
    """monotonic | catapult | diverged.

    A catapult needs both a rise AND a recovery below the pre-rise loss. Without
    the recovery condition every diverging run looks like a catapult.
    """
    loss = np.asarray(loss, dtype=float)
    if not np.all(np.isfinite(loss)) or loss[-1] > blowup * np.nanmin(loss):
        return "diverged"
    rises = np.where(np.diff(loss) > tol * np.maximum(1.0, np.abs(loss[:-1])))[0]
    if len(rises) and loss[-1] < loss[rises[0]]:
        return "catapult"
    return "monotonic"


def group_by_cell(runs):
    """-> {(parameterization, width): [(lr, config, history, name), ...]} sorted by lr."""
    cells = {}
    for config, history, name in runs:
        key = (config.get("model.parameterization"), config.get("model.width"))
        cells.setdefault(key, []).append(
            (config.get("lr_absolute"), config, history, name))
    for key in cells:
        cells[key].sort(key=lambda row: (row[0] is None, row[0]))
    return cells


def panel_limits(cell):
    """Fixed y-limits per panel so frames of the video are comparable."""
    limits = {}
    for key, _, _ in PANELS:
        values = [h[key].dropna().values for _, _, h, _ in cell if key in h]
        values = np.concatenate(values) if values else np.array([])
        values = values[np.isfinite(values) & (values > 0)]
        limits[key] = (values.min() * 0.8, values.max() * 1.2) if len(values) else (1e-3, 1.0)
    return limits


def make_video(key, cell, outdir, fps=4):
    parameterization, width = key
    limits = panel_limits(cell)
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5))
    axs = axs.ravel()

    def draw(index):
        lr, config, history, name = cell[index]
        for ax, (metric, label, logy) in zip(axs, PANELS):
            ax.clear()
            if metric in history:
                ax.plot(history[metric].dropna().values, lw=1.6)
            if metric == "geometry/sharpness_ratio":
                ax.axhline(1.0, color="k", ls="--", lw=1, label="edge of stability")
                ax.legend(fontsize=7)
            ax.set(xlabel="step", ylabel=label)
            if logy:
                ax.set_yscale("log")
            ax.set_ylim(*limits[metric])
        fig.suptitle(f"{parameterization}   width={width}   lr={lr:.5g}"
                     f"   ({index + 1}/{len(cell)})", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    anim = FuncAnimation(fig, draw, frames=len(cell), interval=1000 // fps)
    path = pathlib.Path(outdir) / f"sweep_{parameterization}_m{width}.mp4"
    anim.save(path, writer=FFMpegWriter(fps=fps, bitrate=2400))
    plt.close(fig)
    return path


def make_regime_plot(key, cell, outdir):
    parameterization, width = key
    regimes = {"monotonic": [], "catapult": [], "diverged": []}
    for lr, config, history, name in cell:
        if "loss" not in history:
            continue
        loss = history["loss"].dropna().values
        regimes[classify_regime(loss)].append((lr, loss))

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, (regime, rows) in zip(axs, regimes.items()):
        for lr, loss in rows:
            ax.plot(loss, lw=1.0, alpha=0.8, label=f"{lr:.4g}")
        ax.set(title=f"{regime}  ({len(rows)} runs)", xlabel="step", yscale="log")
        if rows and len(rows) <= 12:
            ax.legend(fontsize=6, ncol=2)
    axs[0].set_ylabel("loss")
    fig.suptitle(f"{parameterization}   width={width}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    path = pathlib.Path(outdir) / f"regimes_{parameterization}_m{width}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--fps", type=int, default=4)
    a = ap.parse_args()

    cfg = load_config(a.config)
    outdir = a.outdir or os.path.join(ROOT, get_in(cfg, "paths.outputs"), "aggregate")
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    runs = load_runs(get_in(cfg, "wandb.entity"), get_in(cfg, "wandb.project"),
                     cfg.get("name"), os.path.join(ROOT, get_in(cfg, "paths.cache")))
    print(f"loaded {len(runs)} runs", flush=True)

    for key, cell in group_by_cell(runs).items():
        video = make_video(key, cell, outdir, a.fps)
        regimes = make_regime_plot(key, cell, outdir)
        print(f"{key}  {len(cell)} runs  ->  {video.name}, {regimes.name}", flush=True)


if __name__ == "__main__":
    main()
