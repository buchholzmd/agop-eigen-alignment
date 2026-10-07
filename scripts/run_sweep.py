#!/usr/bin/env python
"""Run a slice of the sweep. One SLURM array task = one chunk of CELLS.

A "cell" is everything except the learning rate: (parameterization, width, d, r, n).
Within a cell the learning rates are scanned in ASCENDING order and the scan stops
after `lr_scan.stop_after_diverged` consecutive diverged runs -- past the stability
threshold every larger lr diverges identically, so running them buys nothing.

    python scripts/run_sweep.py --config full_sweep.yaml --chunk 0 --nchunks 64
"""
import argparse, json, os, pathlib, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
jax.config.update("jax_default_matmul_precision", "highest")

from config import load_config, expand, flatten, get_in
from sweep_utils import expand_dr, lr_grid, feasible, is_diverged


def descriptive_name(cfg, lr):
    """W&B run title carrying everything that identifies the point in the grid."""
    return (f"lr={lr:.5g}__param={get_in(cfg,'model.parameterization')}"
            f"__m={get_in(cfg,'model.width')}__d={get_in(cfg,'data.d')}"
            f"__r={get_in(cfg,'data.r')}__n={get_in(cfg,'data.n')}")


def cell_id(cfg):
    return (f"param={get_in(cfg,'model.parameterization')}__m={get_in(cfg,'model.width')}"
            f"__d={get_in(cfg,'data.d')}__r={get_in(cfg,'data.r')}__n={get_in(cfg,'data.n')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=1)
    ap.add_argument("--outdir", default=None, help="overrides paths.outputs from the config")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list work, run nothing")
    a = ap.parse_args()

    base = load_config(a.config)
    outdir = a.outdir or os.path.join(ROOT, get_in(base, "paths.outputs"))
    cells = [expand_dr(c) for c in expand(base)]
    cells = [c for c in cells if feasible(c)[0]]
    mine = cells[a.chunk::a.nchunks]            # strided: spreads the slow cells across tasks
    lrs = lr_grid(base)
    stop_after = base.get("lr_scan", {}).get("stop_after_diverged", 2)

    print(f"{a.config}: {len(cells)} feasible cells, chunk {a.chunk}/{a.nchunks} -> {len(mine)}",
          flush=True)
    print(f"lr scan: {len(lrs)} values ascending, stop after {stop_after} consecutive diverged",
          flush=True)
    print(f"jax {jax.__version__} on {jax.devices()}", flush=True)
    if a.dry_run:
        for c in mine: print("  ", cell_id(c))
        return

    from experiment import run_one

    root = pathlib.Path(outdir) / base.get("name", "sweep")
    for ci, cfg in enumerate(mine):
        cid = cell_id(cfg)
        cdir = root / cid
        cdir.mkdir(parents=True, exist_ok=True)
        if (cdir / "cell_done.json").exists():
            print(f"[cell {ci+1}/{len(mine)}] skip (done) {cid}", flush=True)
            continue

        print(f"[cell {ci+1}/{len(mine)}] {cid}", flush=True)
        consecutive, executed, t0 = 0, 0, time.time()
        for lr in lrs:
            one = json.loads(json.dumps(cfg))        # deep copy, no shared state
            one["optim"]["lr"] = float(lr)
            one["optim"]["lr_mode"] = "absolute"
            one["_wandb_name"] = descriptive_name(cfg, lr)
            tag = one["_wandb_name"]
            rdir = cdir / f"lr={lr:.5g}"
            if (rdir / "done.json").exists():
                continue                              # resume is free

            try:
                out = run_one(one, log_wandb=not a.no_wandb)
            except Exception:                         # one bad point must not kill the task
                print(f"    FAILED {tag}", flush=True)
                traceback.print_exc()
                rdir.mkdir(parents=True, exist_ok=True)
                (rdir / "failed.txt").write_text(traceback.format_exc())
                continue

            executed += 1
            div = is_diverged(out["metrics"])
            rdir.mkdir(parents=True, exist_ok=True)
            (rdir / "done.json").write_text(json.dumps({
                "lr": float(lr), "eta_crit": float(out["eta_crit"]),
                "final_loss": float(out["metrics"]["loss"][-1]), "diverged": bool(div)}))
            print(f"    lr={lr:.5g} final={float(out['metrics']['loss'][-1]):.4g} "
                  f"{'DIVERGED' if div else ''}", flush=True)

            consecutive = consecutive + 1 if div else 0
            if consecutive >= stop_after:
                print(f"    -> stopping cell at lr={lr:.5g}: {consecutive} consecutive diverged",
                      flush=True)
                break

        (cdir / "cell_done.json").write_text(json.dumps({
            "executed": executed, "of": len(lrs), "seconds": round(time.time() - t0, 1)}))
        print(f"    cell done: {executed}/{len(lrs)} lrs in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
