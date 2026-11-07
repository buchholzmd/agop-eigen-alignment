#!/usr/bin/env python
"""Run a config's sweep as a W&B Sweep.

A *cell* is every sweep axis except the learning rate; a *point* is one (cell, lr)
pair and becomes one W&B run. W&B's grid serves every point: there is no early exit
over lr, and a point that fails the config's `limits:` checks is closed at once
without training, tagged `infeasible`.

    python scripts/run_sweep.py --config full_sweep.yaml --dry-run      # show the sweep, create nothing
    python scripts/run_sweep.py --config full_sweep.yaml --create       # create it; prints its id last
    python scripts/run_sweep.py --config full_sweep.yaml --sweep-id ID  # run points until the grid is done
    python scripts/run_sweep.py --config full_sweep.yaml --sweep-id ID --count 1

scripts/submit.sh runs --create once, then starts SLURM tasks that each run --sweep-id.
"""
import argparse, itertools, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(HERE), "src"), HERE]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import yaml

from config import load_config, get_in, run_config, run_name, group_name, run_tags
from sweep_utils import wandb_sweep_config, grid_size, point_config, feasible


def describe(base, sweep, config_name):
    """Print the sweep's size to stderr, keeping stdout free for the sweep id."""
    params = sweep["parameters"]
    lrs = params["optim.lr"]["values"]
    names = [k for k in params if k != "optim.lr"]
    cells = list(itertools.product(*(params[k]["values"] for k in names)))
    ok = sum(feasible(point_config(base, {**dict(zip(names, c)), "optim.lr": lrs[0]}))[0]
             for c in cells)
    print(f"{config_name}: {grid_size(sweep):,} points = {len(cells):,} cells x "
          f"{len(lrs)} lrs; {ok:,} cells feasible", file=sys.stderr)


def run_point(base):
    """Run the point W&B serves next: train it, or close it at once if infeasible."""
    import wandb
    # W&B hands over the point only through the run, so the run is created first and
    # named and tagged afterwards. group and job_type cannot change after init, which
    # is why job_type is the config's constant wandb.job_type, not the parameterization.
    run = wandb.init(group=group_name(base), job_type=get_in(base, "wandb.job_type"),
                     notes=get_in(base, "wandb.notes") or None, mode=get_in(base, "wandb.mode"))
    cfg = point_config(base, dict(run.config))
    ok, why = feasible(cfg)
    run.name = run_name(cfg)
    run.tags = tuple(run_tags(cfg) + ([] if ok else ["infeasible"]))
    # The sweep's own parameters are already in run.config and locked; add the rest.
    run.config.update({k: v for k, v in run_config(cfg).items() if k not in run.config})
    if not ok:
        run.summary["infeasible"] = why
        print(f"infeasible, closed without training: {why}", flush=True)
        return

    import jax
    from experiment import run_one
    try:
        run_one(cfg, run=run)
    finally:
        jax.clear_caches()      # one process runs many points; drop their compiled code


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="print the sweep config, create nothing")
    mode.add_argument("--create", action="store_true", help="create the W&B sweep, print its id")
    mode.add_argument("--sweep-id", help="run this sweep's points until the grid is done")
    ap.add_argument("--count", type=int, default=None, help="with --sweep-id: stop after this many points")
    a = ap.parse_args()

    base = load_config(a.config)
    entity, project = get_in(base, "wandb.entity"), get_in(base, "wandb.project")

    if a.sweep_id:
        import jax
        jax.config.update("jax_default_matmul_precision", "highest")
        print(f"jax {jax.__version__} on {jax.devices()}", flush=True)
        import wandb
        wandb.agent(a.sweep_id, function=lambda: run_point(base),
                    entity=entity, project=project, count=a.count)
        return

    sweep = wandb_sweep_config(base, a.config)
    describe(base, sweep, a.config)
    if a.dry_run:
        print(yaml.safe_dump(sweep, sort_keys=False), file=sys.stderr)
        return
    import wandb
    print(wandb.sweep(sweep, entity=entity, project=project))


if __name__ == "__main__":
    main()
