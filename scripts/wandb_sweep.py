#!/usr/bin/env python
"""Create the W&B sweep for a config and print its id as the last line of stdout.

    python scripts/wandb_sweep.py --config full_sweep.yaml            # create it
    python scripts/wandb_sweep.py --config full_sweep.yaml --dry-run  # show it, create nothing

scripts/submit.sh calls this, then starts SLURM tasks that each run `wandb agent`
on the id. Each grid point the agent serves runs scripts/trial.py.
"""
import argparse, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(HERE), "src"), HERE]

import yaml

from config import load_config, get_in
from sweep_utils import wandb_sweep_config, grid_size, trial_config, feasible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="print the sweep config, create nothing")
    a = ap.parse_args()

    cfg = load_config(a.config)
    sweep = wandb_sweep_config(cfg, a.config)
    total = grid_size(sweep)
    lr = sweep["parameters"]["optim.lr"]["values"][0]
    cells = total // len(sweep["parameters"]["optim.lr"]["values"])
    feasible_cells = sum(
        feasible(trial_config(cfg, {**point, "optim.lr": lr}))[0]
        for point in _cell_points(sweep))
    # Counts go to stderr so the id stays the last line of stdout for submit.sh.
    print(f"{a.config}: {total:,} trials = {cells:,} cells x "
          f"{total // cells} lrs; {feasible_cells:,} cells feasible", file=sys.stderr)

    if a.dry_run:
        print(yaml.safe_dump(sweep, sort_keys=False), file=sys.stderr)
        return

    import wandb
    sweep_id = wandb.sweep(sweep, entity=get_in(cfg, "wandb.entity"),
                           project=get_in(cfg, "wandb.project"))
    print(sweep_id)


def _cell_points(sweep):
    """Every combination of the non-lr parameters, as {parameter: value} dicts."""
    import itertools
    names = [k for k in sweep["parameters"] if k != "optim.lr"]
    for combo in itertools.product(*(sweep["parameters"][k]["values"] for k in names)):
        yield dict(zip(names, combo))


if __name__ == "__main__":
    main()
