#!/usr/bin/env python
"""One W&B sweep trial: a single (cell, lr) point, launched by `wandb agent`.

The agent passes the point's parameter values as a JSON file (the sweep's
${args_json_file}); scripts/wandb_sweep.py builds the sweep that serves them.

    python scripts/trial.py --config full_sweep.yaml --params point.json
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(HERE), "src"), HERE]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from config import load_config, get_in, run_config, run_name, group_name, run_tags
from sweep_utils import trial_config, feasible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--params", required=True, help="JSON file of this trial's sweep values")
    a = ap.parse_args()

    with open(a.params) as fh:
        params = json.load(fh)
    cfg = trial_config(load_config(a.config), params)
    cfg["_wandb_name"] = run_name(cfg)

    ok, why = feasible(cfg)
    if not ok:
        # W&B's grid cannot skip a point, and the agent has already created its run.
        # Close it out at once, before JAX loads, tagged so it can be filtered away.
        import wandb
        run = wandb.init(name=cfg["_wandb_name"], group=group_name(cfg),
                         job_type=get_in(cfg, "model.parameterization"),
                         tags=run_tags(cfg) + ["infeasible"], config=run_config(cfg),
                         mode=get_in(cfg, "wandb.mode"))
        run.summary["infeasible"] = why
        run.finish()
        print(f"infeasible, closed without training: {why}", flush=True)
        return

    import jax
    jax.config.update("jax_default_matmul_precision", "highest")
    print(f"jax {jax.__version__} on {jax.devices()}", flush=True)
    from experiment import run_one
    run_one(cfg, log_wandb=True)


if __name__ == "__main__":
    main()
