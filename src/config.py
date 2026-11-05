"""Experiment configuration: load YAML, merge overrides, expand sweeps, name runs.

A config is a plain nested dict. Dotted keys ("optim.lr") address nested values, so
sweep files and CLI overrides stay flat and readable.
"""
import copy
import itertools
import os

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def _deep_merge(base, over):
    """Recursively merge `over` into a copy of `base`."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def get_in(cfg, dotted):
    """get_in(cfg, 'optim.lr') -> cfg['optim']['lr']"""
    node = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


def set_in(cfg, dotted, value):
    """set_in(cfg, 'optim.lr', 0.5), in place. Returns cfg."""
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
    return cfg


def load_config(path, **overrides):
    """Load a YAML config, following `extends`, then apply dotted-key overrides.

        cfg = load_config("lr_sweep.yaml", **{"model.width": 2048})

    `sweep` is left in the returned dict; call expand() to turn it into a list of configs.
    """
    if not os.path.isabs(path):
        path = os.path.join(CONFIG_DIR, path)
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    parent = cfg.pop("extends", None)
    if parent is not None:
        cfg = _deep_merge(load_config(parent), cfg)
    for k, v in overrides.items():
        set_in(cfg, k, v)
    return cfg


def expand(cfg):
    """Expand a `sweep` block into one config per point of the Cartesian product.

    A config with no `sweep` key expands to [cfg], so callers never branch.
    """
    sweep = cfg.get("sweep")
    if not sweep:
        return [copy.deepcopy(cfg)]
    keys = list(sweep)
    out = []
    for combo in itertools.product(*(sweep[k] for k in keys)):
        one = copy.deepcopy(cfg)
        one.pop("sweep", None)
        for k, v in zip(keys, combo):
            set_in(one, k, v)
        one["_sweep_axes"] = dict(zip(keys, combo))
        out.append(one)
    return out


def flatten(cfg, prefix=""):
    """Flatten to dotted keys, for logging every value to W&B as a filterable column."""
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


# Infrastructure, not experiment. W&B filters runs on config values, so the
# config should hold exactly what distinguishes one run from another -- not the
# SLURM account, the cache path, or the sweep definition that generated it.
INFRA_KEYS = ("slurm", "paths", "wandb", "limits", "lr_scan", "sweep",
              "_sweep_axes", "_wandb_name")


def run_config(cfg):
    """Flat hyperparameter dict for wandb.init(config=...), infrastructure stripped."""
    return flatten({k: v for k, v in cfg.items() if k not in INFRA_KEYS})


def run_name(cfg):
    """{experiment}_{parameterization}_m={width}_lr={lr}

        subspace-recovery_spectral_m=512_lr=1.8

    The experiment prefix is `name:` from the YAML. Everything in the name is
    also in the config and filterable there -- the name exists so a run is
    identifiable at a glance in the run table and in overlay legends, not as a
    search key. Under optim.lr_mode: relative the lr shown is the multiple of
    eta_crit, which is the quantity the grid is defined in; the absolute value
    is logged as config.lr_absolute.
    """
    return "_".join([
        cfg.get("name", "run"),
        str(get_in(cfg, "model.parameterization")),
        f"m={get_in(cfg, 'model.width')}",
        f"lr={get_in(cfg, 'optim.lr'):.5g}",
    ])


def group_name(cfg):
    """All runs from one sweep file share this, so W&B groups them together."""
    return cfg.get("name", "run")


def run_tags(cfg):
    """Coarse facets. W&B tag filtering is first-class and cheaper than a config query."""
    return [
        get_in(cfg, "model.parameterization"),
        f"m{get_in(cfg, 'model.width')}",
        f"d{get_in(cfg, 'data.d')}",
        f"r{get_in(cfg, 'data.r')}",
        f"n{get_in(cfg, 'data.n')}",
    ]
