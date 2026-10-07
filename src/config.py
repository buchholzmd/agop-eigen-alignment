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


def run_name(cfg):
    """Readable, sortable run name built from whatever this sweep varies.

    nfa_phase__parameterization=mup__lr=1.20__d=3__m=1000__seed=33

    Without this every run gets a random W&B slug and the dashboard is unusable.
    """
    parts = [cfg.get("name", "run")]
    for dotted, val in (cfg.get("_sweep_axes") or {}).items():
        short = dotted.split(".")[-1]
        parts.append(f"{short}={val:.2f}" if isinstance(val, float) else f"{short}={val}")
    parts += [
        f"d={get_in(cfg, 'data.d')}",
        f"m={get_in(cfg, 'model.width')}",
        f"seed={get_in(cfg, 'data.seed')}",
    ]
    return "__".join(parts)


def group_name(cfg):
    """All configs from one sweep file share this, so W&B groups them together."""
    return cfg.get("name", "run")
