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
        one["_sweep_index"] = {k: sweep[k].index(v) for k, v in zip(keys, combo)}
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
              "_sweep_axes", "_sweep_index", "_wandb_name")


def run_config(cfg):
    """Flat hyperparameter dict for wandb.init(config=...), infrastructure stripped."""
    return flatten({k: v for k, v in cfg.items() if k not in INFRA_KEYS})


def run_name(cfg):
    """{experiment}_{axis}={value}..._lr={lr}, built from the axes this sweep varies.

        width-sweep_width=512_alpha=0_lr=1.8
        target-sweep_terms=2_alpha=1_lr=1.8

    The name is built from _sweep_axes rather than a fixed field list, so every
    varied axis appears and names are unique. A fixed list silently collides the
    moment a config sweeps something it does not mention -- sweeping target.alpha
    alone put 2 runs on every name, and target.terms put 8.

    Axes whose values are not scalars (target.terms is a list of dicts) are
    labelled by their index in the sweep list; the full value is in the logged
    config, which is what you filter on anyway.
    """
    parts = [cfg.get("name", "run")]
    axes  = cfg.get("_sweep_axes") or {}
    index = cfg.get("_sweep_index") or {}
    for dotted, value in axes.items():
        short = dotted.split(".")[-1]
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            parts.append(f"{short}={index.get(dotted, 0)}")
        elif isinstance(value, float):
            parts.append(f"{short}={value:.5g}")
        else:
            parts.append(f"{short}={value}")
    parts.append(f"lr={get_in(cfg, 'optim.lr'):.5g}")
    return "_".join(parts)


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
