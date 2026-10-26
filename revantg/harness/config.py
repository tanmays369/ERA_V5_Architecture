# Name: Revant Gupta
# Pack: ind-hi-fw2
"""Config loading: `extends` chain with deep merge (lists replaced), `--set a.b=v` overrides, derived values.
Used by every harness script: `cfg = config.load_config(path, overrides)`.
"""
import os

import yaml


def merge(base, over):
    """Deep-merge `over` into a copy of `base`; dicts merge, everything else (lists too) is replaced."""
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def chain(path):
    """Files in the extends chain, root parent first, `path` last."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    parent = raw.get("extends")
    files = chain(os.path.join(os.path.dirname(path), parent)) if parent else []
    return files + [(path, raw)]


def parse_override(s):
    """'a.b.c=VALUE' -> (['a','b','c'], yaml.safe_load(VALUE))."""
    key, sep, val = s.partition("=")
    if not sep or not key:
        raise ValueError(f"override must be key=value: {s!r}")
    return key.split("."), yaml.safe_load(val)


def load_config(path, overrides=()):
    cfg = {}
    for _, raw in chain(path):
        cfg = merge(cfg, raw)
    cfg.pop("extends", None)
    for s in overrides:
        keys, val = parse_override(s)
        keys = [int(k) if k.isdigit() else k for k in keys]  # digits index lists: data_stages.0.weights.code=0.3
        d = cfg
        for k in keys[:-1]:
            d = d[k] if isinstance(d, list) else d.setdefault(k, {})
        d[keys[-1]] = val
    return cfg


def get(cfg, key, default=None):
    """get(cfg, "a.b.c") -> nested value, or default if any part is missing."""
    for k in key.split("."):
        if not isinstance(cfg, dict) or k not in cfg:
            return default
        cfg = cfg[k]
    return cfg


def derived(cfg):
    """world_size, grad_accum, head_dim, stage_bounds; a value is None while its inputs are null."""
    dp, tp, pp = (get(cfg, f"parallelism.{k}") for k in ("dp", "tp", "pp"))
    G, m, T = (get(cfg, f"tokens.{k}") for k in ("global_batch_seqs", "micro_batch_seqs", "train_steps"))
    d, h = get(cfg, "model.d_model"), get(cfg, "model.n_heads")
    return {
        "world_size": dp * tp * pp,
        "grad_accum": G // (dp * m) if G and m else None,
        "head_dim": d // h if d and h else None,
        "stage_bounds": [int(s["start_frac"] * T) for s in cfg["data_stages"]] if T else None,
    }
