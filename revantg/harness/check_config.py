# Name: Revant Gupta
# Pack: ind-hi-fw2
"""Validate a harness config: source comments, nulls/open keys, data stages, v0 limits, size/layout divisibility.
Usage: python check_config.py CONFIG [--set k=v ...] [--world-size N]. Prints problems, then `N errors, M open`; exit 1 on errors.
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

NULLABLE = {"general.run"}  # optional label; train.py --out names the run
WHO = {"embedding.ke2_output_shape": "G4 Tue 27 Oct"}  # who fills an open key; default TSAI 31 Oct


def leaf_lines(node, prefix, out):
    """Map dotted key -> 0-based line of each leaf in a composed YAML node (flow collections are one leaf)."""
    if isinstance(node, yaml.MappingNode) and not node.flow_style:
        for k, v in node.value:
            leaf_lines(v, f"{prefix}{k.value}.", out)
    elif isinstance(node, yaml.SequenceNode) and not node.flow_style:
        for i, v in enumerate(node.value):
            leaf_lines(v, f"{prefix}{i}.", out)
    else:
        out[prefix[:-1]] = (node.start_mark.line, node.end_mark.line)


def comment_errors(path):
    """Leaves whose setting file (last in the extends chain) has no inline `#` comment on that line."""
    owner = {}  # dotted key -> (file, lines, start, end)
    for f, _ in config.chain(path):
        with open(f) as fh:
            text = fh.read()
        leaves = {}
        leaf_lines(yaml.compose(text), "", leaves)
        lines = text.splitlines()
        for key, (a, b) in leaves.items():
            # lists replace wholesale on merge: drop earlier files' entries under a list (or leaf) this file sets
            parts = key.split(".")
            cut = next((i for i, p in enumerate(parts) if p.isdigit()), len(parts))
            stale = ".".join(parts[:cut]) + "."
            for k in [k for k in owner if k.startswith(stale) and owner[k][0] != f]:
                del owner[k]
            owner[key] = (f, lines, a, b)
    errs = []
    for key, (f, lines, a, b) in owner.items():
        if not any(" #" in lines[i] for i in {a, b} if i < len(lines)):
            errs.append(f"error: {key} has no '# source' comment ({os.path.relpath(f)}:{a + 1})")
    return errs


def walk(d, prefix=""):
    """Yield (dotted key, value) for dict leaves; lists are leaves."""
    for k, v in d.items():
        if isinstance(v, dict):
            yield from walk(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def check(cfg, path, world_size=None):
    """Return (errors, opens) as lists of message lines."""
    errs, opens = comment_errors(path), []
    g = lambda k: config.get(cfg, k)  # noqa: E731
    open_keys = set(cfg.get("open_keys") or [])
    moe_off = not g("moe.enabled")
    for key, v in walk(cfg):
        if v is None and not (moe_off and key.startswith("moe.")):  # dense arm: moe sizes unused
            if key in open_keys:
                opens.append(f"open: {key} ({WHO.get(key, 'TSAI 31 Oct')})")
            elif key not in NULLABLE:
                errs.append(f"error: {key} is null and not in open_keys")

    names = [d["name"] for d in cfg.get("datasets") or []]
    prev = None
    for i, st in enumerate(cfg.get("data_stages") or []):
        sf, w = st.get("start_frac"), st.get("weights") or {}
        if (i == 0 and sf != 0) or (prev is not None and not sf > prev):
            errs.append(f"error: data_stages[{i}].start_frac {sf} not ascending from 0.0")
        prev = sf
        if abs(sum(w.values()) - 1) > 1e-6:
            errs.append(f"error: data_stages[{i}] ({st.get('name')}) weights sum to {sum(w.values()):.6g}, not 1")
        for k in w:
            if k not in names:
                errs.append(f"error: data_stages[{i}] weight key {k!r} not in datasets")

    if g("moe.enabled"):
        E, k = g("moe.num_experts"), g("moe.top_k")
        if E is not None and E < 2:
            errs.append(f"error: moe.num_experts {E} < 2")
        if k is not None and not 1 <= k <= (E or k):
            errs.append(f"error: moe.top_k {k} not in [1, num_experts={E}]")
        if g("parallelism.tp") != 1:
            errs.append("error: MoE + TP not in v0")
        if g("parallelism.expert_parallel_size") != 1:
            errs.append("error: MoE expert_parallel_size != 1 not in v0")
    if g("precision") not in {"fp32"}:
        errs.append(f"error: precision {g('precision')!r} not in {{fp32}} (T4 has no bf16)")

    dp, tp, pp = g("parallelism.dp"), g("parallelism.tp"), g("parallelism.pp")
    L, d, h, kv, ff = (g(f"model.{k}") for k in ("n_layers", "d_model", "n_heads", "n_kv_heads", "d_ff"))
    G, m, nmb = g("tokens.global_batch_seqs"), g("tokens.micro_batch_seqs"), g("parallelism.pp_microbatches")
    if world_size is not None and dp * tp * pp != world_size:
        errs.append(f"error: dp*tp*pp = {dp * tp * pp} != --world-size {world_size}")
    if sum(x > 1 for x in (dp, tp, pp)) > 1:
        errs.append(f"error: 2D parallel layouts not in v0 (2 GPUs): dp={dp} tp={tp} pp={pp}")
    rules = [
        ("d_model % n_heads", d, h), ("n_heads % n_kv_heads", h, kv), ("n_heads % tp", h, tp),
        ("n_kv_heads % tp", kv, tp), ("d_ff % tp", ff, tp), ("n_layers % pp", L, pp),
        ("global_batch_seqs % (dp*micro_batch_seqs)", G, m and dp * m),
    ]
    for name, a, b in rules:
        if a is not None and b and a % b:
            errs.append(f"error: {name} != 0 ({a} % {b})")
    if pp > 1 and G and m:
        if G // (dp * m) != 1:
            errs.append(f"error: pp>1 needs grad_accum == 1 (global_batch_seqs / (dp*micro_batch_seqs) = {G // (dp * m)})")
        if m % nmb:
            errs.append(f"error: pp>1 needs micro_batch_seqs % pp_microbatches == 0 ({m} % {nmb})")
    return errs, opens


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--set", action="append", default=[], metavar="K=V")
    ap.add_argument("--world-size", type=int)
    a = ap.parse_args()
    errs, opens = check(config.load_config(a.config, a.set), a.config, a.world_size)
    for line in errs + opens:
        print(line)
    print(f"{len(errs)} errors, {len(opens)} open")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
