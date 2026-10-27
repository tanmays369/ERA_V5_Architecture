# Name: dharshan
# Axiom email: dharshanthaigu@gmail.com
# Pack: ind-ta-fw2
"""Check a config: every setting needs a value and a source.

Usage:
    python harness/check_config.py configs/config_v0.yaml [--list]

Each line it prints starts with one of:
    no value:   the setting is empty or missing its value
    no source:  the setting has no "# [tag] reason" comment, or the tag is not one of KE, paper, run, admin, ours, open
    missing:    a key from SCHEMA (harness/config.py) is not in the file
    unknown:    a key in the file is not in SCHEMA
    bad value:  the value breaks a rule (for example n_heads does not divide d_model)
    header:     the first three lines are not "# Name:", "# Axiom email:", "# Pack:"
    open:       allowed in config v0: the value is "open" and the comment says who gives it and when
At the end it prints the counts and how many settings use each source tag. Exit code 1 if there is any error.
A [run] tag must start with a path (relative to the folder that holds configs/ and evidence/) of a file that exists in evidence/.
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OPEN, SCHEMA, TAGS, flatten, source_comments  # noqa: E402

import yaml  # noqa: E402

TAG = re.compile(r"^\[(\w+)\]\s*(.*)$")
ENUMS = {
    "model.norm_position": {"pre", "post"},
    "moe.router_sharing": {"none", "all_layers"},
    "embedding.mode": {"dynamic", "cached"},
    "embedding.type": {"kronecker"},
    "optimizer.matrices_2d": {"muon", "adamw"},
    "optimizer.other_params": {"adamw"},
    "model.ffn": {"swiglu"}, "model.norm": {"rmsnorm"}, "moe.router_scoring": {"sigmoid"},
    "moe.router_dtype": {"float32"}, "moe.balancing": {"aux_free_bias"}, "schedule.type": {"wsd"},
    "precision.dtype": {"float32"}, "data.source": {"synthetic", "parquet"},
}
INT_MIN = {  # key -> minimum integer value
    "general.seed": 0, "general.num_threads": 1, "model.n_layers": 1, "model.d_model": 1, "model.n_heads": 1,
    "model.n_kv_heads": 1, "model.d_ff": 1, "attention.window": 0, "attention.global_every": 0,
    "moe.num_experts": 1, "moe.top_k": 1, "moe.shared_experts": 0, "head.mtp_heads": 0,
    "optimizer.muon_ns_steps": 1, "tokens.seq_len": 2, "tokens.global_batch_seqs": 1,
    "tokens.micro_batch_seqs": 1, "tokens.train_steps": 1, "logging.eval_every": 1,
    "data.val_windows_per_source": 1, "tokenizer.vocab_size": 1, "tokenizer.eos_id": 0,
}
BOOLS = ["embedding.length_normalize", "embedding.z_normalize", "model.qk_norm", "moe.enabled",
         "general.deterministic", "safety.stop_on_nan"]
FRACTIONS = ["schedule.warmup_frac", "schedule.decay_frac", "schedule.min_lr_frac", "data.val_frac"]


def check(path, show_all=False):
    path = Path(path).resolve()
    root = path.parent.parent                    # the folder that holds configs/ and evidence/
    tree = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    flat = flatten(tree)
    comments, head = source_comments(path)
    errors, opens, ok_lines, tags = [], [], [], Counter()

    want = ["# Name:", "# Axiom email:", "# Pack:"]
    for i, w in enumerate(want):
        if i >= len(head) or not head[i].startswith(w):
            errors.append(f"header: line {i + 1} must start with {w!r}")
        elif "<FILL" in head[i]:
            errors.append(f"header: line {i + 1} still has a <FILL ...> placeholder")

    for key in SCHEMA:
        if key not in flat:
            errors.append(f"missing: {key}")
    for key in flat:
        if key not in SCHEMA:
            errors.append(f"unknown: {key}")

    for key in SCHEMA:
        if key not in flat:
            continue
        v = flat[key]
        if v is None or (isinstance(v, str) and not v.strip()):
            errors.append(f"no value: {key}")
            continue
        comment = comments.get(key)
        m = TAG.match(comment) if comment else None
        if not m or m.group(1) not in TAGS or len(m.group(2).strip()) < 3:
            errors.append(f"no source: {key}")
            continue
        tag, text = m.group(1), m.group(2).strip()
        tags[tag] += 1
        if tag == "run":
            ev = text.split()[0]
            if not (root / ev).is_file():
                errors.append(f"no source: {key} ([run] evidence file {ev!r} does not exist)")
                continue
        if v == OPEN:
            if tag != "open" or "," not in text:
                errors.append(f"no source: {key} (an open value needs '[open] who, when')")
            else:
                opens.append(f"open: {key} ({text})")
            continue
        if tag == "open":
            errors.append(f"no source: {key} (tag [open] on a setting that has a value)")
            continue
        ok_lines.append(f"ok: {key} = {v}   {comment}")

    def val(k):
        v = flat.get(k)
        return None if v == OPEN else v

    for k, allowed in ENUMS.items():
        if val(k) is not None and val(k) not in allowed:
            errors.append(f"bad value: {k} = {val(k)!r} (allowed: {sorted(allowed)})")
    for k, lo in INT_MIN.items():
        v = val(k)
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < lo):
            errors.append(f"bad value: {k} = {v!r} (must be an integer >= {lo})")
    for k in BOOLS:
        if val(k) is not None and not isinstance(val(k), bool):
            errors.append(f"bad value: {k} = {val(k)!r} (must be true or false)")
    for k in FRACTIONS:
        v = val(k)
        if v is not None and (not isinstance(v, (int, float)) or not 0 <= v < 1):
            errors.append(f"bad value: {k} = {v!r} (must be a number in [0, 1))")
    d, h, kv = val("model.d_model"), val("model.n_heads"), val("model.n_kv_heads")
    if d and h and d % h:
        errors.append(f"bad value: model.d_model {d} is not divisible by model.n_heads {h}")
    if h and kv and h % kv:
        errors.append(f"bad value: model.n_heads {h} is not divisible by model.n_kv_heads {kv}")
    if d and h and (d // h) % 2:
        errors.append("bad value: head size (d_model / n_heads) must be even for RoPE")
    ne, tk = val("moe.num_experts"), val("moe.top_k")
    if ne and tk and tk > ne:
        errors.append(f"bad value: moe.top_k {tk} is larger than moe.num_experts {ne}")
    g, mb = val("tokens.global_batch_seqs"), val("tokens.micro_batch_seqs")
    if g and mb and g % mb:
        errors.append(f"bad value: tokens.global_batch_seqs {g} is not divisible by tokens.micro_batch_seqs {mb}")
    if val("head.mtp_heads") not in (None, 0, 1):
        errors.append("bad value: head.mtp_heads must be 0 or 1")
    b = val("optimizer.betas")
    if b is not None and not (isinstance(b, list) and len(b) == 2):
        errors.append("bad value: optimizer.betas must be a list of two numbers")
    if val("data.source") == "parquet" and not val("data.paths"):
        errors.append("bad value: data.source is parquet but data.paths is empty")
    if val("moe.enabled") is False and (val("moe.router_sharing") not in (None, "none")):
        errors.append("bad value: moe.router_sharing needs moe.enabled true")

    return errors, opens, ok_lines, tags


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config")
    ap.add_argument("--list", action="store_true", help="also print every setting that is fine")
    a = ap.parse_args()
    errors, opens, ok_lines, tags = check(a.config)
    for line in errors + opens:
        print(line)
    if a.list:
        for line in ok_lines:
            print(line)
    print(f"{len(ok_lines)} ok, {len(opens)} open, {len(errors)} errors")
    print("source tags: " + ", ".join(f"{t}={tags[t]}" for t in TAGS if tags[t]))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
