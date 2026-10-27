#!/usr/bin/env python3
"""sweep.py - measurements from our tiny run. Writes measurements/<id>.json, which config_v0.yaml cites as its source.

One sweep = one setting (or a few linked ones) varied over a small grid, everything else held at the config's value.
Each variant is trained with harness/train.py; the result is judged on final validation loss.

Usage (from teams/05-architecture; run prepare_data.py first):
    python harness/sweep.py --list
    python harness/sweep.py --config configs/small_cpu.yaml --only lr_sweep zloss_sweep
    python harness/sweep.py --config configs/small_cpu.yaml --all                 # about 30 short runs, about 15-40 min on CPU
    python harness/sweep.py --config configs/small_cpu.yaml --all --seeds 1234 1235
    python harness/sweep.py --apply configs/config_v0.yaml configs/small_cpu.yaml configs/small_colab.yaml
        # copies each measurement's chosen value into the config lines that cite it, then run check_config.py

Choice rule (written into every JSON): keep the config's current value unless another variant is healthy and beats it
by more than TOL nats of validation loss. "Healthy" = the biggest expert takes at most 2/num_experts of the load.
This keeps defaults from flipping on noise. With one seed the rule cannot tell noise from signal: use --seeds.
"""
import argparse
import copy
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402

ROOT = train.ROOT
TOL = 0.02


def grid(key, values):
    return [{key: v} for v in values]


SWEEPS = {
    "lr_sweep": grid("train.lr", (3e-4, 1e-3, 3e-3)),
    "decay_sweep": grid("train.min_lr_ratio", (0.0, 0.1, 0.3)),
    "wd_sweep": grid("train.weight_decay", (0.0, 0.1)),
    "clip_sweep": grid("train.grad_clip", (0.5, 1.0, 2.0)),
    "init_sweep": grid("model.init_std", (0.01, 0.02, 0.05)),
    "topk_sweep": grid("moe.top_k", (1, 2)),
    "shared_sweep": grid("moe.num_shared_experts", (0, 1)),
    "router_sweep": grid("moe.router_type", ("softmax", "sigmoid")),
    "normalize_sweep": grid("moe.normalize_topk_weights", (True, False)),
    "zloss_sweep": grid("moe.router_z_loss_coef", (0.0, 1e-3, 1e-2)),
    "mtp_sweep": grid("mtp.mtp_depth", (0, 1, 2)),
    "layout_sweep": [{"moe.first_k_dense": 0, "moe.moe_layer_freq": 1}, {"moe.first_k_dense": 1, "moe.moe_layer_freq": 1},
                     {"moe.first_k_dense": 0, "moe.moe_layer_freq": 2}],
    "balance_sweep": [{"moe.balance_method": "aux_loss", "moe.load_balance_coef": 0.001},
                      {"moe.balance_method": "aux_loss", "moe.load_balance_coef": 0.01},
                      {"moe.balance_method": "aux_loss", "moe.load_balance_coef": 0.1},
                      {"moe.balance_method": "bias"}, {"moe.balance_method": "none"}],
}


def get(cfg, dotted):
    s, k = dotted.split(".")
    return cfg[s][k]


def set_(cfg, dotted, v):
    s, k = dotted.split(".")
    cfg[s][k] = v


def label(d):
    return ", ".join(f"{k.split('.')[1]}={v}" for k, v in d.items())


def variant_list(sid, base):
    keys = sorted({k for v in SWEEPS[sid] for k in v})
    base_vals = {k: get(base, k) for k in keys}
    variants = [{**base_vals, **v} for v in SWEEPS[sid]]
    if base_vals not in variants:
        variants.append(dict(base_vals))
    return base_vals, variants


def run_path(base, diff, seed, outdir):
    """Finished runs are stored under a name that includes a hash of the exact config, so a stopped sweep can resume."""
    cfg = copy.deepcopy(base)
    for k, v in diff.items():
        set_(cfg, k, v)
    h = hashlib.md5((json.dumps(cfg, sort_keys=True, default=str) + f"|{seed}").encode()).hexdigest()[:8]
    tag = re.sub(r"[^A-Za-z0-9.=_-]+", "_", label(diff) or "base")
    return cfg, outdir / f"{tag}_{h}_s{seed}.csv"


def plan_runs(sids, base, seeds, outdir):
    """Unique (variant, seed) runs the chosen sweeps need, and how many are already on disk."""
    need, done = set(), set()
    for sid in sids:
        base_vals, variants = variant_list(sid, base)
        for full in variants:
            diff = {k: v for k, v in full.items() if v != base_vals[k]}
            for seed in seeds:
                cfg, path = run_path(base, diff, seed, outdir)
                try:
                    train.validate(cfg)
                except AssertionError:
                    continue
                need.add(str(path))
                if path.with_suffix(".summary.json").exists():
                    done.add(str(path))
    return need, done


def run_variant(base, diff, seeds, device, tok, data, steps, cache, outdir, prog=None):
    """Average of final summaries over seeds for the base config with `diff` applied. Cached per (diff, seed)."""
    rows = []
    for seed in seeds:
        key = (json.dumps(sorted(diff.items())), seed)
        if key not in cache:
            cfg, path = run_path(base, diff, seed, outdir)
            train.validate(cfg)
            summ = path.with_suffix(".summary.json")
            if summ.exists():
                cache[key] = json.loads(summ.read_text())
                print(f"    reused finished run: {label(diff) or 'base'} seed {seed}", flush=True)
            else:
                if prog:
                    print(f"    running [{prog['done'] + 1}/{prog['total']}] {label(diff) or 'base'} seed {seed} ...", flush=True)
                t0 = time.time()
                cache[key] = train.run_training(cfg, seed, device, path, tok=tok, data=data, quiet=True)
                if prog:
                    prog["done"] += 1
                    prog["secs"].append(time.time() - t0)
                    left = prog["total"] - prog["done"]
                    avg = sum(prog["secs"]) / len(prog["secs"])
                    print(f"      finished in {time.time() - t0:.0f}s ({cache[key]['tokens_per_second']:,.0f} tok/s); "
                          f"{left} runs left, about {left * avg / 60:.0f} min", flush=True)
        rows.append(cache[key])
    mean = lambda k: None if rows[0].get(k) is None else float(np.mean([r[k] for r in rows]))
    return {"val_loss": mean("val_loss"), "val_loss_std": float(np.std([r["val_loss"] for r in rows])) if len(rows) > 1 else None,
            "loss_ntp_last": mean("loss_ntp_last"), "loss_ntp_step1": mean("loss_ntp_step1"), "ln_vocab": rows[0]["ln_vocab"],
            "router_entropy": mean("router_entropy"), "expert_load_max_frac": mean("expert_load_max_frac"),
            "grad_norm_p99": mean("grad_norm_p99"), "params_total": rows[0]["params_total"],
            "params_active": rows[0]["params_active"], "tokens_per_second": mean("tokens_per_second"),
            "num_experts": rows[0]["num_experts"], "ok": all(r["ok"] for r in rows)}


def do_sweep(sid, base, args, tok, data, cache, outdir, prog=None):
    base_vals, variants = variant_list(sid, base)
    table, E = [], base["moe"]["num_experts"]
    for full in variants:
        diff = {k: v for k, v in full.items() if v != base_vals[k]}
        try:
            r = run_variant(base, diff, args.seeds, args.device, tok, data, args.steps, cache, outdir, prog)
        except AssertionError as err:
            print(f"  skipped {label(full)}: {err}")
            continue
        healthy = r["expert_load_max_frac"] is None or r["expert_load_max_frac"] <= 2.0 / E
        table.append({"label": label(full), "is_current_default": full == base_vals, "values": full, "healthy": bool(healthy), **r})
        print(f"  {label(full):52s} val_loss {r['val_loss']:.4f}  load_max {r['expert_load_max_frac'] if r['expert_load_max_frac'] is not None else float('nan'):.3f}"
              f"  {'' if healthy else 'UNHEALTHY'}{'  <- current default' if full == base_vals else ''}")
    cur = next(t for t in table if t["is_current_default"])
    pool = [t for t in table if t["healthy"]] or table
    best = min(pool, key=lambda t: t["val_loss"])
    chosen = cur if (cur["healthy"] and cur["val_loss"] - best["val_loss"] <= TOL) else best
    result = {"id": sid, "date": time.strftime("%Y-%m-%d"), "config": args.config, "steps": base["train"]["steps"],
              "seeds": args.seeds, "tol_nats": TOL, "healthy_rule": "expert_load_max_frac <= 2/num_experts",
              "rule": "keep the current default unless another healthy variant beats it by more than tol_nats of val_loss",
              "chosen_label": chosen["label"], "changed_from_default": not chosen["is_current_default"],
              "chosen": chosen["values"], "table": table}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{sid}.json").write_text(json.dumps(result, indent=2))
    print(f"  -> {sid}: chose {chosen['label']}" + ("   (CHANGED from the config default)" if result["changed_from_default"] else "   (same as config default)"))
    return result


def fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    return v if isinstance(v, str) else repr(v)


def apply(configs, mdir):
    """Copy chosen values into config lines whose source is measurement:<id>."""
    for path in configs:
        lines, changed = Path(path).read_text().splitlines(), 0
        section = None
        for i, line in enumerate(lines):
            hm = re.match(r"^([A-Za-z_]\w*):\s*$", line)
            if hm:
                section = hm[1]
            m = re.match(r"^(\s+)(\w+):\s*(\S+)(\s+# src: measurement:)(\w+)(.*)$", line)
            if not (m and section):
                continue
            f = mdir / f"{m[5]}.json"
            chosen = json.loads(f.read_text())["chosen"] if f.exists() else {}
            dotted = f"{section}.{m[2]}"
            if dotted in chosen and fmt(chosen[dotted]) != m[3]:
                lines[i] = f"{m[1]}{m[2]}: {fmt(chosen[dotted])}".ljust(36) + m[4].lstrip() + m[5] + m[6]
                print(f"{path}: {dotted}: {m[3]} -> {fmt(chosen[dotted])}  (measurement {m[5]})")
                changed += 1
        Path(path).write_text("\n".join(lines) + "\n")
        print(f"{path}: {changed} values updated")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/small_cpu.yaml")
    ap.add_argument("--only", nargs="+", metavar="SWEEP")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--redo", action="store_true", help="repeat sweeps that already have a measurement file")
    ap.add_argument("--apply", nargs="+", metavar="CONFIG", help="write chosen values into these config files")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234])
    ap.add_argument("--steps", type=int, help="override train.steps (warmup becomes 5%% of it, evaluation at the end)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--set", action="append", default=[], metavar="section.key=value", help="override applied before the sweeps")
    ap.add_argument("--out-dir", default=str(ROOT / "measurements"))
    args = ap.parse_args()

    if args.list:
        for sid, v in SWEEPS.items():
            print(f"{sid:16s} covers {', '.join(sorted({k for d in v for k in d})):55s} {len(v)} variants")
        return
    if args.apply:
        apply(args.apply, Path(args.out_dir))
        return
    todo = list(SWEEPS) if args.all else (args.only or [])
    if not todo or any(s not in SWEEPS for s in todo):
        ap.error("choose --all or --only <sweep ...>  (see --list)")

    base = train.load_config(args.config, args.set)
    if args.steps:
        base["train"].update(steps=args.steps, warmup_steps=max(1, round(0.05 * args.steps)), eval_interval=args.steps)
    import torch
    args.device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = train.load_tokenizer(base) if base["embedding"]["embedding_type"] == "kronecker" else None
    data = train.load_tokens(base, base["model"]["vocab_size"])
    cache, outdir = {}, ROOT / "runs" / "sweep"
    print(f"config {args.config} | steps {base['train']['steps']} | seeds {args.seeds} | device {args.device}")
    if not args.redo:
        finished = [sid for sid in todo if (Path(args.out_dir) / f"{sid}.json").exists()]
        if finished:
            print(f"already measured, skipping (use --redo to repeat): {', '.join(finished)}")
        todo = [sid for sid in todo if sid not in finished]
    need, done = plan_runs(todo, base, args.seeds, outdir)
    prog = {"total": len(need) - len(done), "done": 0, "secs": []}
    print(f"planned: {len(need)} distinct runs, {len(done)} already finished on disk, {prog['total']} to run. "
          f"Each prints when it starts and ends; the first one gives the time estimate.", flush=True)
    for sid in todo:
        print(f"\n{sid}", flush=True)
        do_sweep(sid, base, args, tok, data, cache, outdir, prog)
    print(f"\nwrote {len(todo)} measurement files to {args.out_dir}\n"
          f"next: python harness/sweep.py --apply configs/config_v0.yaml configs/small_cpu.yaml configs/small_colab.yaml\n"
          f"      python harness/check_config.py configs/config_v0.yaml")


if __name__ == "__main__":
    main()
