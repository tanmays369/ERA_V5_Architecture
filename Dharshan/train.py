# Name: dharshan
# Axiom email: dharshanthaigu@gmail.com
# Pack: ind-ta-fw2
"""Plain training loop (no PyTorch Lightning).

Run from the folder that holds configs/ and harness/ (Windows: py instead of python):
    python harness/train.py --config configs/config_small.yaml --out <folder outside the repo>/a
    python harness/train.py --config configs/config_small.yaml --out <folder>/b --set attention.window=8

Files written to --out:
    losses.txt         "step loss_ntp loss_mtp loss" as exact hex floats; compare two runs with a plain diff
    metrics.csv        one row per optimiser step (columns in METRICS)
    moe_layers.csv     per-layer MoE stats every logging.eval_every steps
    config_resolved.yaml, run_info.json (versions, parameter counts, keys read, optimiser group of every parameter)
    checkpoint.pt      only with --ckpt_every / --stop_after: model, optimiser state and the next step (written atomically)
    NAN_REPORT.json    only if the loss went non-finite and safety.stop_on_nan is true (exit code 2)
There is no device setting in the config: the run uses cuda if available, else cpu (or --device).
There are no spike or EMA alarms: G8 delivers those.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config  # noqa: E402
from data import Data  # noqa: E402
from model import Transformer  # noqa: E402
from optim import Optim  # noqa: E402

METRICS = ["step", "tokens_per_second", "loss_ntp", "loss_mtp", "loss", "lr", "gpu_alloc_gb",
           "grad_norm", "z_loss", "held_out_loss", "dead_experts", "top_expert_share", "max_vio", "lr_muon"]


def seed_everything(cfg):
    s = cfg.get("general.seed")
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if cfg.get("general.deterministic"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.set_num_threads(cfg.get("general.num_threads"))


@torch.no_grad()
def held_out_loss(model, data, device):
    """Mean over sources of the next-token loss on that source's held-out windows: equal weight per source."""
    model.eval()
    per = [float(model(w.to(device))["loss_ntp"]) for w in data.held_out().values()]
    model.train()
    return sum(per) / len(per)


def save_checkpoint(path, model, opt, next_step, cfg_hash):
    """Write to a temporary file, then rename: a crash in the middle never leaves a half-written checkpoint."""
    tmp = Path(str(path) + ".tmp")
    torch.save({"next_step": next_step, "config_sha256": cfg_hash, "model": model.state_dict(),
                "adamw": opt.adamw.state_dict(), "muon": opt.muon.state_dict() if opt.muon is not None else None}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, opt, cfg_hash):
    ck = torch.load(path, map_location="cpu")
    if ck["config_sha256"] != cfg_hash:
        raise SystemExit("checkpoint was made with a different resolved config (check --config and --set)")
    model.load_state_dict(ck["model"])
    opt.adamw.load_state_dict(ck["adamw"])
    if opt.muon is not None:
        opt.muon.load_state_dict(ck["muon"])
    return ck["next_step"]


def trim_rows(path, keep_below, header):
    """On resume: drop rows written after the checkpoint (first column = step) so the files stay one row per step."""
    lines = Path(path).read_text().splitlines()
    head, body = (lines[:1], lines[1:]) if header else ([], lines)
    keep = [l for l in body if int(l.split(",")[0].split()[0]) < keep_below]
    Path(path).write_text("\n".join(head + keep) + ("\n" if head + keep else ""))


def moe_stats(loads):
    """From per-layer routed-token counts: (dead experts summed over layers, top share, max_vio), plus per-layer rows."""
    rows = []
    for i, l in enumerate(loads):
        mean = l.mean()
        rows.append((i, int((l == 0).sum()), float(l.max() / l.sum()), float((l.max() - mean) / mean)))
    if not rows:
        return "", "", "", rows
    return sum(r[1] for r in rows), max(r[2] for r in rows), max(r[3] for r in rows), rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="a.b=value")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--ckpt_every", type=int, default=0, help="save <out>/checkpoint.pt every N steps (0 = never)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/checkpoint.pt")
    ap.add_argument("--stop_after", type=int, default=0, help="test only: save a checkpoint and stop after this many steps")
    ap.add_argument("--nan_at", type=int, default=-1, help="test only: make the loss NaN at this step")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.set)
    g = cfg.get
    seed_everything(cfg)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    data = Data(cfg)
    bb, lb = data.byte_buffers()
    model = Transformer(cfg, bb, lb).to(device)
    opt = Optim(model, cfg)
    total_p, active_p = model.count_params()
    G, M, T = g("tokens.global_batch_seqs"), g("tokens.micro_batch_seqs"), g("tokens.seq_len")
    if G % M:
        raise SystemExit("tokens.global_batch_seqs must be divisible by tokens.micro_batch_seqs")
    accum, steps, eval_every = G // M, g("tokens.train_steps"), g("logging.eval_every")
    speed, stop_nan = g("moe.bias_update_speed"), g("safety.stop_on_nan")
    cfg_hash = hashlib.sha256(yaml.safe_dump(cfg.tree, sort_keys=True).encode()).hexdigest()
    ckpt, start = out / "checkpoint.pt", 0
    if a.resume:
        if not ckpt.is_file():
            raise SystemExit(f"--resume: {ckpt} does not exist")
        start = load_checkpoint(ckpt, model, opt, cfg_hash)
        for name, header in (("losses.txt", False), ("metrics.csv", True), ("moe_layers.csv", True)):
            trim_rows(out / name, start, header)
    print(f"device={device} params={total_p} active={active_p} seed={g('general.seed')} steps={steps}", flush=True)

    mode = "a" if a.resume else "w"
    fl = open(out / "losses.txt", mode)
    fm = open(out / "metrics.csv", mode, newline="")
    fs = open(out / "moe_layers.csv", mode, newline="")
    wm, ws = csv.writer(fm), csv.writer(fs)
    if not a.resume:
        wm.writerow(METRICS)
        ws.writerow(["step", "layer", "dead_experts", "top_expert_share", "max_vio"])
    model.train()
    for step in range(start, steps):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        lr_a, lr_m = opt.set_lr(step)
        opt.zero_grad()
        s = {"loss_ntp": 0.0, "loss_mtp": 0.0, "loss": 0.0, "z_loss": 0.0}
        has_mtp = False
        for micro in range(accum):
            o = model(data.train_batch(step, micro).to(device))
            (o["loss"] / accum).backward()
            for k in s:
                if o[k] is not None:
                    s[k] += float(o[k]) / accum
            has_mtp = has_mtp or o["loss_mtp"] is not None
        gnorm = opt.clip_grads()
        if step == a.nan_at:
            s["loss"] = float("nan")
        if stop_nan and not all(np.isfinite(v) for v in (*s.values(), gnorm)):
            json.dump({"step": step, "losses": s, "grad_norm": gnorm, "lr": lr_a}, open(out / "NAN_REPORT.json", "w"), indent=1)
            print(f"non-finite value at step {step}, see NAN_REPORT.json", flush=True)
            sys.exit(2)
        opt.step()
        loads = model.finish_step(speed)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        gpu = f"{torch.cuda.max_memory_allocated() / 1e9:.4f}" if device == "cuda" else ""
        dead, share, vio, rows = moe_stats(loads)
        held = held_out_loss(model, data, device) if (step % eval_every == 0 or step == steps - 1) else ""
        mtp_hex = float(s["loss_mtp"]).hex() if has_mtp else "-"
        fl.write(f"{step} {float(s['loss_ntp']).hex()} {mtp_hex} {float(s['loss']).hex()}\n")
        wm.writerow([step, f"{G * T / dt:.1f}", f"{s['loss_ntp']:.9g}", f"{s['loss_mtp']:.9g}" if has_mtp else "",
                     f"{s['loss']:.9g}", f"{lr_a:.6g}", gpu, f"{gnorm:.6g}", f"{s['z_loss']:.6g}",
                     f"{held:.9g}" if held != "" else "", dead, f"{share:.6g}" if share != "" else "",
                     f"{vio:.6g}" if vio != "" else "", f"{lr_m:.6g}"])
        if step % eval_every == 0 or step == steps - 1:
            for r in rows:
                ws.writerow([step, r[0], r[1], f"{r[2]:.6g}", f"{r[3]:.6g}"])
        if step % 10 == 0 or step == steps - 1:
            print(f"step {step:5d} loss_ntp {s['loss_ntp']:.4f} loss {s['loss']:.4f} lr {lr_a:.5f} "
                  f"tok/s {G * T / dt:8.0f} gpu_gb {gpu or 'null'}", flush=True)
        stop_here = a.stop_after and step + 1 == a.stop_after
        if (a.ckpt_every and (step + 1) % a.ckpt_every == 0) or stop_here:
            for f in (fl, fm, fs):
                f.flush()
            save_checkpoint(ckpt, model, opt, step + 1, cfg_hash)
        if stop_here:
            break
    for f in (fl, fm, fs):
        f.close()
    with open(out / "config_resolved.yaml", "w") as f:
        yaml.safe_dump(cfg.tree, f, sort_keys=False)
    json.dump({"torch": torch.__version__, "device": device, "params_total": total_p, "params_active": active_p,
               "config_sha256": cfg_hash, "config_keys_read": sorted(cfg.reads), "optimizer_group": opt.assignment, "argv": sys.argv[1:]},
              open(out / "run_info.json", "w"), indent=1)


if __name__ == "__main__":
    main()
