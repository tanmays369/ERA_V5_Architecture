"""Train a tiny test model from a G5 config, and log the agreed metrics at every step.

Usage:
    python harness/train.py --config configs/tiny/tiny_moe.yaml [--steps N] [--seed S] [--out DIR] [--device cpu]

What one step does (S10, S11):
  1. take global_batch_tokens worth of windows from the mixed sample, split into micro-batches;
  2. for each micro-batch: forward pass, loss = next-token loss + mtp_loss_weight x MTP loss + z-loss
     + MoE balance loss, divided by the number of target tokens in the whole step (the loss is normalised
     per token, not per micro-batch), then backward;
  3. clip the gradient norm, take an AdamW step at the learning rate of the WSD schedule, and move the
     expert biases towards balance;
  4. write one line of metrics to metrics.jsonl, and the exact loss values to losses.txt.
Every eval_interval steps it also measures bits per byte on a fixed set of held-out windows.

Outputs (in --out, default work/runs/<config>_seed<seed>/):
  metrics.jsonl   one JSON line per step with the metric names agreed for G8
  losses.txt      step and losses written as exact hexadecimal floats; the seed test compares these files
  run_info.json   the resolved config, parameter counts, device and library versions
The run stops with an error at the first NaN or infinite loss.
"""
import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configlib import get, load_config, require  # noqa: E402
from determinism import make_deterministic  # noqa: E402

import torch  # noqa: E402  (after determinism's environment setting is importable)

from data import MixedSample  # noqa: E402
from model import TinyModel, chunked_cross_entropy, count_parameters  # noqa: E402
from tokenizer import HFLikeTokenizer  # noqa: E402

LN2 = math.log(2)


def wsd_lr(step, s):
    """Warm-up, stable, decay [S11]: linear warm-up, flat at peak, then a decay over the last
    decay_fraction of the steps down to final_lr_ratio x peak."""
    peak, warm, total = s["peak_lr"], s["warmup_steps"], s["total_steps"]
    decay_start = int(total * (1 - s["decay_fraction"]))
    if step < warm:
        return peak * (step + 1) / warm
    if step < decay_start:
        return peak
    frac = (step - decay_start) / max(1, total - decay_start)
    return peak * (1 - (1 - s["final_lr_ratio"]) * frac)


def param_groups(model, weight_decay):
    """No weight decay on normalisation scales, biases or the router [S11]."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.ndim < 2 or "norm" in name or name.endswith(".bias") or ".router." in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def run(config_path, steps=None, seed=None, out=None, device=None):
    cfg = load_config(config_path)
    sched = dict(cfg["lr_schedule"]["schedule"])
    if steps:
        sched["total_steps"] = steps
        sched["warmup_steps"] = min(sched["warmup_steps"], max(1, steps // 10))
    seed = int(seed if seed is not None else cfg["training"]["seed"])
    h = cfg.get("harness", {})
    make_deterministic(seed, h.get("threads"))
    dev = torch.device(device or h.get("device", "cpu"))
    if dev.type == "cuda" and cfg["embedding"]["mode"] == "dynamic":
        raise SystemExit("embedding.mode 'dynamic' uses scatter_add, which deterministic mode refuses on a GPU. "
                         "Use mode: cached for the tiny vocabulary.")

    out = Path(out or Path(get(cfg, "data_mix", "harness_sample", "output_dir")).parent / "runs" /
               f"{Path(config_path).stem}_seed{seed}")
    out.mkdir(parents=True, exist_ok=True)

    tok = HFLikeTokenizer(require(cfg, "tokenizer", "path"))
    seq_len = require(cfg, "training", "sequence_length")
    rows_per_step = require(cfg, "training", "global_batch_tokens") // seq_len
    micro = h.get("micro_batch_size", rows_per_step)
    n_micro = rows_per_step // micro
    mtp_n = cfg["output_head"]["mtp_heads"]
    hs = cfg["data_mix"]["harness_sample"]
    data = MixedSample(hs["output_dir"], {s: hs[s + "_share"] for s in ("english", "hindi", "bengali", "telugu")},
                       seq_len, mtp_n, seed)
    val_set = data.validation_set(h.get("eval_batches", 4), micro)

    model = TinyModel(cfg, tok).to(dev)
    o = cfg["optimizer"]
    opt = torch.optim.AdamW(param_groups(model, o["weight_decay"]), lr=sched["peak_lr"],
                            betas=(o["beta1"], o["beta2"]), eps=o["eps"])
    params = count_parameters(model)
    info = {"config": str(Path(config_path).resolve()), "seed": seed, "device": str(dev), "steps": sched["total_steps"],
            "rows_per_step": rows_per_step, "micro_batch_size": micro, "sequence_length": seq_len,
            "parameters": params, "torch": torch.__version__, "python": platform.python_version(),
            "sample_manifest": data.manifest, "resolved_config": cfg}
    (out / "run_info.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    print(f"{Path(config_path).name}: {params['total']:,} parameters ({params['active']:,} active per token), "
          f"{sched['total_steps']} steps of {rows_per_step * seq_len:,} tokens, device {dev}, seed {seed}")

    w_mtp, z_coef = cfg["output_head"]["mtp_loss_weight"], cfg["output_head"]["z_loss_coef"]
    rate = cfg["moe"]["bias_update_rate"]
    metrics_f = open(out / "metrics.jsonl", "w", encoding="utf-8")
    losses_f = open(out / "losses.txt", "w", encoding="utf-8")
    eval_every = h.get("eval_interval", 0)

    for step in range(sched["total_steps"]):
        t0 = time.perf_counter()
        lr = wsd_lr(step, sched)
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        opt.zero_grad(set_to_none=True)
        n_targets = rows_per_step * seq_len
        sums = {"ntp": 0.0, "mtp": 0.0, "z": 0.0, "aux": 0.0, "bytes": 0}
        for _ in range(n_micro):
            win = data.batch(micro).to(dev)
            inp = win[:, :seq_len]
            hidden = model(inp)
            ntp, z = chunked_cross_entropy(hidden, model.head, win[:, 1:seq_len + 1], z_coef=z_coef)
            mtp = hidden.new_zeros((), dtype=torch.float32)
            for j, head in enumerate(model.mtp_heads):
                m_sum, _ = chunked_cross_entropy(hidden, head, win[:, 2 + j:seq_len + 2 + j])
                mtp = mtp + m_sum
            aux = sum((m.last_aux for m in model.moe_layers()), hidden.new_zeros((), dtype=torch.float32))
            loss = (ntp + w_mtp * mtp / max(1, mtp_n) + z_coef * z) / n_targets + aux / n_micro
            loss.backward()
            sums["ntp"] += ntp.item(); sums["mtp"] += mtp.item(); sums["z"] += z.item(); sums["aux"] += aux.item()
            sums["bytes"] += int(data.token_bytes[win[:, 1:seq_len + 1].cpu()].sum())
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), o["grad_clip_norm"]).item()
        opt.step()
        loads = [m.update_expert_bias(rate) for m in model.moe_layers()]   # counts of the whole step
        dt = time.perf_counter() - t0

        loss_ntp = sums["ntp"] / n_targets
        loss_mtp = sums["mtp"] / n_targets / max(1, mtp_n)
        loss_total = loss_ntp + w_mtp * loss_mtp + z_coef * sums["z"] / n_targets + sums["aux"] / n_micro
        if not math.isfinite(loss_total):
            raise SystemExit(f"Step {step}: loss is {loss_total}. Stopping (the harness never trains on a NaN).")
        rec = {"step": step, "loss": loss_total, "loss_ntp": loss_ntp, "loss_mtp": loss_mtp, "lr": lr,
               "tokens_per_second": n_targets / dt,
               "gpu_alloc_gb": torch.cuda.memory_allocated() / 1e9 if dev.type == "cuda" else 0.0,
               "bits_per_byte": sums["ntp"] / (LN2 * sums["bytes"]), "grad_norm": grad_norm,
               "moe_aux_loss": sums["aux"] / n_micro}
        if loads:
            rel = [l / l.mean() for l in loads]                       # 1.0 = an even share of the tokens
            rec["expert_load_max"] = max(float(r.max()) for r in rel)
            rec["expert_load_min"] = min(float(r.min()) for r in rel)
        if eval_every and (step % eval_every == 0 or step == sched["total_steps"] - 1):
            rec["val_bits_per_byte"] = evaluate(model, val_set, data, seq_len, dev)
        metrics_f.write(json.dumps(rec) + "\n")
        losses_f.write(f"{step} {float(loss_total).hex()} {float(loss_ntp).hex()} {float(loss_mtp).hex()} "
                       f"{float(grad_norm).hex()}\n")
        if step % max(1, sched["total_steps"] // 10) == 0 or step == sched["total_steps"] - 1:
            extra = f" val_bpb {rec['val_bits_per_byte']:.3f}" if "val_bits_per_byte" in rec else ""
            load = f" load {rec['expert_load_min']:.2f}-{rec['expert_load_max']:.2f}" if loads else ""
            print(f"  step {step:4d}  loss {loss_total:.4f}  ntp {loss_ntp:.4f}  mtp {loss_mtp:.4f}  "
                  f"bpb {rec['bits_per_byte']:.3f}{extra}  lr {lr:.2e}{load}  {rec['tokens_per_second']:,.0f} tok/s")
    metrics_f.close()
    losses_f.close()
    print(f"Wrote {out}")
    return out


@torch.no_grad()
def evaluate(model, val_set, data, seq_len, dev):
    model.eval()
    nats, nbytes = 0.0, 0
    for win in val_set:
        win = win.to(dev)
        hidden = model(win[:, :seq_len])
        s, _ = chunked_cross_entropy(hidden, model.head, win[:, 1:seq_len + 1])
        nats += s.item()
        nbytes += int(data.token_bytes[win[:, 1:seq_len + 1].cpu()].sum())
    model.train()
    return nats / (LN2 * nbytes)


def main():
    ap = argparse.ArgumentParser(description="Train a tiny G5 test model.")
    ap.add_argument("--config", default=str(HERE.parent / "configs" / "tiny" / "tiny_moe.yaml"))
    ap.add_argument("--steps", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out")
    ap.add_argument("--device")
    a = ap.parse_args()
    run(a.config, a.steps, a.seed, a.out, a.device)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
