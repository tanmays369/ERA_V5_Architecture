# Name: Revant Gupta
# Pack: ind-hi-fw2
"""Deterministic trainer: DP (DDP, optional ZeRO-1) / TP / PP on a (pp, dp, tp) mesh; NaN stop; checkpoints; JSONL metrics.
torchrun --nproc_per_node=N train.py --config CFG [--set k=v ...] --out DIR [--inject-nan S] [--resume CKPT] [--device cpu|cuda]
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before any CUDA init

import argparse
import collections
import contextlib
import glob
import hashlib
import json
import math
import platform
import shlex
import shutil
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import data  # noqa: E402
import model as model_mod  # noqa: E402
import optim  # noqa: E402
import parallel  # noqa: E402
import tok  # noqa: E402

try:
    from torch.distributed.tensor import DTensor
except ImportError:  # torch < 2.5
    from torch.distributed._tensor import DTensor

EMA_WARMUP = 10  # [ours] ema_z is null before this step: the EMA variance starts at 0, so early z-scores are meaningless

SCHEMA = [  # name, unit, cadence, source (G8 hand-off)
    ("step", "count", "every step", "optimizer steps completed before this one"),
    ("stage", "name", "every step", "data_stages entry active at this step"),
    ("loss", "nats/token", "every step", "total = loss_ntp + z_loss, mean over the global batch"),
    ("loss_ntp", "nats/token", "every step", "next-token cross-entropy, mean over the global batch (losses.tsv)"),
    ("loss_mtp", "nats/token", "every step", "reserved, MTP off in v0 (head.mtp_heads 0): always null"),
    ("z_loss", "nats/token", "every step", "z_loss_coef * mean(logsumexp(logits)^2)"),
    ("grad_norm", "L2", "every step", "global grad norm before clipping (all shards and stages)"),
    ("lr", "1", "every step", "WSD schedule optim.lr_at, applied to all param groups"),
    ("tokens_per_second", "tokens/s", "every step", "global_batch_seqs*seq_len / step wall time"),
    ("step_time_ms", "ms", "every step", "wall time of the step on rank 0 (incl. comm, excl. eval/checkpoint)"),
    ("gpu_alloc_gb", "GB", "every step", "peak allocated during the step: torch.cuda.max_memory_allocated()/1e9 on rank 0 "
     "(peak stats reset each step); 0 on CPU"),
    ("spike_score", "sigma", "every step", "(loss - mean)/std over the last safety.spike_window losses; null < 10 losses"),
    ("ema_z", "sigma", "every step", "(loss - EMA)/sqrt(EMA var), alpha safety.ema_alpha; null before step 10"),
    ("update_to_weight_ratio", "1", "every logging.per_layer_every", "||w_after - w_before|| / ||w_before||, all params"),
    ("max_attn_logit", "logit", "every logging.per_layer_every", "max |q.k|/sqrt(hd) over layers, last micro-batch"),
    ("held_out_loss", "nats/token", "every logging.eval_every", "mean CE over data.val_batch() windows (equal windows "
     "per source) after the step"),
    ("held_out_loss_<source>", "nats/token", "every logging.eval_every", "same, one key per dataset name (e.g. "
     "held_out_loss_hin)"),
    ("bpb", "bits/byte", "every logging.eval_every", "sum of window CE nats / ln2 / UTF-8 bytes of the evaluated target "
     "tokens (tokenizer byte lengths; EOS = 0), so it weights sources like the CE"),
    ("dead_experts", "count", "every step (MoE only)", "experts with 0 routed tokens this step, summed over MoE layers"),
    ("top_expert_share", "1", "every step (MoE only)", "max over MoE layers of the busiest expert's token share"),
    ("moe_aux", "nats/token", "every step (MoE only)", "seq-wise aux loss (coef applied), last micro-batch, summed "
     "over layers; in the gradient, not in `loss`"),
    ("moe_layers", "list", "every logging.per_layer_every (MoE only)", "one object per MoE layer: layer, load (fractions "
     "of routed slots per expert), dead_experts, top_expert_share, max_vio (max_load/mean_load - 1), router_entropy "
     "(nats, of scores/sum), intervention_rate (share of top-k picks where the bias changed the choice vs topk of the "
     "raw scores), topk_mass (mean share of the score sum on the selected experts), aux. Load/MaxVio are global "
     "(dp all-reduced); entropy/intervention/mass are token means over this step's training forwards, dp-reduced"),
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--inject-nan", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None)
    return ap.parse_args()


def set_determinism():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def local(x):
    """DTensors -> local shards, recursively (checkpoints are per rank)."""
    if isinstance(x, DTensor):
        return x.to_local()
    if isinstance(x, dict):
        return {k: local(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(local(v) for v in x)
    return x


def inner(opt):
    """ZeRO keeps this rank's shard optimizer in `.optim`; checkpoint that, per rank."""
    return getattr(opt, "optim", opt)


def save_ckpt(out, step, raw, opts, d, keep_last, frozen_hash, meta, monitor):
    """Atomic: all ranks write `step_N.tmp/rank{r}.pt`, rank 0 adds manifest.json and renames the dir to `step_N`."""
    path = os.path.join(out, "checkpoints", f"step_{step:06d}")
    tmp = path + ".tmp"
    if d.rank == 0 and os.path.exists(tmp):
        shutil.rmtree(tmp)  # leftover of a crashed save
    dist.barrier()
    os.makedirs(tmp, exist_ok=True)
    torch.save({"step": step, "model": local(raw.state_dict()), "optim": [local(inner(o).state_dict()) for o in opts],
                "frozen_sha256": frozen_hash, "monitor": monitor}, os.path.join(tmp, f"rank{d.rank}.pt"))
    dist.barrier()
    if d.rank == 0:
        with open(os.path.join(tmp, "manifest.json"), "w") as f:
            json.dump({**meta, "torch": torch.__version__, "step": step, "frozen_sha256": frozen_hash}, f, indent=1)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.replace(tmp, path)
        done = sorted(p for p in glob.glob(os.path.join(out, "checkpoints", "step_*")) if not p.endswith(".tmp"))
        for old in done[:-keep_last]:
            shutil.rmtree(old)
    dist.barrier()
    return path


def check_manifest(path, meta):
    """Refuse to resume under another layout or config (every rank raises, so none hangs)."""
    mf = os.path.join(path, "manifest.json")
    if not os.path.exists(mf):
        raise SystemExit(f"resume: {path} has no manifest.json (not a checkpoint written by this trainer)")
    with open(mf) as f:
        man = json.load(f)
    bad = [f"{k}: checkpoint {man.get(k)!r} vs this run {v!r}" for k, v in meta.items() if man.get(k) != v]
    if bad:
        raise SystemExit("resume refused, checkpoint does not match this run:\n  " + "\n  ".join(bad))
    if man.get("torch") != torch.__version__:
        print(f"warning: checkpoint written with torch {man.get('torch')}, running {torch.__version__}", flush=True)


def load_ckpt(path, raw, opts, d, strict, meta):
    """Restore this rank's model + optimizer state; returns (step, monitor state)."""
    check_manifest(path, meta)
    ck = torch.load(os.path.join(path, f"rank{d.rank}.pt"), map_location=d.device)
    sd = raw.state_dict()
    if strict and set(sd) != set(ck["model"]):
        raise SystemExit(f"checkpoint keys differ: {sorted(set(sd) ^ set(ck['model']))[:5]}")
    if ck["frozen_sha256"] is not None and frozen_sha(raw) != ck["frozen_sha256"]:  # frozen = never trained: must equal init
        raise SystemExit("resume refused: frozen weights differ from the checkpoint (frozen_sha256 mismatch)")
    with torch.no_grad():
        for k, v in ck["model"].items():
            local(sd[k]).copy_(v)
    for o, s in zip(opts, ck["optim"]):
        o = inner(o)
        o.load_state_dict(s)
        for p, st in o.state.items():  # re-wrap TP state as DTensor like its param
            if isinstance(p, DTensor):
                for k, v in st.items():
                    if torch.is_tensor(v) and not isinstance(v, DTensor) and v.dim() > 0:
                        st[k] = DTensor.from_local(v, p.device_mesh, p.placements, run_check=False)
    return ck["step"], ck["monitor"]


def frozen_sha(raw):
    h = hashlib.sha256()
    for n, p in raw.named_parameters():
        if not p.requires_grad:
            h.update(n.encode())
            h.update(local(p.detach()).cpu().numpy().tobytes())
    return h.hexdigest()


def resume_command(args, ckpt):
    """Same command line minus --inject-nan/--resume, plus --resume <ckpt>."""
    argv, skip = [], False
    for a in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if a in ("--inject-nan", "--resume"):
            skip = True
            continue
        if a.startswith(("--inject-nan=", "--resume=")):
            continue
        argv.append(a)
    n = int(os.environ.get("WORLD_SIZE", 1))
    return shlex.join(["torchrun", f"--nproc_per_node={n}", sys.argv[0], *argv, "--resume", ckpt])


def truncate_from(path, start):
    """Keep only records with step < start (resume into the same --out continues the files)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        lines = f.readlines()
    key = (lambda ln: int(ln.split("\t")[0])) if path.endswith(".tsv") else (lambda ln: json.loads(ln)["step"])
    with open(path, "w") as f:
        f.writelines(ln for ln in lines if key(ln) < start)


def main():
    args = parse_args()
    set_determinism()
    for k, v in (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0"), ("MASTER_ADDR", "127.0.0.1"),
                 ("MASTER_PORT", "29500")):
        os.environ.setdefault(k, v)  # plain `python train.py` works as a 1-proc run
    cfg = config.load_config(args.config, args.set)
    der = config.derived(cfg)
    d = parallel.setup_dist(cfg, args.device)
    cuda = d.device.type == "cuda"
    seed, T = cfg["general"]["seed"], cfg["tokens"]["train_steps"]
    G, m, L = cfg["tokens"]["global_batch_seqs"], cfg["tokens"]["micro_batch_seqs"], cfg["tokens"]["seq_len"]
    zc, clip = cfg["head"]["z_loss_coef"], cfg["optimizer"]["grad_clip"]
    lg, ck_cfg, sf = cfg["logging"], cfg["checkpoint"], cfg["safety"]
    ga = der["grad_accum"]
    out = args.out
    is0 = d.rank == 0
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "train.log"), "a") if is0 else None

    def log(msg):
        if is0:
            print(msg, flush=True)
            logf.write(msg + "\n")
            logf.flush()

    tok_path = cfg["paths"]["tokenizer"]
    if is0:
        tok.byte_buffers(tok_path, cfg["embedding"]["pos_dim"])  # build the cache once; others load it
    dist.barrier()

    torch.manual_seed(seed)  # identical full init on every rank and layout, then shard
    raw = model_mod.build_model(cfg, tok_path)
    for n, p in raw.named_parameters():
        if any(n.startswith(f) for f in cfg["freeze"]["frozen_parts"]):
            p.requires_grad_(False)
    if d.pp > 1:
        parallel.pp_stage_module(raw, d.pp_rank, d.pp)
    raw.to(d.device)
    if d.tp > 1:
        parallel.apply_tp(raw, d.mesh["tp"])
    net = raw
    if d.dp > 1:
        net = torch.nn.parallel.DistributedDataParallel(raw, process_group=d.dp_group)
    zero = cfg["parallelism"]["zero_stage"] == 1 and d.dp > 1
    opts = optim.build_optimizers(raw, cfg, zero_group=d.dp_group if zero else None)
    params = [p for p in raw.parameters() if p.requires_grad]
    moe_layers = raw.moe_layers()
    for mo in moe_layers:  # aux grad weight = this micro-batch's loss weight; PP schedule divides grads itself
        mo.aux_scale = 1.0 if d.pp > 1 else 1.0 / ga
    has_frozen = any(not p.requires_grad for p in raw.parameters())
    frozen_hash = frozen_sha(raw) if has_frozen and cfg["freeze"]["hash_frozen_weights"] else None

    meta = {"world": d.world, "pp": d.pp, "dp": d.dp, "tp": d.tp, "zero_stage": cfg["parallelism"]["zero_stage"],
            "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()}
    hist = collections.deque(maxlen=sf["spike_window"])  # spike/EMA state: checkpointed so alarms survive a resume
    ema = ema_var = None
    start, last_good = 0, None
    if args.resume:
        start, mon = load_ckpt(args.resume, raw, opts, d, ck_cfg["strict_load"], meta)
        hist.extend(mon["hist"])
        ema, ema_var = mon["ema"], mon["ema_var"]
        last_good = args.resume
        log(f"resumed from {args.resume} at step {start}")
    if is0:
        if start:
            truncate_from(os.path.join(out, "losses.tsv"), start)
            truncate_from(os.path.join(out, "metrics.jsonl"), start)
        man = os.path.join(cfg["paths"]["data"], "manifest.json")
        with open(os.path.join(out, "run.json"), "w") as f:
            json.dump({"config": cfg, "derived": der, "argv": sys.argv, "torch": torch.__version__,
                       "python": platform.python_version(), "cuda": torch.version.cuda,
                       "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda
                       else [platform.processor() or platform.machine()],
                       "layout": {"world": d.world, "pp": d.pp, "dp": d.dp, "tp": d.tp, "backend": dist.get_backend(),
                                  "zero_stage": cfg["parallelism"]["zero_stage"], "grad_accum": ga},
                       "data_manifest_sha256": hashlib.sha256(open(man, "rb").read()).hexdigest(),
                       "frozen_sha256": frozen_hash}, f, indent=1)
        with open(os.path.join(out, "metrics_schema.json"), "w") as f:
            json.dump([dict(zip(("name", "unit", "cadence", "source"), s)) for s in SCHEMA], f, indent=1)
    log(f"layout pp={d.pp} dp={d.dp} tp={d.tp} zero={int(zero)} grad_accum={ga} device={d.device} "
        f"params={sum(p.numel() for p in raw.parameters())} (rank-local) frozen_sha256={frozen_hash}")

    batches = data.Batches(cfg)
    val = batches.val_batch()
    val_src, val_bytes = batches.val_meta(tok_path)  # source index and target bytes of each val window (BPB)
    last_rank = parallel.last_stage_rank(d)
    nan_box = []

    def pp_loss(logits, y):
        tot, ce = model_mod.loss_fn(logits, y, zc)
        nan_box.append(torch.stack([tot.detach(), ce.detach()]))
        return tot * nan_box[0].new_tensor(float("nan")) if inject else tot

    schedule = parallel.build_schedule(raw, d, cfg, pp_loss) if d.pp > 1 else None
    if d.pp > 1 and moe_layers:  # the schedule's first forward is shape inference on dummy data: do not count it
        def drop_probe(mod, inp, outp):
            for mo in moe_layers:
                mo.load = mo.stats = None
            probe.remove()
        probe = raw.register_forward_hook(drop_probe)

    def held_out():
        """Mean CE per val window [Nv], chunked by micro_batch_seqs; same on every rank."""
        raw.eval()
        ce = torch.zeros(val.shape[0], device=d.device)
        with torch.no_grad():
            for i in range(0, val.shape[0], m):
                v = val[i:i + m].to(d.device)
                logits = parallel.pp_forward(raw, d, v[:, :-1], cfg["model"]["d_model"]) if d.pp > 1 else raw(v[:, :-1])
                if logits is not None:
                    for j in range(v.shape[0]):
                        ce[i + j] = model_mod.loss_fn(logits[j:j + 1], v[j:j + 1, 1:], 0.0)[1]
        raw.train()
        if d.pp > 1:
            dist.broadcast(ce, src=last_rank, group=d.pp_group)
        return ce.double().cpu()

    if is0:
        lf = open(os.path.join(out, "losses.tsv"), "a")
        mf = open(os.path.join(out, "metrics.jsonl"), "a")
    step_times = []
    for step in range(start, T):
        t0 = time.perf_counter()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        inject = args.inject_nan == step
        per_layer = lg["per_layer_every"] and step % lg["per_layer_every"] == 0
        raw.track_attn_logits = bool(per_layer)
        lr = optim.lr_at(step, cfg)
        for o in opts:
            for pg in o.param_groups:
                pg["lr"] = lr
        batch, _ = batches.global_batch(step)
        xs = data.Batches.dp_slice(batch, d.dp_rank, d.dp).to(d.device)
        acc = torch.zeros(2, device=d.device)  # (total, ce) summed over micro-batches
        if d.pp > 1:
            nan_box.clear()
            parallel.pp_step(schedule, d, xs[:, :-1].contiguous(), xs[:, 1:].contiguous(), [])  # torch 2.11 PipelineStage checks strides
            if nan_box:
                # first step() also calls loss_fn once for shape inference on uninitialised buffers: keep the real ones
                acc = torch.stack(nan_box[-cfg["parallelism"]["pp_microbatches"]:]).mean(0)
                if inject:
                    acc = acc * float("nan")
            dist.broadcast(acc, src=last_rank, group=d.pp_group)
        else:
            for i in range(ga):
                mb = xs[i * m:(i + 1) * m]
                sync = d.dp == 1 or i == ga - 1
                with contextlib.nullcontext() if sync else net.no_sync():
                    tot, ce = model_mod.loss_fn(net(mb[:, :-1]), mb[:, 1:], zc)
                    if inject:
                        tot = tot * float("nan")
                    (tot / ga).backward()  # equal-size micro-batches: per-token mean over the global batch
                acc += torch.stack([tot.detach(), ce.detach()])
            acc /= ga
        if d.dp > 1:
            dist.all_reduce(acc, group=d.dp_group)
            acc /= d.dp
        if d.tp > 1:
            parallel.sync_tp_grads(raw, d)
        gnorm = parallel.clip_grads(params, clip, d)
        loss, ce = float(acc[0]), float(acc[1])
        bad = torch.tensor(float(not (math.isfinite(loss) and math.isfinite(gnorm))), device=d.device)
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        if is0:
            lf.write(f"{step}\t{ce!r}\n")
            lf.flush()
        if bad.item():
            cmd = resume_command(args, last_good) if last_good else None
            if is0:
                with open(os.path.join(out, "NAN_REPORT.json"), "w") as f:
                    json.dump({"step": step, "loss": repr(loss), "loss_ntp": repr(ce), "grad_norm": repr(gnorm),
                               "injected": inject, "last_good_checkpoint": last_good, "resume_command": cmd}, f,
                              indent=1)
                log(f"NaN/Inf at step {step}. Last good checkpoint: {last_good or 'none'}. Resume: {cmd or 'none'}")
            dist.destroy_process_group()
            sys.exit(3)
        prev = [p.detach().clone() for p in params] if per_layer else None
        for o in opts:
            o.step()
        for o in opts:
            o.zero_grad(set_to_none=True)
        if moe_layers:
            model_mod.update_expert_bias(raw, d.dp_group if d.dp > 1 else None)
        if cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        step_times.append(dt)

        rec = {"step": step, "stage": batches.stage_name(step), "loss": loss, "loss_ntp": ce, "loss_mtp": None,
               "z_loss": loss - ce, "grad_norm": gnorm, "lr": lr, "tokens_per_second": G * L / dt,
               "step_time_ms": dt * 1e3, "gpu_alloc_gb": torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0}
        rec["spike_score"] = (loss - sum(hist) / len(hist)) / (
            math.sqrt(sum((h - sum(hist) / len(hist)) ** 2 for h in hist) / len(hist)) + 1e-12) if len(hist) >= 10 else None
        rec["ema_z"] = (loss - ema) / (math.sqrt(ema_var) + 1e-12) if step >= EMA_WARMUP else None
        hist.append(loss)
        a = sf["ema_alpha"]
        if ema is None:
            ema, ema_var = loss, 0.0
        else:
            ema_var = a * ema_var + (1 - a) * (loss - ema) ** 2
            ema = a * ema + (1 - a) * loss
        if per_layer:
            dw = parallel.global_sq([p.detach() - q for p, q in zip(params, prev)], d)
            w = parallel.global_sq(prev, d)
            rec["update_to_weight_ratio"] = float((dw / w).sqrt())
            mx = torch.tensor(max([b.attn.last_max_logit or 0.0 for b in raw.layers], default=0.0), device=d.device)
            dist.all_reduce(mx, op=dist.ReduceOp.MAX)
            rec["max_attn_logit"] = float(mx)
        if cfg["moe"]["enabled"]:  # every step; PP stages hold disjoint layers, so reduce over pp
            mm = model_mod.moe_metrics(raw, d.dp_group if d.dp > 1 else None)
            v = torch.tensor([sum(x["dead_experts"] for x in mm), max([x["top_expert_share"] for x in mm], default=0.0),
                              sum(x["aux"] for x in mm)], dtype=torch.float64, device=d.device)
            if d.pp > 1:
                mx = v.clone()
                dist.all_reduce(v, group=d.pp_group)
                dist.all_reduce(mx, op=dist.ReduceOp.MAX, group=d.pp_group)
                v[1] = mx[1]
            rec["dead_experts"], rec["top_expert_share"], rec["moe_aux"] = int(v[0]), float(v[1]), float(v[2])
            if per_layer:
                mine = [{"layer": d.pp_rank * len(raw.layers) + i, **x} for i, x in enumerate(mm)]
                if d.pp > 1:
                    gath = [None] * d.pp
                    dist.all_gather_object(gath, mine, group=d.pp_group)
                    mine = [x for g in gath for x in g]
                rec["moe_layers"] = mine
        if lg["eval_every"] and (step + 1) % lg["eval_every"] == 0:
            ce_rows = held_out()  # mean CE per val window; equal windows per source, so CE and BPB weight sources equally
            rec["held_out_loss"] = float(ce_rows.mean())
            rec["bpb"] = float(ce_rows.sum() * L / math.log(2) / val_bytes.sum())  # total nats / ln2 / total target bytes
            for s, name in enumerate(batches.names):
                rec[f"held_out_loss_{name}"] = float(ce_rows[val_src == s].mean())
        if is0:
            mf.write(json.dumps(rec) + "\n")
            mf.flush()
        log(f"step {step} loss {loss:.6f} ce {ce:.6f} gnorm {gnorm:.4f} lr {lr:.3e} {dt * 1e3:.1f} ms"
            + (f" held_out {rec['held_out_loss']:.4f} bpb {rec['bpb']:.4f}" if "bpb" in rec else ""))
        if (step + 1) % ck_cfg["every_steps"] == 0 or step + 1 == T:
            if has_frozen and cfg["freeze"]["hash_frozen_weights"]:
                frozen_hash = frozen_sha(raw)
                log(f"frozen_sha256 {frozen_hash}")
            last_good = save_ckpt(out, step + 1, raw, opts, d, ck_cfg["keep_last"], frozen_hash, meta,
                                  {"hist": list(hist), "ema": ema, "ema_var": ema_var})
            log(f"checkpoint {last_good}")
    if step_times:
        st = sorted(step_times[1:] or step_times)
        log(f"done: median step_time_ms {st[len(st) // 2] * 1e3:.1f} (excl. step {start})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
