#!/usr/bin/env python3
"""train.py - tiny MoE harness (Team 05), written from scratch in PyTorch, Kronecker input embeddings.

Model (all options are in the YAML; nothing is hard-coded to a size):
  * decoder-only transformer: RMSNorm, RoPE, grouped-query attention (optional QK-norm), SwiGLU feed-forwards
  * MoE feed-forward layers: routed experts (top-k) + optional always-on shared experts
      - router: softmax or sigmoid scores, optional renormalised top-k weights
      - load balancing: "aux_loss" (Switch-style loss), "bias" (loss-free: a per-expert selection bias nudged
        toward an even load after every step), or "none"; plus an optional router z-loss
      - layer layout: first_k_dense dense layers, then an MoE layer every moe_layer_freq layers
  * input embedding: KroneckerEmbedding (kronecker-embeddings package) with the course tokenizer; the output
    layer is separate (Kronecker embeddings cannot be tied)
  * multi-token prediction (MTP): mtp_depth extra heads predict tokens t+2 ... t+1+depth from the final hidden
    state through a small residual MLP and the shared output matrix; this gives loss_mtp
Data: token file from harness/prepare_data.py (Hindi + English).

Usage (from teams/05-architecture):
    python harness/train.py --config configs/small_cpu.yaml
    python harness/train.py --config configs/small_colab.yaml --device cuda
    python harness/train.py --config configs/small_cpu.yaml --set moe.top_k=1 --set train.steps=50
    python harness/train.py --list-metrics

Determinism test (the Sat 31 Oct check):
    python harness/train.py --config configs/small_cpu.yaml --seed 1234 --log /tmp/run1.csv
    python harness/train.py --config configs/small_cpu.yaml --seed 1234 --log /tmp/run2.csv
    diff /tmp/run1.csv /tmp/run2.csv && echo IDENTICAL

Notes
  * <log>.csv holds only reproducible numbers (METRIC_NAMES). Speed and memory go to <log>.system.csv (never diff it).
    <log>.summary.json holds the end-of-run numbers that harness/sweep.py reads.
  * Experts are computed densely (every expert on every token, multiplied by a gate that is zero for unselected
    experts). Outputs and gradients equal sparse routing; it avoids scatter/index_add, which can be non-deterministic
    on GPU. Fine for a tiny harness, NOT for real training: tokens/second here says nothing about real MoE speed.
  * Logging goes through CsvLogger.log_row(); G8's trainer logger can replace it later.
"""
import os

# Must be set before CUDA starts, or deterministic cuBLAS raises an error.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent.parent

# ---- metric names: this list is the contract with G8. Do not rename casually. ----
METRIC_HELP = {
    "step": "1-based optimizer step",
    "loss": "total train loss = loss_ntp + mtp_loss_coef*loss_mtp + load_balance_coef*load_balance_loss (aux_loss method) + router_z_loss_coef*router_z_loss",
    "loss_ntp": "next-token-prediction cross-entropy on the train batch (nats)",
    "loss_mtp": "mean cross-entropy of the multi-token-prediction heads (empty when mtp_depth = 0)",
    "load_balance_loss": "Switch-style balance loss, mean over MoE layers (1.0 = perfectly balanced)",
    "router_z_loss": "ST-MoE style z-loss, mean over MoE layers",
    "router_entropy": "mean per-token router entropy in nats (max = ln(num_experts)); falling toward 0 = router collapse",
    "expert_load_cv": "std/mean of the expert load share, mean over MoE layers (0 = even)",
    "expert_load_max_frac": "largest expert share of assignments, mean over MoE layers (ideal = 1/num_experts)",
    "grad_norm": "global grad norm before clipping",
    "lr": "learning rate used at this step",
    "val_loss": "held-out next-token cross-entropy; filled only on eval steps, else empty",
}
METRIC_NAMES = list(METRIC_HELP)
SYSTEM_METRIC_NAMES = ["step", "tokens_per_second", "gpu_alloc_gb"]  # not reproducible, kept in the .system.csv


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


# --------------------------------------------------------------------------- determinism
def set_determinism(seed, warn_only=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# --------------------------------------------------------------------------- config
def apply_overrides(cfg, overrides):
    for item in overrides:
        key, val = item.split("=", 1)
        sec, name = key.split(".", 1)
        if sec not in cfg or name not in cfg[sec]:
            sys.exit(f"--set {key}: unknown setting")
        cfg[sec][name] = yaml.safe_load(val)
    return cfg


def validate(cfg):
    open_sizes = [f"{s}.{k}" for s, d in cfg.items() for k, v in d.items() if v is None]
    if open_sizes:
        sys.exit("These settings are still null (sizes TSAI has not given yet): " + ", ".join(open_sizes)
                 + "\nUse configs/small_cpu.yaml or small_colab.yaml, or pass --set section.key=value.")
    m, e, t, emb, mt = cfg["model"], cfg["moe"], cfg["train"], cfg["embedding"], cfg["mtp"]
    assert m["d_model"] % m["n_heads"] == 0, "d_model must be divisible by n_heads"
    assert m["n_heads"] % m["n_kv_heads"] == 0, "n_heads must be divisible by n_kv_heads"
    assert (m["d_model"] // m["n_heads"]) % 2 == 0, "head dim must be even for RoPE"
    assert t["seq_len"] <= m["max_seq_len"], "train.seq_len must be <= model.max_seq_len"
    assert m["norm_type"] == "rmsnorm" and m["activation"] == "swiglu", "harness implements rmsnorm + swiglu only"
    assert emb["embedding_type"] in ("kronecker", "standard"), "embedding_type must be kronecker or standard"
    if emb["embedding_type"] == "kronecker":
        assert not emb["tie_embeddings"], "tie_embeddings must be false with Kronecker embeddings"
    assert e["router_type"] in ("softmax", "sigmoid"), "router_type must be softmax or sigmoid"
    assert e["balance_method"] in ("aux_loss", "bias", "none"), "balance_method must be aux_loss, bias or none"
    assert 1 <= e["top_k"] <= e["num_experts"], "need 1 <= top_k <= num_experts"
    assert e["moe_layer_freq"] >= 1 and e["first_k_dense"] >= 0
    assert mt["mtp_depth"] >= 0 and mt["mtp_depth"] < t["seq_len"]
    return cfg


def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return validate(apply_overrides(cfg, list(overrides)))


# --------------------------------------------------------------------------- model
class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def apply_rope(x, cos, sin):  # x: [B, H, T, hd]; cos/sin: [T, hd/2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, m):
        super().__init__()
        d, H, KV = m["d_model"], m["n_heads"], m["n_kv_heads"]
        self.H, self.KV, self.hd, self.p = H, KV, d // H, m["attn_dropout"]
        self.q = nn.Linear(d, H * self.hd, bias=False)
        self.k = nn.Linear(d, KV * self.hd, bias=False)
        self.v = nn.Linear(d, KV * self.hd, bias=False)
        self.o = nn.Linear(H * self.hd, d, bias=False)
        self.qk_norm = m["qk_norm"]
        if self.qk_norm:
            self.qn, self.kn = RMSNorm(self.hd, m["norm_eps"]), RMSNorm(self.hd, m["norm_eps"])
        inv = 1.0 / (m["rope_theta"] ** (torch.arange(0, self.hd, 2).float() / self.hd))
        freqs = torch.outer(torch.arange(m["max_seq_len"]).float(), inv)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)
        L = m["max_seq_len"]
        self.register_buffer("mask", torch.triu(torch.ones(L, L, dtype=torch.bool), 1), persistent=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.H, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.KV, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.KV, self.hd).transpose(1, 2)
        if self.qk_norm:
            q, k = self.qn(q), self.kn(k)
        q, k = apply_rope(q, self.cos[:T], self.sin[:T]), apply_rope(k, self.cos[:T], self.sin[:T])
        rep = self.H // self.KV
        if rep > 1:  # grouped-query attention: share each KV head across `rep` query heads (views only)
            k = k[:, :, None].expand(B, self.KV, rep, T, self.hd).reshape(B, self.H, T, self.hd)
            v = v[:, :, None].expand(B, self.KV, rep, T, self.hd).reshape(B, self.H, T, self.hd)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        att = att.masked_fill(self.mask[:T, :T], float("-inf")).softmax(-1)
        att = F.dropout(att, self.p, self.training)
        return self.o((att @ v).transpose(1, 2).reshape(B, T, self.H * self.hd))


class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w1, self.w3, self.w2 = nn.Linear(d, h, bias=False), nn.Linear(d, h, bias=False), nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    def __init__(self, m, e):
        super().__init__()
        d, E, h = m["d_model"], e["num_experts"], e["expert_hidden_dim"]
        self.E, self.k = E, e["top_k"]
        self.router_type, self.normalize, self.balance = e["router_type"], e["normalize_topk_weights"], e["balance_method"]
        self.router = nn.Linear(d, E, bias=False)
        self.w1 = nn.Parameter(torch.empty(E, d, h))
        self.w3 = nn.Parameter(torch.empty(E, d, h))
        self.w2 = nn.Parameter(torch.empty(E, h, d))
        self.shared = nn.ModuleList([SwiGLU(d, h) for _ in range(e["num_shared_experts"])])
        self.register_buffer("expert_bias", torch.zeros(E))                    # used only by balance_method = "bias"
        self.register_buffer("load_sum", torch.zeros(E), persistent=False)
        self.n_acc = 0

    def forward(self, x):
        B, T, D = x.shape
        xf = x.reshape(-1, D)                                                  # [N, D]
        logits = self.router(xf).float()                                       # [N, E]
        scores = logits.softmax(-1) if self.router_type == "softmax" else torch.sigmoid(logits)
        pick = scores.detach() + (self.expert_bias if self.balance == "bias" else 0.0)   # bias steers selection only
        sel = F.one_hot(pick.topk(self.k, dim=-1).indices, self.E).sum(-2).to(scores.dtype)  # [N, E], 1.0 if chosen
        gate = scores * sel
        if self.normalize:
            gate = gate / (gate.sum(-1, keepdim=True) + 1e-9)
        hid = F.silu(torch.einsum("nd,edh->enh", xf, self.w1)) * torch.einsum("nd,edh->enh", xf, self.w3)
        y = torch.einsum("ne,end->nd", gate, torch.einsum("enh,ehd->end", hid, self.w2))
        for sh in self.shared:
            y = y + sh(xf)
        probs = scores / scores.sum(-1, keepdim=True)                          # routing distribution for the metrics
        load = sel.mean(0) / self.k                                            # share of assignments per expert, sums to 1
        if self.training and self.balance == "bias":
            self.load_sum += load.detach()
            self.n_acc += 1
        stats = {
            "lb": self.E * (load * probs.mean(0)).sum(),
            "z": torch.logsumexp(logits, -1).pow(2).mean(),
            "entropy": -(probs * torch.log(probs + 1e-9)).sum(-1).mean().detach(),
            "load_cv": (load.std(unbiased=False) / load.mean()).detach(),
            "load_max": load.max().detach(),
        }
        return y.view(B, T, D), stats

    @torch.no_grad()
    def update_bias(self, rate):
        """Loss-free balancing: raise the bias of under-used experts, lower it for over-used ones."""
        if self.n_acc:
            self.expert_bias += rate * torch.sign(1.0 / self.E - self.load_sum / self.n_acc)
            self.load_sum.zero_()
            self.n_acc = 0


class Block(nn.Module):
    def __init__(self, m, e, use_moe, dense_hidden):
        super().__init__()
        self.n1, self.n2 = RMSNorm(m["d_model"], m["norm_eps"]), RMSNorm(m["d_model"], m["norm_eps"])
        self.attn = Attention(m)
        self.use_moe = use_moe
        self.ffn = MoE(m, e) if use_moe else SwiGLU(m["d_model"], dense_hidden)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        if self.use_moe:
            y, st = self.ffn(self.n2(x))
            return x + y, st
        return x + self.ffn(self.n2(x)), None


class MTPHead(nn.Module):
    """Residual MLP in front of the shared output matrix: predicts one further-ahead token."""
    def __init__(self, d, hidden, eps):
        super().__init__()
        self.norm, self.fc1, self.fc2 = RMSNorm(d, eps), nn.Linear(d, hidden, bias=False), nn.Linear(hidden, d, bias=False)

    def forward(self, h):
        return h + self.fc2(F.silu(self.fc1(self.norm(h))))


class TinyMoELM(nn.Module):
    def __init__(self, cfg, tok=None):
        super().__init__()
        m, e, emb, mt = cfg["model"], cfg["moe"], cfg["embedding"], cfg["mtp"]
        d, V = m["d_model"], m["vocab_size"]
        self.dense_baseline = e["dense_baseline"]
        if self.dense_baseline:    # no MoE: one dense FFN as wide as the experts that are active per token
            dense_hidden = e["expert_hidden_dim"] * (e["top_k"] + e["num_shared_experts"])
        else:
            dense_hidden = m["dense_ffn_hidden_dim"]
        plan = []
        for i in range(m["n_layers"]):
            is_moe = (not self.dense_baseline) and i >= e["first_k_dense"] and (i - e["first_k_dense"]) % e["moe_layer_freq"] == 0
            plan.append(is_moe)
        self.layer_plan = plan
        self.blocks = nn.ModuleList([Block(m, e, use_moe, dense_hidden) for use_moe in plan])
        self.norm_f = RMSNorm(d, m["norm_eps"])
        self.head = nn.Linear(d, V, bias=False)
        self.mtp = nn.ModuleList([MTPHead(d, mt["mtp_hidden_mult"] * d, m["norm_eps"]) for _ in range(mt["mtp_depth"])])
        std = m["init_std"]
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, 0.0, std)
            elif isinstance(mod, MoE):
                for w in (mod.w1, mod.w3, mod.w2):
                    nn.init.normal_(w, 0.0, std)
        if emb["embedding_type"] == "kronecker":
            from kronecker_embeddings import KroneckerEmbedding
            kron_vocab = max(tok.get_vocab().values()) + 1
            assert kron_vocab <= V, f"tokenizer has {kron_vocab} ids but model.vocab_size is {V}"
            self.emb = KroneckerEmbedding(vocab_size=kron_vocab, d_model=d, tokenizer=tok, char_dim=emb["kron_char_dim"],
                                          pos_dim=emb["kron_pos_dim"], mode=emb["kron_mode"])
        else:
            self.emb = nn.Embedding(V, d)
            nn.init.normal_(self.emb.weight, 0.0, std)
            if emb["tie_embeddings"]:
                self.head.weight = self.emb.weight

    def forward(self, idx, want_mtp=False):
        x, stats = self.emb(idx), []
        for b in self.blocks:
            x, st = b(x)
            if st is not None:
                stats.append(st)
        h = self.norm_f(x)
        logits = self.head(h)
        mtp_logits = [self.head(mod(h)) for mod in self.mtp] if want_mtp else []
        return logits, mtp_logits, stats

    def moe_layers(self):
        return [b.ffn for b in self.blocks if b.use_moe]

    def param_report(self, k):
        total = sum(p.numel() for p in self.parameters())
        expert = sum(p.numel() for ml in self.moe_layers() for p in (ml.w1, ml.w3, ml.w2))
        active = total - int(sum((ml.w1.numel() + ml.w3.numel() + ml.w2.numel()) * (1 - k / ml.E) for ml in self.moe_layers()))
        return total, active, expert, sum(p.numel() for p in self.emb.parameters())


def router_agg(stats):
    if not stats:
        return None
    mean = lambda key: torch.stack([s[key] for s in stats]).mean()
    return {k: mean(k) for k in ("lb", "z", "entropy", "load_cv", "load_max")}


# --------------------------------------------------------------------------- data
def load_tokens(cfg, vocab):
    path = resolve(cfg["data"]["token_file"])
    if not path.exists():
        sys.exit(f"{path} not found. Run first:  python harness/prepare_data.py")
    tokens = np.load(path)
    assert tokens.max() < vocab, "token file has ids outside the model vocabulary"
    data = torch.from_numpy(tokens.astype(np.int64))
    n_val = max(1, int(len(data) * cfg["data"]["val_fraction"]))
    return data[:-n_val], data[-n_val:]


def get_batch(data, batch_size, seq_len, gen):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,), generator=gen)
    offs = ix[:, None] + torch.arange(seq_len)[None, :]
    return data[offs], data[offs + 1]


def load_tokenizer(cfg):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cfg["tokenizer"]["name_or_path"])


# --------------------------------------------------------------------------- logging (swap for G8's logger later)
class CsvLogger:
    def __init__(self, path, names):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.names, self.f = names, open(path, "w")
        self.f.write(",".join(names) + "\n")

    def log_row(self, row):
        out = []
        for n in self.names:
            v = row.get(n)
            out.append("" if v is None else (str(v) if n == "step" else f"{v:.9g}"))
        self.f.write(",".join(out) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


# --------------------------------------------------------------------------- training
def lr_at(step, t):
    if step <= t["warmup_steps"]:
        return t["lr"] * step / max(1, t["warmup_steps"])
    prog = (step - t["warmup_steps"]) / max(1, t["steps"] - t["warmup_steps"])
    return t["lr"] * (t["min_lr_ratio"] + (1 - t["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * prog)))


def ce(logits, y):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


@torch.no_grad()
def evaluate(model, val, t, seed, device):
    model.eval()
    gen = torch.Generator().manual_seed(seed + 2)   # same val batches at every eval
    losses = []
    for _ in range(t["eval_batches"]):
        x, y = get_batch(val, t["batch_size"], t["seq_len"], gen)
        logits, _, _ = model(x.to(device))
        losses.append(ce(logits, y.to(device)).item())
    model.train()
    return sum(losses) / len(losses)


def run_training(cfg, seed, device, log_path, tok=None, data=None, print_every=10, warn_only=False,
                 min_drop=0.10, quiet=False):
    """Train once. Returns a summary dict (also written to <log>.summary.json)."""
    t, m, e, mt = cfg["train"], cfg["model"], cfg["moe"], cfg["mtp"]
    say = (lambda *a, **k: None) if quiet else print
    if tok is None and cfg["embedding"]["embedding_type"] == "kronecker":
        tok = load_tokenizer(cfg)
    train_data, val_data = data if data is not None else load_tokens(cfg, m["vocab_size"])

    set_determinism(seed, warn_only)
    model = TinyMoELM(cfg, tok).to(device)       # built after seeding so init is reproducible
    total, active, expert, embed = model.param_report(e["top_k"])
    say(f"device={device} seed={seed} embedding={cfg['embedding']['embedding_type']} dense_baseline={model.dense_baseline} "
        f"layers={''.join('M' if x else 'D' for x in model.layer_plan)} (M=MoE, D=dense)\n"
        f"params total={total:,} active/token={active:,} experts={expert:,} input-embedding={embed:,} "
        f"(ln(vocab)={math.log(m['vocab_size']):.2f})")

    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": t["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=t["lr"], betas=(t["beta1"], t["beta2"]))
    logger, syslog = CsvLogger(log_path, METRIC_NAMES), CsvLogger(Path(log_path).with_suffix(".system.csv"), SYSTEM_METRIC_NAMES)
    gen = torch.Generator().manual_seed(seed + 1)
    accum, tok_per_step = t["grad_accum_steps"], t["batch_size"] * t["grad_accum_steps"] * t["seq_len"]
    depth, aux_on = mt["mtp_depth"], e["balance_method"] == "aux_loss"
    hist = {k: [] for k in ("loss_ntp", "grad_norm", "entropy", "load_max", "load_cv", "tps")}
    t0, row = time.time(), {}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    for step in range(1, t["steps"] + 1):
        t_step = time.perf_counter()
        lr = lr_at(step, t)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        acc = {k: 0.0 for k in ("loss", "ntp", "mtp", "lb", "z", "entropy", "cv", "max")}
        have_router = False
        for _ in range(accum):
            x, y = get_batch(train_data, t["batch_size"], t["seq_len"], gen)
            x, y = x.to(device), y.to(device)
            logits, mtp_logits, stats = model(x, want_mtp=depth > 0)
            ntp = ce(logits, y)
            loss = ntp
            if mtp_logits:   # head j predicts the token j positions further ahead than the next token
                mtp = torch.stack([ce(lg[:, :-j], y[:, j:]) for j, lg in enumerate(mtp_logits, 1)]).mean()
                loss = loss + mt["mtp_loss_coef"] * mtp
                acc["mtp"] += mtp.item() / accum
            ra = router_agg(stats)
            if ra is not None:
                have_router = True
                if aux_on:
                    loss = loss + e["load_balance_coef"] * ra["lb"]
                loss = loss + e["router_z_loss_coef"] * ra["z"]
                for k, key in (("lb", "lb"), ("z", "z"), ("entropy", "entropy"), ("cv", "load_cv"), ("max", "load_max")):
                    acc[k] += ra[key].item() / accum
            (loss / accum).backward()
            acc["loss"] += loss.item() / accum
            acc["ntp"] += ntp.item() / accum
        if not math.isfinite(acc["loss"]):
            logger.close(); syslog.close()
            sys.exit(f"FAIL: loss is {acc['loss']} at step {step} (NaN/inf)")
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        opt.step()
        if e["balance_method"] == "bias":
            for ml in model.moe_layers():
                ml.update_bias(e["bias_update_rate"])

        row = {"step": step, "loss": acc["loss"], "loss_ntp": acc["ntp"], "grad_norm": gn.item(), "lr": lr}
        if depth > 0:
            row["loss_mtp"] = acc["mtp"]
        if have_router:
            row.update(load_balance_loss=acc["lb"], router_z_loss=acc["z"], router_entropy=acc["entropy"],
                       expert_load_cv=acc["cv"], expert_load_max_frac=acc["max"])
        if step % t["eval_interval"] == 0 or step == t["steps"]:
            row["val_loss"] = evaluate(model, val_data, t, seed, device)
        logger.log_row(row)
        dt = time.perf_counter() - t_step
        tps = tok_per_step / dt
        syslog.log_row({"step": step, "tokens_per_second": tps,
                        "gpu_alloc_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0})
        hist["loss_ntp"].append(row["loss_ntp"]); hist["grad_norm"].append(row["grad_norm"]); hist["tps"].append(tps)
        if have_router:
            hist["entropy"].append(acc["entropy"]); hist["load_max"].append(acc["max"]); hist["load_cv"].append(acc["cv"])
        if step % print_every == 0 or step == 1:
            extra = f"mtp {row['loss_mtp']:.4f} " if depth > 0 else ""
            ent = f"ent {acc['entropy']:.3f} cv {acc['cv']:.3f} " if have_router else ""
            say(f"step {step:5d} loss {row['loss']:.4f} ntp {row['loss_ntp']:.4f} {extra}{ent}{tps:,.0f} tok/s")
    logger.close(); syslog.close()

    n = max(1, min(10, len(hist["loss_ntp"]) // 4))
    first, last = float(np.mean(hist["loss_ntp"][:n])), float(np.mean(hist["loss_ntp"][-n:]))
    tail = lambda xs: float(np.mean(xs[-n:])) if xs else None
    summary = {"seed": seed, "steps": t["steps"], "loss_ntp_step1": hist["loss_ntp"][0], "loss_ntp_first": first,
               "loss_ntp_last": last, "val_loss": row.get("val_loss"), "ok": bool(last < (1 - min_drop) * first),
               "router_entropy": tail(hist["entropy"]), "expert_load_max_frac": tail(hist["load_max"]),
               "expert_load_cv": tail(hist["load_cv"]), "grad_norm_p50": float(np.percentile(hist["grad_norm"], 50)),
               "grad_norm_p99": float(np.percentile(hist["grad_norm"], 99)), "tokens_per_second": float(np.mean(hist["tps"])),
               "params_total": total, "params_active": active, "ln_vocab": math.log(m["vocab_size"]),
               "num_experts": e["num_experts"], "wall_s": time.time() - t0}
    Path(log_path).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--seed", type=int, help="overrides train.seed")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--log", help="CSV path (default runs/<config>_seed<seed>.csv)")
    ap.add_argument("--set", action="append", default=[], metavar="section.key=value")
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument("--min-drop", type=float, default=0.10, help="PASS needs loss_ntp to fall by this fraction")
    ap.add_argument("--warn-only", action="store_true",
                    help="if an op has no deterministic version, warn instead of crash (determinism NOT guaranteed)")
    ap.add_argument("--list-metrics", action="store_true")
    args = ap.parse_args()

    if args.list_metrics:
        for n in METRIC_NAMES:
            print(f"{n:22s} {METRIC_HELP[n]}")
        print("\nsystem metrics (separate .system.csv, not reproducible): " + ", ".join(SYSTEM_METRIC_NAMES[1:]))
        return
    if not args.config:
        ap.error("--config is required")

    cfg = load_config(args.config, args.set)
    seed = args.seed if args.seed is not None else cfg["train"]["seed"]
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    log_path = Path(args.log) if args.log else ROOT / f"runs/{Path(args.config).stem}_seed{seed}.csv"
    s = run_training(cfg, seed, device, log_path, print_every=args.print_every, warn_only=args.warn_only, min_drop=args.min_drop)
    print(f"\nloss_ntp first -> last: {s['loss_ntp_first']:.4f} -> {s['loss_ntp_last']:.4f}  "
          f"{'PASS (loss went down, no NaN)' if s['ok'] else f'FAIL (loss did not fall by >= {args.min_drop:.0%})'}")
    print(f"RESULT config={Path(args.config).name} seed={seed} steps={s['steps']} first={s['loss_ntp_first']:.4f} "
          f"last={s['loss_ntp_last']:.4f} final_val={s['val_loss']:.4f} params_total={s['params_total']} "
          f"params_active={s['params_active']} wall_s={s['wall_s']:.1f}")
    print(f"logs: {log_path}, {log_path.with_suffix('.system.csv')}, {log_path.with_suffix('.summary.json')}")
    sys.exit(0 if s["ok"] else 1)


if __name__ == "__main__":
    main()
