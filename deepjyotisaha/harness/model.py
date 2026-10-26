"""The tiny test model: a decoder-only transformer with the Kronecker input embedding and, optionally,
mixture-of-experts feed-forward blocks. It follows the settled V5 decisions at a size that trains in
minutes on a laptop CPU.

    token ids ──> KroneckerEmbedding (fixed byte codec + one Linear)            [S7, call 25 Oct]
              ──> N blocks:  x + Attention(RMSNorm(x))                           GQA + rotary positions [S8]
                             x + FFN(RMSNorm(x))       first_k_dense_layers dense (SwiGLU), the rest MoE [S14]
              ──> RMSNorm ──> output head (dense)  -> next-token logits          [S9; KE2 head is open, G4]
                          ──> MTP head (dense)     -> logits for the token after next  [S9]

The mixture-of-experts block (MoE):
  - a router scores every token for every expert (sigmoid scores, computed in float32);
  - a per-expert bias is added to the scores only to choose the top-k experts (the "aux-loss-free"
    balancing of S14); the gate weights use the scores without the bias, normalised over the k chosen;
  - every chosen (token, expert) pair is computed: no capacity limit, no dropped tokens (dropless);
  - a small sequence-level balance loss (coefficient 0.0001) is added to the loss;
  - one shared expert is applied to every token;
  - after each optimizer step the bias moves by bias_update_rate towards balance (update_expert_bias).
The expert outputs are written into a buffer with one slot per (token, choice) and summed over the k
choices, instead of being added with scatter/index_add. Each slot is written once, which keeps the forward
and backward passes deterministic on CPU and GPU.
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # the installed package if available, else the vendored copy (third_party/, Apache 2.0)
    from kronecker_embeddings import KroneckerEmbedding
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party"))
    from kronecker_embeddings import KroneckerEmbedding


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rotary_tables(seq_len, head_dim, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    ang = torch.outer(torch.arange(seq_len, device=device, dtype=torch.float32), inv)
    return ang.cos(), ang.sin()


def apply_rotary(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    """Grouped-query attention: num_heads query heads share num_kv_heads key/value heads."""

    def __init__(self, d, n_heads, n_kv, head_dim):
        super().__init__()
        self.n_heads, self.n_kv, self.head_dim = n_heads, n_kv, head_dim
        self.q = nn.Linear(d, n_heads * head_dim, bias=False)
        self.k = nn.Linear(d, n_kv * head_dim, bias=False)
        self.v = nn.Linear(d, n_kv * head_dim, bias=False)
        self.o = nn.Linear(n_heads * head_dim, d, bias=False)

    def forward(self, x, cos, sin):
        B, L, _ = x.shape
        q = self.q(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, L, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, L, self.n_kv, self.head_dim).transpose(1, 2)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        rep = self.n_heads // self.n_kv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, L, -1))


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoE(nn.Module):
    def __init__(self, d, n_experts, top_k, expert_hidden, n_shared, scoring, norm_topk, scale, aux_coef):
        super().__init__()
        self.E, self.k, self.scoring, self.norm_topk, self.scale, self.aux_coef = (
            n_experts, top_k, scoring, norm_topk, scale, aux_coef)
        self.router = nn.Linear(d, n_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(n_experts))   # used only to choose experts
        std = 0.02
        self.w_gate = nn.Parameter(torch.randn(n_experts, d, expert_hidden) * std)
        self.w_up = nn.Parameter(torch.randn(n_experts, d, expert_hidden) * std)
        self.w_down = nn.Parameter(torch.randn(n_experts, expert_hidden, d) * std)
        self.shared = SwiGLU(d, expert_hidden * n_shared) if n_shared else None
        self.step_counts = torch.zeros(n_experts)   # tokens per expert, summed over the micro-batches of a step
        self.last_aux = torch.zeros(())

    def forward(self, x):
        B, L, d = x.shape
        h = x.reshape(-1, d)
        T = h.size(0)
        logits = h.float() @ self.router.weight.float().t()               # router in float32 [S14]
        scores = torch.sigmoid(logits) if self.scoring == "sigmoid" else logits.softmax(-1)
        chosen = (scores + self.expert_bias).topk(self.k, dim=-1).indices  # bias only for choosing
        w = scores.gather(1, chosen)
        if self.norm_topk:
            w = w / w.sum(-1, keepdim=True)
        w = (w * self.scale).to(h.dtype)

        flat_e = chosen.reshape(-1)                                         # (T*k,)
        flat_tok = torch.arange(T, device=h.device).repeat_interleave(self.k)
        order = torch.sort(flat_e, stable=True).indices
        counts = F.one_hot(flat_e, self.E).sum(0)                          # tokens per expert (deterministic)
        out = h.new_zeros(T * self.k, d)
        start = 0
        for e, n in enumerate(counts.tolist()):                             # dropless: every pair is computed
            if n:
                slots = order[start:start + n]
                xe = h.index_select(0, flat_tok.index_select(0, slots))
                ye = (F.silu(xe @ self.w_gate[e]) * (xe @ self.w_up[e])) @ self.w_down[e]
                out = out.index_put((slots,), ye)                           # each slot written once
                start += n
        y = (out.view(T, self.k, d) * w.unsqueeze(-1)).sum(1)
        if self.shared is not None:
            y = y + self.shared(h)

        # sequence-level balance loss: per sequence, sum_i f_i * P_i  [S14, DeepSeek-V3 style]
        affin = scores / scores.sum(-1, keepdim=True)
        onehot = F.one_hot(chosen, self.E).sum(1).float()                  # (T, E) chosen count per token
        f = onehot.view(B, L, self.E).sum(1) * (self.E / (self.k * L))     # (B, E)
        P = affin.view(B, L, self.E).mean(1)                               # (B, E)
        self.last_aux = self.aux_coef * (f * P).sum(-1).mean()
        if self.training:  # load is counted over the whole step, all micro-batches together [S14]
            self.step_counts = self.step_counts + counts.detach().float().cpu()
        return y.view(B, L, d)

    @torch.no_grad()
    def update_expert_bias(self, rate):
        """After an optimizer step: raise the bias of under-loaded experts, lower it for over-loaded ones,
        using the token counts of the whole step. Returns those counts and starts a new count."""
        load = self.step_counts
        self.expert_bias += rate * torch.sign(load.mean() - load).to(self.expert_bias.device)
        self.step_counts = torch.zeros(self.E)
        return load


class Block(nn.Module):
    def __init__(self, cfg, use_moe):
        super().__init__()
        m, a, moe = cfg["model"], cfg["attention"], cfg["moe"]
        d = m["hidden_size"]
        self.norm1, self.norm2 = RMSNorm(d, m["rms_norm_eps"]), RMSNorm(d, m["rms_norm_eps"])
        self.attn = Attention(d, m["num_attention_heads"], m["num_key_value_heads"], m["head_dim"])
        if use_moe:
            self.ffn = MoE(d, moe["num_experts"], moe["num_experts_per_tok"], moe["moe_intermediate_size"],
                           moe["num_shared_experts"], moe["router_scoring"], moe["norm_topk_prob"],
                           moe["routed_scaling_factor"], moe["seq_aux_loss_coef"])
        else:
            self.ffn = SwiGLU(d, m["intermediate_size"])

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.ffn(self.norm2(x))


class TinyModel(nn.Module):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        m, e, moe, out = cfg["model"], cfg["embedding"], cfg["moe"], cfg["output_head"]
        d = m["hidden_size"]
        self.cfg = cfg
        self.embed = KroneckerEmbedding(vocab_size=tokenizer.vocab_size, d_model=d, tokenizer=tokenizer,
                                        char_dim=e["char_dim"], pos_dim=e["pos_dim"], mode=e["mode"],
                                        projection_init=e["projection_init"],
                                        length_normalize=e["length_normalize"], z_normalize=e["z_normalize"])
        n_moe_layers = moe["num_experts"] and m["num_hidden_layers"] - m["first_k_dense_layers"]
        self.blocks = nn.ModuleList(Block(cfg, use_moe=bool(moe["num_experts"]) and i >= m["first_k_dense_layers"])
                                    for i in range(m["num_hidden_layers"]))
        self.n_moe_layers = n_moe_layers
        self.norm = RMSNorm(d, m["rms_norm_eps"])
        self.head = nn.Linear(d, tokenizer.vocab_size, bias=False)
        self.mtp_heads = nn.ModuleList(nn.Sequential(RMSNorm(d, m["rms_norm_eps"]),
                                                     nn.Linear(d, tokenizer.vocab_size, bias=False))
                                       for _ in range(out["mtp_heads"]))
        self._init(m["initializer_range"])
        self.rope_theta = cfg["attention"]["rope_theta"]

    def _init(self, std):
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear) and not name.startswith("embed"):   # the Kronecker projection keeps its own init
                nn.init.normal_(mod.weight, mean=0.0, std=std)

    def moe_layers(self):
        return [b.ffn for b in self.blocks if isinstance(b.ffn, MoE)]

    def forward(self, ids):
        """ids: (B, L). Returns the final hidden state (B, L, d)."""
        L = ids.size(1)
        cos, sin = rotary_tables(L, self.cfg["model"]["head_dim"], self.rope_theta, ids.device)
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x, cos, sin)
        return self.norm(x)


def chunked_cross_entropy(hidden, head, targets, chunk=1024, z_coef=0.0):
    """Sum of token losses (nats) without building the full logits tensor at once [S9]. Also returns the
    z-loss sum (log of the softmax normaliser, squared), which keeps the logits from drifting [PaLM]."""
    h = hidden.reshape(-1, hidden.size(-1))
    t = targets.reshape(-1)
    loss_sum = h.new_zeros((), dtype=torch.float32)
    z_sum = h.new_zeros((), dtype=torch.float32)
    for i in range(0, h.size(0), chunk):
        logits = head(h[i:i + chunk]).float()
        loss_sum = loss_sum + F.cross_entropy(logits, t[i:i + chunk], reduction="sum")
        if z_coef:
            z_sum = z_sum + torch.logsumexp(logits, -1).pow(2).sum()
    return loss_sum, z_sum


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    moe = model.moe_layers()
    routed = sum(m.w_gate.numel() + m.w_up.numel() + m.w_down.numel() for m in moe)
    active_routed = sum((m.w_gate.numel() + m.w_up.numel() + m.w_down.numel()) * m.k // m.E for m in moe)
    return {"total": total, "active": total - routed + active_routed,
            "embedding_projection": model.embed.projection.weight.numel(),
            "output_head": model.head.weight.numel()}
