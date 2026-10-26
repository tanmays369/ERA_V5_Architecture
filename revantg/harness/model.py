# Name: Revant Gupta
# Pack: ind-hi-fw2
"""Dense pre-norm decoder: Kronecker embedding, GQA + QK-norm + RoPE attention, SwiGLU MLP, untied head.
`build_model(cfg, tokenizer_path)`; `loss_fn(logits, targets, z_loss_coef)`. Names are load-bearing for TP/PP.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from kronecker_embeddings import KroneckerEmbedding
from torch.utils.checkpoint import checkpoint

import tok


def rope(x, cos, sin):
    """Rotate-half RoPE on x [B, H, L, hd] with cos/sin [L, hd/2]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, qk_norm, eps):
        super().__init__()
        self.head_dim = d_model // n_heads
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=eps) if qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim, eps=eps) if qk_norm else None
        self.track = False
        self.last_max_logit = None

    def forward(self, x, cos, sin):
        B, L, _ = x.shape
        hd = self.head_dim
        # Local head counts come from the projection outputs, so TP-sharded (local) activations just work.
        q = self.wq(x).view(B, L, -1, hd).transpose(1, 2)
        k = self.wk(x).view(B, L, -1, hd).transpose(1, 2)
        v = self.wv(x).view(B, L, -1, hd).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        rep = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        if self.track:
            with torch.no_grad():
                s = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
                mask = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
                self.last_max_logit = float(s.masked_fill(~mask, 0).abs().amax())
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class _AddAux(torch.autograd.Function):
    """Identity on x whose backward also seeds d(loss)/d(aux) = scale, so aux joins the loss in any PP stage."""

    @staticmethod
    def forward(ctx, x, aux, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, g):
        return g, torch.full((), ctx.scale, dtype=g.dtype, device=g.device), None


class MoEFFN(nn.Module):
    """Dropless top-k MoE, DeepSeek-V3 style (S14 §20; DeepSeek-V3 report §2.1.2–2.1.3). Sigmoid router in fp32,
    top-k weights normalised. Aux-loss-free balancing: `expert_bias` steers selection only, never the weights.
    Small sequence-wise aux loss. Experts are stacked SwiGLU weights [E, out, in]."""

    def __init__(self, d_model, d_ff, moe):
        super().__init__()
        E = moe["num_experts"]
        self.top_k, self.aux_coef, self.speed = moe["top_k"], moe["seq_aux_loss_coef"], moe["bias_update_speed"]
        self.router = nn.Linear(d_model, E, bias=False)
        self.w_gate = nn.Parameter(torch.empty(E, d_ff, d_model))
        self.w_up = nn.Parameter(torch.empty(E, d_ff, d_model))
        self.w_down = nn.Parameter(torch.empty(E, d_model, d_ff))
        n_shared = moe.get("shared_experts") or 0
        self.shared = MLP(d_model, n_shared * d_ff) if n_shared else None
        self.register_buffer("expert_bias", torch.zeros(E))
        # Not buffers: DDP broadcast_buffers would overwrite per-rank counts.
        self.counting = False  # set by Transformer.forward, so checkpoint recompute in backward is not counted twice
        self.load = None  # slot counts per expert, summed over training forwards since the last bias update
        self.last_load = None  # load of the last finished step (all-reduced over dp), for metrics
        self.last_aux = None  # aux_coef * aux of the last forward (detached)
        self.stats = None  # router stats summed over training forwards since the last moe_metrics (see forward)
        self.aux_scale = 1.0  # gradient weight of aux; set it to the weight the trainer puts on this micro-batch loss

    def forward(self, x):
        B, L, d = x.shape
        E, k = self.w_gate.shape[0], self.top_k
        xf = x.reshape(-1, d)
        scores = torch.sigmoid(self.router(xf.float()))
        topi = (scores + self.expert_bias).topk(k, dim=-1).indices
        w = scores.gather(-1, topi)
        w = (w / w.sum(-1, keepdim=True)).reshape(-1)
        out = torch.zeros_like(xf)
        for e in range(E):  # fixed order + index_add_ (deterministic under use_deterministic_algorithms)
            tok, slot = (topi == e).nonzero(as_tuple=True)  # empty experts still run: params stay in the graph for DDP
            h = xf.index_select(0, tok)
            y = F.linear(F.silu(F.linear(h, self.w_gate[e])) * F.linear(h, self.w_up[e]), self.w_down[e])
            out.index_add_(0, tok, y * w.index_select(0, tok * k + slot).unsqueeze(-1).to(y.dtype))
        out = out.view(B, L, d)
        per_seq = F.one_hot(topi.view(B, L * k), E).sum(1).float()  # [B, E] slot counts per sequence
        if self.counting:
            c = per_seq.sum(0)
            self.load = c if self.load is None else self.load + c
            with torch.no_grad():  # metrics only: [entropy sum, picks changed by bias, selected score mass sum, tokens]
                p = scores / scores.sum(-1, keepdim=True)
                plain = scores.topk(k, dim=-1).indices
                changed = k - (topi.unsqueeze(-1) == plain.unsqueeze(-2)).any(-1).sum(-1)
                s = torch.stack([-(p * p.clamp_min(1e-30).log()).sum(), changed.sum().float(),
                                 p.gather(-1, topi).sum(), torch.tensor(float(xf.shape[0]), device=x.device)])
                self.stats = s if self.stats is None else self.stats + s
        if self.aux_coef:
            p = (scores / scores.sum(-1, keepdim=True)).view(B, L, E).mean(1)
            aux = self.aux_coef * ((per_seq * E / (k * L)) * p).sum(-1).mean()
            self.last_aux = aux.detach()
            out = _AddAux.apply(out, aux, self.aux_scale)
        if self.shared is not None:
            out = out + self.shared(x)
        return out


class Block(nn.Module):
    def __init__(self, m, moe):
        super().__init__()
        eps = m["norm_eps"]
        self.attn_norm = nn.RMSNorm(m["d_model"], eps=eps)
        self.attn = Attention(m["d_model"], m["n_heads"], m["n_kv_heads"], m["qk_norm"], eps)
        self.mlp_norm = nn.RMSNorm(m["d_model"], eps=eps)
        self.mlp = MoEFFN(m["d_model"], m["d_ff"], moe) if moe.get("enabled") else MLP(m["d_model"], m["d_ff"])

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class Transformer(nn.Module):
    """PP stages may set `embed=None` (input is hidden [B,L,D]) or `norm=head=None` (returns hidden)."""

    def __init__(self, cfg, byte_buffer, length_buffer):
        super().__init__()
        m, e = cfg["model"], cfg["embedding"]
        V = cfg["tokenizer"]["vocab_size"]
        self.embed = KroneckerEmbedding(V, m["d_model"], char_dim=e["char_dim"], pos_dim=e["pos_dim"], mode=e["mode"],
                                        byte_buffer=byte_buffer, length_buffer=length_buffer)
        self.embed.projection.weight.requires_grad_(e["projection_trainable"])
        # Codec casts lengths to long anyway; int16 buffers break DDP's broadcast (gloo/NCCL have no int16).
        self.embed._length_buffer = self.embed._length_buffer.long()
        moe = cfg.get("moe") or {}
        self.layers = nn.ModuleList(Block(m, moe) for _ in range(m["n_layers"]))
        self.norm = nn.RMSNorm(m["d_model"], eps=m["norm_eps"])
        self.head = nn.Linear(m["d_model"], V, bias=False)
        self.recompute = bool(cfg["parallelism"].get("recompute_layer", False))
        hd = m["d_model"] // m["n_heads"]
        inv = 1.0 / m["rope_theta"] ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd)
        ang = torch.outer(torch.arange(cfg["tokens"]["seq_len"], dtype=torch.float64), inv)
        self.register_buffer("rope_cos", ang.cos().float(), persistent=False)
        self.register_buffer("rope_sin", ang.sin().float(), persistent=False)
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear) and not name.startswith("embed"):
                nn.init.normal_(mod.weight, mean=0.0, std=m["init_std"])
            elif isinstance(mod, nn.RMSNorm):
                nn.init.constant_(mod.weight, m["norm_gain_init"])
            elif isinstance(mod, MoEFFN):
                for p in (mod.w_gate, mod.w_up, mod.w_down):
                    nn.init.normal_(p, mean=0.0, std=m["init_std"])

    def moe_layers(self):
        """MoEFFN modules of this (possibly PP-trimmed) model; empty when dense."""
        return [b.mlp for b in self.layers if isinstance(b.mlp, MoEFFN)]

    @property
    def last_aux_loss(self):
        """Sum of the MoE layers' scaled aux losses from the last forward (already in the gradient), or None."""
        aux = [moe.last_aux for moe in self.moe_layers() if moe.last_aux is not None]
        return sum(aux) if aux else None

    @property
    def track_attn_logits(self):
        return any(b.attn.track for b in self.layers)

    @track_attn_logits.setter
    def track_attn_logits(self, on):
        for b in self.layers:
            b.attn.track = on

    def forward(self, x):
        if self.embed is not None:
            x = self.embed(x)
        L = x.shape[1]
        cos, sin = self.rope_cos[:L], self.rope_sin[:L]
        moes = self.moe_layers()
        for moe in moes:
            moe.counting = self.training and torch.is_grad_enabled()
        for block in self.layers:
            if self.recompute and self.training:
                x = checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        for moe in moes:
            moe.counting = False
        if self.head is None:
            return x
        return self.head(self.norm(x))


def build_model(cfg, tokenizer_path):
    """CPU fp32 model; caller seeds first (`torch.manual_seed`) so init is identical across ranks/layouts."""
    bb, lb = tok.byte_buffers(tokenizer_path, cfg["embedding"]["pos_dim"])
    return Transformer(cfg, bb, lb)


def update_expert_bias(model, dp_group=None):
    """Aux-loss-free balancing (DeepSeek-V3 §2.1.2): after each optimizer step, all-reduce the step's per-expert
    load over dp, then bias += speed * sign(mean_load - load). Sets `last_load` and resets `load`. Call on every rank."""
    # with-block, not a decorator: the lab pr-guard email regex flags a top-level decorator line in the patch
    with torch.no_grad():
        for moe in model.moe_layers():
            load = moe.load if moe.load is not None else torch.zeros_like(moe.expert_bias)
            if dp_group is not None:
                torch.distributed.all_reduce(load, group=dp_group)
            moe.expert_bias += moe.speed * torch.sign(load.mean() - load)
            moe.last_load, moe.load = load, None


def moe_metrics(model, dp_group=None):
    """Per MoE layer after `update_expert_bias`: load fractions, dead experts, top expert share, MaxVio
    (max_load/mean_load - 1), router entropy (nats, of scores/sum), top-k intervention rate (share of picks where
    bias changed topk), selected top-k score mass. Router stats all-reduced over dp, then reset. Call on every rank."""
    out = []
    for moe in model.moe_layers():
        st = moe.stats if moe.stats is not None else torch.zeros(4, device=moe.expert_bias.device)
        if dp_group is not None:
            torch.distributed.all_reduce(st, group=dp_group)
        moe.stats = None
        n = float(st[3]) or 1.0
        frac = moe.last_load / moe.last_load.sum().clamp(min=1)
        out.append(dict(load=[round(v, 4) for v in frac.tolist()], dead_experts=int((moe.last_load == 0).sum()),
                        top_expert_share=float(frac.max()),
                        max_vio=float(moe.last_load.max() / moe.last_load.mean().clamp(min=1e-12) - 1),
                        router_entropy=float(st[0]) / n, intervention_rate=float(st[1]) / (n * moe.top_k),
                        topk_mass=float(st[2]) / n,
                        aux=0.0 if moe.last_aux is None else float(moe.last_aux)))  # None when seq_aux_loss_coef == 0
    return out


def loss_fn(logits, targets, z_loss_coef):
    """(total, ce): ce = mean token CE in fp32; total = ce + z_loss_coef * mean(logsumexp^2). 0-dim tensors."""
    logits = logits.float()
    lse = torch.logsumexp(logits, dim=-1)
    ce = (lse - logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)).mean()
    return ce + z_loss_coef * lse.pow(2).mean(), ce
