# Name: dharshan
# Axiom email: dharshanthaigu@gmail.com
# Pack: ind-ta-fw2
"""The tiny model. Built from the config; every switch below is read from it.

  tokens -> Kronecker embedding -> N blocks -> final norm -> untied head -> next-token loss
  block (pre)  : x + attn(norm(x)), then x + ffn(norm(x))        norm_position: pre
  block (post) : norm(x + attn(x)), then norm(x + ffn(x))        norm_position: post (the RoBERTa/BERT placement)
  attention    : grouped-query, QK-norm optional, RoPE; attention.window > 0 limits each token to the last N tokens,
                 attention.global_every = N makes every Nth layer full attention again
  ffn          : MoE (sigmoid router in fp32, top-k picked on score + bias, weights from the raw scores,
                 dropless, optional shared experts) or a dense SwiGLU when moe.enabled is false
  router       : moe.router_sharing = none (one router per layer) or all_layers (one router matrix, every layer;
                 each layer keeps its own balancing bias because load differs by layer)
  MTP head     : one extra block that predicts token t+2 from hidden state t and the embedding of token t+1,
                 using the same output head (head.mtp_heads = 1)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # older torch
    sdpa_kernel = None


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_tables(T, head_dim, theta):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    f = torch.outer(torch.arange(T).float(), inv)
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, d, n_heads, n_kv, eps, qk_norm, local):
        super().__init__()
        self.h, self.kv, self.hd, self.local = n_heads, n_kv, d // n_heads, local
        self.wq = nn.Linear(d, n_heads * self.hd, bias=False)
        self.wk = nn.Linear(d, n_kv * self.hd, bias=False)
        self.wv = nn.Linear(d, n_kv * self.hd, bias=False)
        self.wo = nn.Linear(n_heads * self.hd, d, bias=False)
        self.q_norm = RMSNorm(self.hd, eps) if qk_norm else None
        self.k_norm = RMSNorm(self.hd, eps) if qk_norm else None

    def forward(self, x, cos, sin, mask):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos[:T], sin[:T]), apply_rope(k, cos[:T], sin[:T])
        rep = self.h // self.kv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        kwargs = {"attn_mask": mask[:T, :T]} if self.local else {"is_causal": True}
        if sdpa_kernel is not None:
            with sdpa_kernel(SDPBackend.MATH):      # the math backend has a deterministic backward pass
                y = F.scaled_dot_product_attention(q, k, v, **kwargs)
        else:
            y = F.scaled_dot_product_attention(q, k, v, **kwargs)
        return self.wo(y.transpose(1, 2).reshape(B, T, self.h * self.hd))


class SwiGLU(nn.Module):
    def __init__(self, d, f):
        super().__init__()
        self.w_gate, self.w_up, self.w_down = nn.Linear(d, f, bias=False), nn.Linear(d, f, bias=False), nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Router(nn.Module):
    def __init__(self, d, n_experts):
        super().__init__()
        self.proj = nn.Linear(d, n_experts, bias=False)


class MoE(nn.Module):
    def __init__(self, d, f, n_experts, top_k, n_shared, router, seq_aux_coef):
        super().__init__()
        self.E, self.k, self.coef = n_experts, top_k, seq_aux_coef
        self.router = router                         # the same object in every layer when router_sharing = all_layers
        self.w_gate = nn.Parameter(torch.empty(n_experts, d, f))
        self.w_up = nn.Parameter(torch.empty(n_experts, d, f))
        self.w_down = nn.Parameter(torch.empty(n_experts, f, d))
        self.shared = SwiGLU(d, n_shared * f) if n_shared > 0 else None
        self.register_buffer("bias", torch.zeros(n_experts))                   # only changes which experts are picked
        self.register_buffer("load_acc", torch.zeros(n_experts), persistent=False)

    def forward(self, x):
        B, T, d = x.shape
        xf = x.reshape(-1, d)
        scores = torch.sigmoid(self.router.proj(xf.float()))                   # [N, E], fp32
        sel = torch.topk(scores + self.bias, self.k, dim=-1).indices            # [N, k]
        gates = scores.gather(1, sel)
        gates = gates / (gates.sum(-1, keepdim=True) + 1e-9)
        out = torch.zeros_like(xf)
        for e in range(self.E):
            rows, slot = torch.nonzero(sel == e, as_tuple=True)
            if rows.numel() == 0:
                continue
            xe = xf[rows]
            y = (F.silu(xe @ self.w_gate[e]) * (xe @ self.w_up[e])) @ self.w_down[e]
            out.index_add_(0, rows, y * gates[rows, slot].unsqueeze(-1))
        counts = torch.bincount(sel.reshape(-1), minlength=self.E).float()
        if self.training:
            self.load_acc += counts.detach()
        aux = xf.new_zeros(())
        if self.coef > 0 and self.training:                                    # per-sequence balance loss
            sc = scores.view(B, T, self.E)
            P = (sc / (sc.sum(-1, keepdim=True) + 1e-9)).mean(1)               # [B, E]
            fr = F.one_hot(sel.view(B, T, self.k), self.E).sum(2).float().mean(1) * (self.E / self.k)
            aux = self.coef * (fr.detach() * P).sum(-1).mean()
        y = out.view(B, T, d)
        if self.shared is not None:
            y = y + self.shared(x)
        return y, aux

    @torch.no_grad()
    def take_load(self):
        load = self.load_acc.clone()
        self.load_acc.zero_()
        return load

    @torch.no_grad()
    def update_bias(self, load, speed):
        self.bias += speed * torch.sign(load.mean() - load)


class Block(nn.Module):
    def __init__(self, d, n_heads, n_kv, eps, qk_norm, local, ffn, post):
        super().__init__()
        self.post = post
        self.norm1, self.norm2 = RMSNorm(d, eps), RMSNorm(d, eps)
        self.attn = Attention(d, n_heads, n_kv, eps, qk_norm, local)
        self.ffn = ffn

    def _ffn(self, x):
        r = self.ffn(x)
        return r if isinstance(r, tuple) else (r, x.new_zeros(()))

    def forward(self, x, cos, sin, mask):
        if self.post:
            x = self.norm1(x + self.attn(x, cos, sin, mask))
            y, aux = self._ffn(x)
            return self.norm2(x + y), aux
        x = x + self.attn(self.norm1(x), cos, sin, mask)
        y, aux = self._ffn(self.norm2(x))
        return x + y, aux


class Transformer(nn.Module):
    def __init__(self, cfg, byte_buffer, length_buffer):
        super().__init__()
        from kronecker_embeddings import KroneckerEmbedding
        g = cfg.get
        for key, only in (("model.ffn", "swiglu"), ("model.norm", "rmsnorm"), ("model.bias", False), ("model.dropout", 0.0),
                          ("model.tie_embeddings", False), ("moe.router_scoring", "sigmoid"), ("moe.router_dtype", "float32"),
                          ("moe.balancing", "aux_free_bias"), ("moe.dropless", True), ("optimizer.other_params", "adamw"),
                          ("schedule.type", "wsd"), ("precision.dtype", "float32"), ("embedding.type", "kronecker")):
            if g(key) != only:
                raise ValueError(f"{key} = {g(key)!r}: only {only!r} is implemented")
        d, L, V = g("model.d_model"), g("model.n_layers"), g("tokenizer.vocab_size")
        n_heads, n_kv, f, eps = g("model.n_heads"), g("model.n_kv_heads"), g("model.d_ff"), g("model.norm_eps")
        self.T, self.mtp = g("tokens.seq_len"), g("head.mtp_heads")
        self.z_coef, self.mtp_w = g("head.z_loss_coef"), g("head.mtp_loss_weight")
        self.post = g("model.norm_position") == "post"
        window, every = g("attention.window"), g("attention.global_every")
        moe_on = g("moe.enabled")
        E, k = (g("moe.num_experts"), g("moe.top_k")) if moe_on else (0, 0)
        self.moe_on, self.init_std = moe_on, g("model.init_std")

        self.embed = KroneckerEmbedding(vocab_size=V, d_model=d, char_dim=g("embedding.char_dim"), pos_dim=g("embedding.pos_dim"),
                                        mode=g("embedding.mode"), byte_buffer=byte_buffer, length_buffer=length_buffer,
                                        projection_init=g("embedding.projection_init"),
                                        length_normalize=g("embedding.length_normalize"), z_normalize=g("embedding.z_normalize"))
        hd = d // n_heads
        cos, sin = rope_tables(self.T, hd, g("model.rope_theta"))
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        idx = torch.arange(self.T)
        self.register_buffer("local_mask", (idx[None, :] <= idx[:, None]) & (idx[:, None] - idx[None, :] < max(window, 1)), persistent=False)

        shared_router = Router(d, E) if (moe_on and g("moe.router_sharing") == "all_layers") else None
        self.layers = nn.ModuleList()
        for i in range(L):
            local = window > 0 and not (every > 0 and (i + 1) % every == 0)
            if moe_on:
                router = shared_router if shared_router is not None else Router(d, E)
                ffn = MoE(d, f, E, k, g("moe.shared_experts"), router, g("moe.seq_aux_loss_coef"))
            else:
                ffn = SwiGLU(d, f)
            self.layers.append(Block(d, n_heads, n_kv, eps, g("model.qk_norm"), local, ffn, self.post))
        self.shared_router = shared_router
        self.final_norm = nn.Identity() if self.post else RMSNorm(d, eps)
        self.head = nn.Linear(d, V, bias=False)
        if self.mtp:
            dense_f = f * k if moe_on else f
            self.mtp_norm_h, self.mtp_norm_e = RMSNorm(d, eps), RMSNorm(d, eps)
            self.mtp_proj = nn.Linear(2 * d, d, bias=False)
            self.mtp_block = Block(d, n_heads, n_kv, eps, g("model.qk_norm"), False, SwiGLU(d, dense_f), self.post)
        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and not name.startswith("embed."):
                nn.init.normal_(m.weight, mean=0.0, std=self.init_std)
        for m in self.modules():
            if isinstance(m, MoE):
                for p in (m.w_gate, m.w_up, m.w_down):
                    nn.init.normal_(p, mean=0.0, std=self.init_std)

    def forward(self, window_ids):
        """window_ids: [B, T + 1 + mtp_heads] token ids. Returns a dict of losses."""
        T = self.T
        x_ids, tgt1 = window_ids[:, :T], window_ids[:, 1:T + 1]
        h = self.embed(x_ids)
        aux_total = h.new_zeros(())
        for blk in self.layers:
            h, aux = blk(h, self.cos, self.sin, self.local_mask)
            aux_total = aux_total + aux
        logits = self.head(self.final_norm(h)).float()
        V = logits.shape[-1]
        loss_ntp = F.cross_entropy(logits.reshape(-1, V), tgt1.reshape(-1))
        z_loss = self.z_coef * torch.logsumexp(logits, dim=-1).pow(2).mean()
        loss = loss_ntp + z_loss + aux_total
        out = {"loss_ntp": loss_ntp, "z_loss": z_loss, "moe_aux": aux_total, "loss_mtp": None}
        if self.mtp:
            e1 = self.embed(tgt1)
            m = self.mtp_proj(torch.cat([self.mtp_norm_h(h), self.mtp_norm_e(e1)], dim=-1))
            m, _ = self.mtp_block(m, self.cos, self.sin, self.local_mask)
            logits2 = self.head(self.final_norm(m)).float()
            loss_mtp = F.cross_entropy(logits2.reshape(-1, V), window_ids[:, 2:T + 2].reshape(-1))
            loss = loss + self.mtp_w * loss_mtp
            out["loss_mtp"] = loss_mtp
        out["loss"] = loss
        return out

    def moe_layers(self):
        return [b.ffn for b in self.layers if isinstance(b.ffn, MoE)]

    @torch.no_grad()
    def finish_step(self, bias_speed):
        """Read each layer's routed-token counts for this step, update its balancing bias, return the counts."""
        loads = []
        for m in self.moe_layers():
            load = m.take_load()
            m.update_bias(load, bias_speed)
            loads.append(load)
        return loads

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        active = total
        for m in self.moe_layers():
            active -= (m.E - m.k) * (m.w_gate[0].numel() + m.w_up[0].numel() + m.w_down[0].numel())
        return total, active
