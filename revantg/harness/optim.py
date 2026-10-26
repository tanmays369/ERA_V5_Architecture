# Name: Revant Gupta
# Pack: ind-hi-fw2
"""Optimizers: Muon for 2D block weights (DTensor-aware, NS in configurable dtype), AdamW for the rest; WSD schedule.
`build_optimizers(model, cfg, zero_group=None)`, `lr_at(step, cfg)`.
"""
import math

import torch

try:
    from torch.distributed.tensor import DTensor, distribute_tensor
except ImportError:  # torch < 2.5 keeps it under _tensor
    from torch.distributed._tensor import DTensor, distribute_tensor

NS_COEFFS = (3.4445, -4.7750, 2.0315)
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
MUON_NAMES = ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")


def param_groups(model, cfg):
    """Split requires_grad params: muon (2D block matrices + stacked 3D MoE experts), adamw_decay (head),
    adamw_nodecay (norms, biases, embed, MoE router — S14 §20: router stays on AdamW without decay)."""
    g = dict(muon=[], adamw_decay=[], adamw_nodecay=[])
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        parts = name.split(".")
        if "layers" in parts and ((p.ndim == 2 and parts[-2] in MUON_NAMES) or (p.ndim == 3 and parts[-1] in MUON_NAMES)):
            g["muon"].append(p)
        elif name.endswith("head.weight"):
            g["adamw_decay"].append(p)
        else:
            g["adamw_nodecay"].append(p)
    return g


def zeropower_ns(G, steps, eps, dtype):
    """Newton-Schulz quintic orthogonalization (same as torch.optim.Muon, but in `dtype`)."""
    a, b, c = NS_COEFFS
    X = G.to(dtype)
    tall = G.size(0) > G.size(1)
    if tall:
        X = X.T
    X = X / X.norm().clamp(min=eps)
    for _ in range(steps):
        A = X @ X.T
        X = torch.addmm(X, torch.addmm(A, A, A, beta=b, alpha=c), X, beta=a)
    return X.T if tall else X


class Muon(torch.optim.Optimizer):
    """torch.optim.Muon update with NS in `ns_dtype`; DTensor params are orthogonalized on the full matrix.
    3D params (stacked MoE experts [E, out, in]) are E independent matrices: NS and lr adjustment per slice."""

    def __init__(self, params, lr=1e-3, weight_decay=0.1, momentum=0.95, nesterov=True, ns_steps=5,
                 adjust_lr_fn="match_rms_adamw", ns_dtype="fp32", eps=1e-7):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        adjust_lr_fn=adjust_lr_fn, ns_dtype=ns_dtype, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for gr in self.param_groups:
            lr, m = gr["lr"], gr["momentum"]
            for p in gr["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                buf = self.state[p].setdefault("momentum_buffer", torch.zeros_like(g))
                buf.lerp_(g, 1 - m)
                u = g.lerp(buf, m) if gr["nesterov"] else buf
                if isinstance(u, DTensor):
                    full = zeropower_ns(u.full_tensor(), gr["ns_steps"], gr["eps"], DTYPES[gr["ns_dtype"]])
                    u = distribute_tensor(full.to(p.dtype), u.device_mesh, u.placements)
                elif u.ndim == 3:
                    u = torch.stack([zeropower_ns(s, gr["ns_steps"], gr["eps"], DTYPES[gr["ns_dtype"]]) for s in u])
                else:
                    u = zeropower_ns(u, gr["ns_steps"], gr["eps"], DTYPES[gr["ns_dtype"]])
                A, B = p.shape[-2:]  # DTensor .shape is the global shape
                if gr["adjust_lr_fn"] == "match_rms_adamw":
                    adj = lr * 0.2 * math.sqrt(max(A, B))
                elif gr["adjust_lr_fn"] in (None, "original"):
                    adj = lr * math.sqrt(max(1, A / B))
                else:
                    raise ValueError(f"adjust_lr_fn {gr['adjust_lr_fn']!r}")
                p.mul_(1 - lr * gr["weight_decay"])
                p.add_(u, alpha=-adj)
        return loss


def build_optimizers(model, cfg, zero_group=None):
    """[Muon, AdamW]; each wrapped in ZeroRedundancyOptimizer when `zero_group` is given. Groups carry `base_lr`."""
    o, lr = cfg["optimizer"], cfg["optimizer"]["lr"]
    g = param_groups(model, cfg)
    muon_kw = dict(lr=lr, weight_decay=o["weight_decay"], momentum=o["muon_momentum"], nesterov=o["muon_nesterov"],
                   ns_steps=o["muon_ns_steps"], adjust_lr_fn=o["muon_adjust_lr"], ns_dtype=o["muon_ns_dtype"])
    adam_kw = dict(lr=lr, betas=tuple(o["adam_betas"]), eps=o["adam_eps"], fused=False, foreach=False)
    adam_groups = [dict(params=g["adamw_decay"], weight_decay=o["weight_decay"]),
                   dict(params=g["adamw_nodecay"], weight_decay=0.0)]
    if zero_group is None:
        opts = [Muon(g["muon"], **muon_kw), torch.optim.AdamW(adam_groups, **adam_kw)]
    else:
        from torch.distributed.optim import ZeroRedundancyOptimizer as Z
        opts = [Z(g["muon"], optimizer_class=Muon, process_group=zero_group, **muon_kw),
                Z(adam_groups, optimizer_class=torch.optim.AdamW, process_group=zero_group, **adam_kw)]
    for opt in opts:
        for pg in opt.param_groups:
            pg["base_lr"] = lr
    return opts


def lr_at(step, cfg):
    """WSD: linear warmup over ceil(warmup_frac*T) steps (step 0 -> lr/W), flat, linear decay over the last decay_frac*T."""
    s, lr, T = cfg["schedule"], cfg["optimizer"]["lr"], cfg["tokens"]["train_steps"]
    W = max(1, math.ceil(s["warmup_frac"] * T))
    D = int(s["decay_frac"] * T)
    lo = s["min_lr_ratio"] * lr
    if step < W:
        return lr * (step + 1) / W
    if D > 0 and step >= T - D:
        return lr + (lo - lr) * (step - (T - D) + 1) / D
    return lr
