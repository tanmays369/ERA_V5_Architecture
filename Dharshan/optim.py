# Name: dharshan
# Axiom email: dharshanthaigu@gmail.com
# Pack: ind-ta-fw2
"""Optimiser: Muon on matrix weights, AdamW on the rest, plus the warm-up / flat / linear-decay (wsd) schedule.

Which parameter goes where (my reading of the lead's rule "Muon on 2D weights, AdamW on the rest, no weight decay
on norms, biases and router"; to confirm with the lead; saved per parameter in run_info.json):
  KE projection (embed.*)                          AdamW, weight decay = embedding.weight_decay
  output head (head.*)                             AdamW, weight decay = optimizer.weight_decay
  router, norms (1D weights)            AdamW, no weight decay
  every other matrix (attention, experts,          Muon (weight decay = optimizer.weight_decay), or AdamW with
  shared expert, dense ffn, MTP block/proj)        decay if optimizer.matrices_2d = adamw
Expert weights are stored as [experts, d, f]; Muon treats each [d, f] slice as its own matrix.

Muon here is written from my memory of the public algorithm (momentum, Newton-Schulz orthogonalisation of the
update, shape scale). The Newton-Schulz coefficients are from memory and NOT checked against reference code.
"""
import torch


def newton_schulz(G, steps):
    a, b, c = 3.4445, -4.7750, 2.0315                   # from memory of the public Muon code, unverified
    X = G.float()
    flip = X.shape[-2] > X.shape[-1]
    if flip:
        X = X.transpose(-2, -1)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.transpose(-2, -1) if flip else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, weight_decay, ns_steps):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                st["buf"].mul_(g["momentum"]).add_(p.grad)
                upd = newton_schulz(p.grad + g["momentum"] * st["buf"], g["ns_steps"])
                scale = max(1.0, p.shape[-2] / p.shape[-1]) ** 0.5
                p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(upd.to(p.dtype), alpha=-g["lr"] * scale)


def lr_factor(step, total, warmup_frac, decay_frac, min_frac):
    """wsd: linear warm-up, flat at 1.0, linear decay to min_frac (the last step reaches min_frac)."""
    warm, decay = int(warmup_frac * total), int(decay_frac * total)
    if warm > 0 and step < warm:
        return (step + 1) / warm
    start = total - decay
    if decay > 0 and step >= start:
        prog = (step - start) / max(1, decay - 1)
        return 1.0 - (1.0 - min_frac) * min(prog, 1.0)
    return 1.0


class Optim:
    def __init__(self, model, cfg):
        g = cfg.get
        self.total = g("tokens.train_steps")
        self.sched = (g("schedule.warmup_frac"), g("schedule.decay_frac"), g("schedule.min_lr_frac"))
        self.lr_muon, self.lr_adamw = g("optimizer.lr_muon"), g("optimizer.lr_adamw")
        self.clip = g("optimizer.grad_clip")
        wd, use_muon = g("optimizer.weight_decay"), g("optimizer.matrices_2d") == "muon"
        decay, nodecay, ke, muon = [], [], [], []
        self.assignment = {}
        for name, p in model.named_parameters():
            if name.startswith("embed."):
                ke.append(p); grp = "adamw_ke_projection"
            elif name.startswith("head."):
                decay.append(p); grp = "adamw_decay"
            elif "router" in name or p.ndim < 2:
                nodecay.append(p); grp = "adamw_nodecay"
            elif use_muon:
                muon.append(p); grp = "muon"
            else:
                decay.append(p); grp = "adamw_decay"
            self.assignment[name] = grp
        betas = tuple(g("optimizer.betas"))
        groups = [dict(params=ps, weight_decay=w, lr=self.lr_adamw, base_lr=self.lr_adamw)
                  for ps, w in ((decay, wd), (nodecay, 0.0), (ke, g("embedding.weight_decay"))) if ps]
        self.adamw = torch.optim.AdamW(groups, betas=betas)
        self.muon = None
        if muon:
            self.muon = Muon(muon, self.lr_muon, g("optimizer.muon_momentum"), wd, g("optimizer.muon_ns_steps"))
            for grp in self.muon.param_groups:
                grp["base_lr"] = self.lr_muon
        self.params = [p for p in model.parameters()]

    def set_lr(self, step):
        f = lr_factor(step, self.total, *self.sched)
        for opt in (self.adamw, self.muon):
            if opt is not None:
                for grp in opt.param_groups:
                    grp["lr"] = grp["base_lr"] * f
        return self.lr_adamw * f, self.lr_muon * f

    def zero_grad(self):
        self.adamw.zero_grad(set_to_none=True)
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=True)

    def clip_grads(self):
        return float(torch.nn.utils.clip_grad_norm_(self.params, self.clip))

    def step(self):
        self.adamw.step()
        if self.muon is not None:
            self.muon.step()
