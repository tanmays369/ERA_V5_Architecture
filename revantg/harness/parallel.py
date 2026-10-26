# Name: Revant Gupta
# Pack: ind-hi-fw2
"""Layouts on a (pp, dp, tp) DeviceMesh: process groups, TP plan, PP stage trimming/schedule, sharding-aware grad norm.
DP is plain DDP over the dp group (train.py); ZeRO-1 lives in optim.build_optimizers(zero_group).
"""
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleGPipe
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

try:
    from torch.distributed.tensor import DTensor, Replicate, Shard
except ImportError:  # torch < 2.5
    from torch.distributed._tensor import DTensor, Replicate, Shard


@dataclass
class Dist:
    rank: int
    world: int
    local_rank: int
    device: torch.device
    mesh: object
    pp: int
    dp: int
    tp: int
    pp_rank: int
    dp_rank: int
    tp_rank: int
    pp_group: object
    dp_group: object
    tp_group: object


def setup_dist(cfg, device=None):
    """Init the default group (nccl on cuda, else gloo) and the (pp, dp, tp) mesh; world must equal pp*dp*tp."""
    p = cfg["parallelism"]
    pp, dp, tp = p["pp"], p["dp"], p["tp"]
    cuda = device == "cuda" if device else torch.cuda.is_available()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if cuda:
        torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank) if cuda else torch.device("cpu")
    dist.init_process_group("nccl" if cuda else "gloo")
    world = dist.get_world_size()
    if world != pp * dp * tp:
        raise SystemExit(f"world size {world} != pp*dp*tp = {pp}*{dp}*{tp}")
    mesh = init_device_mesh(dev.type, (pp, dp, tp), mesh_dim_names=("pp", "dp", "tp"))
    r, g = mesh.get_local_rank, mesh.get_group
    return Dist(dist.get_rank(), world, local_rank, dev, mesh, pp, dp, tp,
                r("pp"), r("dp"), r("tp"), g("pp"), g("dp"), g("tp"))


def apply_tp(model, tp_mesh):
    """Megatron plan: q/k/v/gate/up column-, o/down row-parallel; head column-parallel gathered; embed+norms replicated."""
    if model.moe_layers():
        raise SystemExit("MoE + TP not supported in v0")
    plan = {"attn.wq": ColwiseParallel(), "attn.wk": ColwiseParallel(), "attn.wv": ColwiseParallel(),
            "attn.wo": RowwiseParallel(), "mlp.w_gate": ColwiseParallel(), "mlp.w_up": ColwiseParallel(),
            "mlp.w_down": RowwiseParallel()}
    for block in model.layers:
        parallelize_module(block, tp_mesh, plan)
    if model.head is not None:
        parallelize_module(model, tp_mesh, {"head": ColwiseParallel(output_layouts=Replicate())})
    return model


def sync_tp_grads(model, d):
    """q_norm/k_norm see only local heads under TP, so their grads are partial sums: all-reduce over tp."""
    for block in model.layers:
        for norm in (block.attn.q_norm, block.attn.k_norm):
            if norm is not None and norm.weight.grad is not None:
                dist.all_reduce(norm.weight.grad, group=d.tp_group)


def pp_stage_module(model, pp_rank, pp):
    """Trim the full model in place to this stage: contiguous layers; embed on first, norm+head on last."""
    n = len(model.layers) // pp
    model.layers = nn.ModuleList(list(model.layers)[pp_rank * n:(pp_rank + 1) * n])
    if pp_rank > 0:
        model.embed = None
    if pp_rank < pp - 1:
        model.norm = None
        model.head = None
    return model


def build_schedule(stage_module, d, cfg, loss_fn):
    """PipelineStage + 1F1B (pp_engine 1f1b) or GPipe all-forward-all-backward (afab); grads averaged over micro-batches."""
    p = cfg["parallelism"]
    stage = PipelineStage(stage_module, d.pp_rank, d.pp, d.device, group=d.pp_group)
    cls = {"1f1b": Schedule1F1B, "afab": ScheduleGPipe}[p["pp_engine"]]
    return cls(stage, n_microbatches=p["pp_microbatches"], loss_fn=loss_fn)


def pp_step(schedule, d, x, y, losses):
    """One pipelined forward+backward of this dp rank's batch; last stage fills `losses` (one per micro-batch)."""
    kw = {"return_outputs": False} if "return_outputs" in schedule.step.__code__.co_varnames else {}
    if d.pp_rank == 0:
        schedule.step(x, **kw)
    elif d.pp_rank == d.pp - 1:
        schedule.step(target=y, losses=losses, **kw)
    else:
        schedule.step(**kw)


def pp_forward(stage_module, d, x, d_model):
    """No-grad forward through the pipeline with plain p2p; returns logits on the last stage, else None."""
    if d.pp_rank > 0:
        h = torch.empty(x.shape[0], x.shape[1], d_model, device=d.device)
        dist.recv(h, src=dist.get_global_rank(d.pp_group, d.pp_rank - 1), group=d.pp_group)
        x = h
    out = stage_module(x)
    if d.pp_rank < d.pp - 1:
        dist.send(out.contiguous(), dst=dist.get_global_rank(d.pp_group, d.pp_rank + 1), group=d.pp_group)
        return None
    return out


def last_stage_rank(d):
    return dist.get_global_rank(d.pp_group, d.pp - 1) if d.pp > 1 else d.rank


def global_sq(tensors, d):
    """Sum of squares over the full (unsharded, all-stage) tensor set: TP shards summed over tp, PP stages over pp."""
    rep = torch.zeros((), device=d.device)
    sh = torch.zeros((), device=d.device)
    for t in tensors:
        if isinstance(t, DTensor):
            s = t.to_local().float().pow(2).sum()
            if any(isinstance(pl, Shard) for pl in t.placements):
                sh += s
            else:
                rep += s
        else:
            rep += t.float().pow(2).sum()
    if d.tp > 1:
        dist.all_reduce(sh, group=d.tp_group)
    tot = rep + sh
    if d.pp > 1:
        dist.all_reduce(tot, group=d.pp_group)
    return tot


def clip_grads(params, max_norm, d):
    """Global grad-norm clip (same rule as torch.nn.utils.clip_grad_norm_); returns the pre-clip norm as float."""
    grads = [p.grad for p in params if p.grad is not None]
    norm = float(global_sq(grads, d).sqrt())
    coef = max_norm / (norm + 1e-6)
    if max_norm > 0 and coef < 1:
        for g in grads:
            g.mul_(coef)
    return norm
