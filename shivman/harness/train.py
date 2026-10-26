"""Tiny MoE smoke test.

Kronecker embeddings, eager attention, top-1 experts.
Seed 1234, run twice: the loss file must match, the loss must fall, no NaN.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from kronecker_embeddings import KroneckerEmbedding
from torch import nn

from check_config import errors_for, load_settings, merge


def as_map(settings: list[dict]) -> dict:
    return {item["key"]: item["value"] for item in settings}


def byte_buffers(vocab: int, pos_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    byte_buffer = torch.zeros(vocab, pos_dim, dtype=torch.uint8)
    length_buffer = torch.zeros(vocab, dtype=torch.int16)
    for token_id in range(1, vocab):
        byte_buffer[token_id, 0] = token_id % 256
        length_buffer[token_id] = 1
    return byte_buffer, length_buffer


class CausalAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(x.shape[-1])
        length = x.shape[1]
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=x.device), 1)
        weights = torch.softmax(scores.masked_fill(mask, -1e9), dim=-1)
        return self.o(weights @ v)


class Top1MoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_expert: int):
        super().__init__()
        self.router = nn.Linear(d_model, n_expert, bias=False)
        self.up = nn.Parameter(torch.empty(n_expert, d_model, d_ff))
        self.down = nn.Parameter(torch.empty(n_expert, d_ff, d_model))
        nn.init.normal_(self.up, std=0.02)
        nn.init.normal_(self.down, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.softmax(self.router(x), dim=-1)
        choice = gates.argmax(dim=-1)
        hidden = torch.einsum("btd,edh->bteh", x, self.up)
        hidden = F.gelu(hidden)
        expert_out = torch.einsum("bteh,ehd->bted", hidden, self.down)
        picked = expert_out.gather(2, choice.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, x.shape[-1]))
        picked = picked.squeeze(2)
        gate = gates.gather(2, choice.unsqueeze(-1))
        return picked * gate


class Block(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_expert: int):
        super().__init__()
        self.norm_1 = nn.RMSNorm(d_model)
        self.attn = CausalAttention(d_model)
        self.norm_2 = nn.RMSNorm(d_model)
        self.moe = Top1MoE(d_model, d_ff, n_expert)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x))
        x = x + self.moe(self.norm_2(x))
        return x


class TinyMoE(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        vocab = int(cfg["tiny.vocab"])
        d_model = int(cfg["model.d_model"])
        pos_dim = int(cfg["embedding.pos_dim"])
        raw, lengths = byte_buffers(vocab, pos_dim)
        self.embed = KroneckerEmbedding(
            vocab_size=vocab,
            d_model=d_model,
            pos_dim=pos_dim,
            mode="cached",
            byte_buffer=raw,
            length_buffer=lengths,
        )
        self.blocks = nn.ModuleList(
            Block(d_model, int(cfg["model.d_ff"]), int(cfg["moe.n_expert"]))
            for _ in range(int(cfg["model.n_layer"]))
        )
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(token_ids)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(self.norm(hidden)), hidden


def batch_for(cfg: dict, seed: int) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    vocab = int(cfg["tiny.vocab"])
    tokens = torch.randint(
        1,
        vocab,
        (int(cfg["tiny.batch"]), int(cfg["data.seq_len"])),
        generator=generator,
    )
    return tokens


def run(cfg: dict, out_dir: Path) -> None:
    seed = int(cfg["run.seed"])
    torch.manual_seed(seed)
    model = TinyMoE(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["optim.lr"]))
    tokens = batch_for(cfg, seed)
    clip = float(cfg["run.grad_clip"])
    out_dir.mkdir(parents=True, exist_ok=True)
    losses = []
    metrics_path = out_dir / "metrics.jsonl"
    with metrics_path.open("w") as handle:
        for step in range(int(cfg["tiny.steps"])):
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits, hidden = model(tokens)
            ntp = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))
            mtp = F.cross_entropy(logits[:, :-2].reshape(-1, logits.shape[-1]), tokens[:, 2:].reshape(-1))
            loss = ntp
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            value = float(loss.detach())
            if math.isnan(value):
                raise SystemExit(f"NaN at step {step}")
            losses.append(value)
            elapsed = max(time.perf_counter() - started, 1e-9)
            n_tokens = tokens.numel()
            row = {
                "step": step,
                "tokens_per_second": n_tokens / elapsed,
                "loss_ntp": value,
                "loss_mtp": float(mtp.detach()),
                "loss": value,
                "lr": float(cfg["optim.lr"]),
                "gpu_alloc_gb": 0.0,
            }
            handle.write(json.dumps(row) + "\n")
    (out_dir / "losses.txt").write_text("\n".join(f"{item:.8f}" for item in losses) + "\n")
    if not (losses[-1] < losses[0]):
        raise SystemExit(f"loss did not fall: {losses[0]} -> {losses[-1]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tiny", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    settings = merge(load_settings(args.config), args.tiny)
    found = errors_for(settings)
    if found:
        raise SystemExit("\n".join(found))
    run(as_map(settings), args.out)


if __name__ == "__main__":
    main()
