# Name: dharshan
# Axiom email: dharshanthaigu@gmail.com
# Pack: ind-ta-fw2
"""Data for the harness. Batch k depends only on (seed, k), never on earlier batches, so the data order is the same
on every run and after a resume: to continue at step N, ask for step N.

data.source = synthetic   two built-in rules over a small token alphabet; needs no download and no tokenizer file.
                          Token ids are 0..SYN_ALPHABET-1; the KE byte table for ALL tokenizer.vocab_size ids is
                          built from the made-up names "t0", "t1", ... (not from the real tokenizer).
data.source = parquet     one parquet file per source in data.paths, text in data.text_column, tokenised with the
                          pinned tokenizer (downloaded once, sha256 checked). NOT RUN YET.

A window has tokens.seq_len + 1 + head.mtp_heads ids. Model inputs are window[:, :T], next-token targets
window[:, 1:T+1], MTP targets window[:, 2:T+2].

Held-out: data.val_windows_per_source windows from EACH source, so every source counts equally.
"""
import hashlib
import os
import urllib.request

import numpy as np
import torch

SYN_ALPHABET = 64                                    # [ours] small alphabet so the loss can fall in 60 steps
SYN_RULES = {"rule_a": (3, 1), "rule_b": (5, 7)}     # [ours] next = (a * cur + b) % alphabet
SYN_NOISE = 0.1                                      # [ours] 10% of next tokens are random

TRAIN_STREAM, HELD_STREAM = 0, 1


def _gen(seed, stream, index):
    g = torch.Generator()
    g.manual_seed((seed * 1_000_003 + stream * 999_983 + index) % (2**63 - 1))
    return g


def _rule_block(g, n, width, mul, add):
    x = torch.empty(n, width, dtype=torch.long)
    x[:, 0] = torch.randint(0, SYN_ALPHABET, (n,), generator=g)
    flip = torch.rand(n, width - 1, generator=g) < SYN_NOISE
    rnd = torch.randint(0, SYN_ALPHABET, (n, width - 1), generator=g)
    for t in range(width - 1):
        x[:, t + 1] = torch.where(flip[:, t], rnd[:, t], (mul * x[:, t] + add) % SYN_ALPHABET)
    return x


def load_tokenizer(cfg, cache_dir=None):
    """Download tokenizer.json at the pinned URL, check sha256, return a fast tokenizer. NOT RUN YET."""
    from transformers import PreTrainedTokenizerFast
    cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "g5_harness")
    os.makedirs(cache_dir, exist_ok=True)
    want = cfg.get("tokenizer.sha256")
    path = os.path.join(cache_dir, f"tokenizer_{want[:12]}.json")
    if not os.path.isfile(path):
        raw = urllib.request.urlopen(cfg.get("tokenizer.url"), timeout=120).read()
        got = hashlib.sha256(raw).hexdigest()
        if got != want:
            raise SystemExit(f"tokenizer sha256 mismatch: file has {got}, config says {want}")
        with open(path, "wb") as f:
            f.write(raw)
    elif hashlib.sha256(open(path, "rb").read()).hexdigest() != want:
        raise SystemExit(f"cached tokenizer {path} does not match the sha256 in the config; delete it")
    return PreTrainedTokenizerFast(tokenizer_file=path)


class Data:
    def __init__(self, cfg):
        self.cfg = cfg
        self.seed = cfg.get("general.seed")
        self.T, self.mtp = cfg.get("tokens.seq_len"), cfg.get("head.mtp_heads")
        self.W = self.T + 1 + self.mtp
        self.M = cfg.get("tokens.micro_batch_seqs")
        self.accum = cfg.get("tokens.global_batch_seqs") // self.M
        self.V, self.pos_dim = cfg.get("tokenizer.vocab_size"), cfg.get("embedding.pos_dim")
        self.n_val = cfg.get("data.val_windows_per_source")
        self.source = cfg.get("data.source")
        self._held = None
        if self.source == "synthetic":
            self.names = list(SYN_RULES)
            self.tok = None
        elif self.source == "parquet":
            self._load_parquet()
        else:
            raise ValueError(f"data.source = {self.source!r}: only synthetic and parquet are implemented")

    # ---------- KE byte table ----------
    def byte_buffers(self):
        from kronecker_embeddings import build_byte_buffer
        if self.tok is not None:
            bb, lb = build_byte_buffer(self.tok, pos_dim=self.pos_dim)
            if bb.shape[0] != self.V:
                raise SystemExit(f"tokenizer has {bb.shape[0]} ids but tokenizer.vocab_size is {self.V}")
            return bb, lb
        bb = np.zeros((self.V, self.pos_dim), dtype=np.uint8)
        lb = np.zeros((self.V,), dtype=np.int16)
        for i in range(self.V):
            raw = f"t{i}".encode()[: self.pos_dim]
            bb[i, : len(raw)] = np.frombuffer(raw, dtype=np.uint8)
            lb[i] = len(raw)
        return torch.from_numpy(bb), torch.from_numpy(lb)

    # ---------- parquet ----------
    def _load_parquet(self):
        import pandas as pd
        paths = self.cfg.get("data.paths")
        if not isinstance(paths, list) or not paths:
            raise ValueError("data.paths must be a non-empty list of parquet files")
        self.tok = load_tokenizer(self.cfg)
        eos, col, vf = self.cfg.get("tokenizer.eos_id"), self.cfg.get("data.text_column"), self.cfg.get("data.val_frac")
        self.names, self.train_ids, self.val_ids = [], [], []
        for p in paths:
            texts = pd.read_parquet(p, columns=[col])[col].tolist()
            ids = []
            for t in texts:
                ids.extend(self.tok(t, add_special_tokens=False)["input_ids"] + [eos])
            ids = torch.tensor(ids, dtype=torch.long)
            cut = int(len(ids) * (1 - vf))
            if cut < self.W or len(ids) - cut < self.W:
                raise SystemExit(f"{p}: too few tokens ({len(ids)}) for train/val windows of {self.W}")
            self.names.append(os.path.basename(p))
            self.train_ids.append(ids[:cut])
            self.val_ids.append(ids[cut:])
        w = self.cfg.get("data.weights")
        w = [1.0] * len(paths) if w == "equal" else list(w)
        if len(w) != len(paths):
            raise ValueError("data.weights must be 'equal' or one number per path")
        self.weights = torch.tensor(w, dtype=torch.float)

    # ---------- batches ----------
    def train_batch(self, step, micro):
        """LongTensor [micro_batch_seqs, W] for optimiser step `step`, micro-batch `micro`."""
        g = _gen(self.seed, TRAIN_STREAM, step * self.accum + micro)
        if self.source == "synthetic":
            out = torch.empty(self.M, self.W, dtype=torch.long)
            n = len(self.names)
            for s, name in enumerate(self.names):          # rows alternate between sources: equal weight
                rows = list(range(s, self.M, n))
                if rows:
                    out[rows] = _rule_block(g, len(rows), self.W, *SYN_RULES[name])
            return out
        rows = []
        for _ in range(self.M):
            s = int(torch.multinomial(self.weights, 1, generator=g))
            ids = self.train_ids[s]
            start = int(torch.randint(0, len(ids) - self.W + 1, (1,), generator=g))
            rows.append(ids[start:start + self.W])
        return torch.stack(rows)

    def held_out(self):
        """{source name: LongTensor [val_windows_per_source, W]}, the same on every call."""
        if self._held is None:
            if self.source == "synthetic":
                self._held = {n: _rule_block(_gen(self.seed, HELD_STREAM, i), self.n_val, self.W, *SYN_RULES[n])
                              for i, n in enumerate(self.names)}
            else:
                self._held = {}
                for n, ids in zip(self.names, self.val_ids):
                    starts = np.linspace(0, len(ids) - self.W, self.n_val).astype(int)
                    self._held[n] = torch.stack([ids[s:s + self.W] for s in starts])
        return self._held
