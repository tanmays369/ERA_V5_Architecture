"""Serve the mixed sample to the training loop in a fixed, repeatable order.

Each row of a batch is a window of sequence_length + 1 + mtp_heads tokens, so the same window gives the
input, the next-token targets and the multi-token-prediction targets. For every row the source (English,
Hindi, Bengali, Telugu) is drawn with the probabilities of data_mix.yaml, then a start position inside that
source. Both draws come from a torch.Generator seeded with the run's seed, so two runs with the same seed
see exactly the same windows in the same order. Indic text is present from the first step (call 25 Oct).

Validation windows are drawn once, from a separate generator (seed + 1), from the held-out parts.
"""
import json
from pathlib import Path

import numpy as np
import torch

SOURCES = ["english", "hindi", "bengali", "telugu"]


class MixedSample:
    def __init__(self, sample_dir, shares, seq_len, extra, seed):
        d = Path(sample_dir)
        if not (d / "manifest.json").exists():
            raise SystemExit(f"No sample in {d}. Build it first: python harness/build_sample.py <config>")
        self.manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        self.train = {s: torch.from_numpy(np.load(d / f"{s}_train.npy").astype(np.int64)) for s in SOURCES}
        self.val = {s: torch.from_numpy(np.load(d / f"{s}_val.npy").astype(np.int64)) for s in SOURCES}
        self.token_bytes = torch.from_numpy(np.load(d / "token_bytes.npy").astype(np.int64))
        self.shares = torch.tensor([float(shares[s]) for s in SOURCES])
        self.shares = self.shares / self.shares.sum()
        self.window = seq_len + 1 + extra
        self.gen = torch.Generator().manual_seed(seed)
        self.val_gen = torch.Generator().manual_seed(seed + 1)

    def _rows(self, data, n, gen, source_ids=None):
        if source_ids is None:
            source_ids = torch.multinomial(self.shares, n, replacement=True, generator=gen)
        rows = []
        for sid in source_ids.tolist():
            arr = data[SOURCES[sid]]
            start = int(torch.randint(0, arr.numel() - self.window, (1,), generator=gen))
            rows.append(arr[start:start + self.window])
        return torch.stack(rows), source_ids

    def batch(self, n):
        """n training windows, shape (n, window)."""
        return self._rows(self.train, n, self.gen)[0]

    def validation_set(self, n_batches, n):
        """A fixed list of validation batches, the same for every evaluation of the run."""
        return [self._rows(self.val, n, self.val_gen)[0] for _ in range(n_batches)]
