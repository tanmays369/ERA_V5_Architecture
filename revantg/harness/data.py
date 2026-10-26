# Name: Revant Gupta
# Pack: ind-hi-fw2
"""Deterministic batches from prepare_data.py output.

A global batch depends only on (general.seed, step, config): never on rank or parallel layout.
"""
import os

import numpy as np
import torch

VAL_WINDOWS_PER_SOURCE = 4


class Batches:
    def __init__(self, cfg):
        d = cfg["paths"]["data"]
        self.names = [ds["name"] for ds in cfg["datasets"]]
        self.seed = cfg["general"]["seed"]
        self.seq_len = cfg["tokens"]["seq_len"]
        self.G = cfg["tokens"]["global_batch_seqs"]
        T = cfg["tokens"]["train_steps"]
        stages = cfg["data_stages"]
        self.stages = [(int(s["start_frac"] * T), s["name"], np.array([s["weights"].get(n, 0.0) for n in self.names]))
                       for s in stages]
        self.train = [np.memmap(os.path.join(d, f"{n}.train.bin"), dtype="<u4", mode="r") for n in self.names]
        self.val = [np.memmap(os.path.join(d, f"{n}.val.bin"), dtype="<u4", mode="r") for n in self.names]

    def _stage(self, step):
        return [s for s in self.stages if s[0] <= step][-1]

    def stage_name(self, step):
        return self._stage(step)[1]

    def global_batch(self, step):
        """(LongTensor[G, seq_len+1], source_idx LongTensor[G])."""
        p = self._stage(step)[2]
        rng = np.random.default_rng([self.seed, step])
        rows, src = [], []
        for _ in range(self.G):
            s = int(rng.choice(len(self.names), p=p))
            off = int(rng.integers(0, len(self.train[s]) - self.seq_len - 1))
            rows.append(self.train[s][off: off + self.seq_len + 1])
            src.append(s)
        return torch.from_numpy(np.stack(rows).astype(np.int64)), torch.tensor(src)

    @staticmethod
    def dp_slice(batch, dp_rank, dp):
        """Rows [dp_rank*G/dp, (dp_rank+1)*G/dp) of a tensor or a tuple of tensors."""
        if isinstance(batch, (tuple, list)):
            return type(batch)(Batches.dp_slice(b, dp_rank, dp) for b in batch)
        n = batch.shape[0] // dp
        return batch[dp_rank * n: (dp_rank + 1) * n]

    def val_batch(self):
        """Fixed LongTensor[4*n_sources, seq_len+1] of val windows; seed-only, same every call."""
        rng = np.random.default_rng([self.seed, 2**32 - 1])  # distinct from any training step's stream
        rows = []
        for v in self.val:
            for off in rng.integers(0, len(v) - self.seq_len - 1, size=VAL_WINDOWS_PER_SOURCE):
                rows.append(v[off: off + self.seq_len + 1])
        return torch.from_numpy(np.stack(rows).astype(np.int64))

    def val_meta(self, tokenizer_path):
        """(source_idx LongTensor[Nv], target_bytes LongTensor[Nv]) for val_batch() rows: BPB divides by the bytes of
        the evaluated target tokens, so its weighting matches the CE (equal windows per source)."""
        nbytes = token_bytes(tokenizer_path)
        rows = self.val_batch()[:, 1:].numpy()
        src = np.repeat(np.arange(len(self.names)), VAL_WINDOWS_PER_SOURCE)
        return torch.from_numpy(src), torch.from_numpy(nbytes[rows].sum(1))


def token_bytes(tokenizer_path):
    """UTF-8 byte length per token id. ByteLevel BPE: one char of the token string = one byte; added tokens (EOS) 0.
    Checked: summed over each smoke `*.val.bin` it equals the manifest `val_bytes` for all 12 sources."""
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(tokenizer_path)
    added = t.get_added_tokens_decoder()
    return np.array([0 if i in added else len(t.id_to_token(i)) for i in range(t.get_vocab_size(with_added_tokens=True))],
                    dtype=np.int64)
