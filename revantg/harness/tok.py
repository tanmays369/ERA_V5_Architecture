# Name: Revant Gupta
# Pack: ind-hi-fw2
"""Tokenizer loading and cached Kronecker byte buffers.

`byte_buffers(path, pos_dim)` caches `<tokenizer>.bb<pos_dim>.pt` next to the tokenizer file.
"""
import os

import torch
from kronecker_embeddings import build_byte_buffer
from tokenizers import Tokenizer


def load_tokenizer(path):
    return Tokenizer.from_file(path)


class HFLikeTokenizer:
    """The HF-style surface `kronecker_embeddings.build_byte_buffer` needs."""

    def __init__(self, tok):
        self.tok = tok
        self.all_special_ids = [i for i, t in tok.get_added_tokens_decoder().items() if t.special]

    def get_vocab(self):
        return self.tok.get_vocab(with_added_tokens=True)

    def convert_ids_to_tokens(self, i):
        return self.tok.id_to_token(i)

    def decode(self, ids, skip_special_tokens=False, **kw):
        return self.tok.decode(ids, skip_special_tokens=skip_special_tokens)


def byte_buffers(path, pos_dim):
    """(byte_buffer uint8 [V, pos_dim], length_buffer int16 [V]); built once (~131k decodes), then cached."""
    cache = f"{path}.bb{pos_dim}.pt"
    if os.path.exists(cache):
        d = torch.load(cache)
        return d["byte_buffer"], d["length_buffer"]
    hf = HFLikeTokenizer(load_tokenizer(path))
    bb, lb = build_byte_buffer(hf, pos_dim=pos_dim)
    texts = [hf.decode([i]) for i in range(bb.shape[0])]
    n_fffd = sum("\ufffd" in s for s in texts)
    max_len = max(len(s.encode()) for s in texts)
    print(f"byte_buffers: vocab {bb.shape[0]}, {n_fffd} tokens decode to U+FFFD (partial UTF-8, kept per paper), "
          f"max decoded byte length {max_len}, {int((lb >= pos_dim).sum())} tokens fill all {pos_dim} positions")
    tmp = f"{cache}.{os.getpid()}.tmp"
    torch.save({"byte_buffer": bb, "length_buffer": lb}, tmp)
    os.replace(tmp, cache)  # atomic: concurrent ranks never read a partial file
    return bb, lb


if __name__ == "__main__":
    import sys
    bb, lb = byte_buffers(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 32)
    print(bb.shape, bb.dtype, lb.shape, lb.dtype)
