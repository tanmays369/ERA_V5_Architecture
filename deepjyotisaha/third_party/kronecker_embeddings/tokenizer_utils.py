"""
Utilities for building the per-token byte buffer from a HuggingFace tokenizer.

The byte buffer is the canonical input to the Kronecker codec: for each token
id, what UTF-8 byte sequence does that token represent in actual text? Three
cases are handled:

1. Standard tokens: the byte representation of the token's surface form,
   reconstructed via ``tokenizer.decode([id])``.
2. Byte-fallback tokens like ``<0xNN>``: the single byte v with value 0xNN.
3. Special / added tokens: the literal string representation (e.g. ``<s>``
   becomes bytes(0x3c, 0x73, 0x3e)). This gives them a deterministic,
   distinct codec output.

Long tokens are UTF-8-safe-truncated to at most ``pos_dim`` bytes — never
splitting a multibyte codepoint.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

BYTE_FALLBACK_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def utf8_safe_truncate(byte_seq: bytes, max_bytes: int) -> bytes:
    """
    Truncate ``byte_seq`` to at most ``max_bytes`` bytes without splitting
    a UTF-8 multibyte codepoint. Returns valid UTF-8 bytes (possibly empty).
    """
    if len(byte_seq) <= max_bytes:
        return byte_seq
    # UTF-8 codepoints are at most 4 bytes; walk back from max_bytes.
    for end in range(max_bytes, max(max_bytes - 4, -1), -1):
        try:
            byte_seq[:end].decode("utf-8")
            return byte_seq[:end]
        except UnicodeDecodeError:
            continue
    return b""


def token_id_to_bytes(
    tokenizer,
    token_id: int,
    special_ids: Optional[set] = None,
) -> bytes:
    """
    Map a token id to its canonical UTF-8 byte sequence.

    Resolution order:
      1. If the token piece matches ``<0xNN>``, return ``bytes([0xNN])``.
      2. If ``token_id`` is in ``special_ids``, return the literal piece string
         encoded as UTF-8 (e.g. ``<s>`` -> b"<s>").
      3. Otherwise return ``tokenizer.decode([token_id]).encode("utf-8")``,
         which is the natural surface-form bytes for that token.
    """
    if special_ids is None:
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    piece = tokenizer.convert_ids_to_tokens(token_id)
    if piece:
        m = BYTE_FALLBACK_RE.match(piece)
        if m:
            return bytes([int(m.group(1), 16)])

    if token_id in special_ids:
        if piece is None:
            piece = ""
        return piece.encode("utf-8")

    try:
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except Exception:
        decoded = piece or ""
    if decoded is None:
        decoded = ""
    return decoded.encode("utf-8")


def build_byte_buffer(
    tokenizer,
    pos_dim: int = 32,
) -> Tuple[Tensor, Tensor]:
    """
    Build ``(byte_buffer, length_buffer)`` tensors from a HuggingFace tokenizer.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizerBase
        Any HF tokenizer with ``get_vocab``, ``convert_ids_to_tokens``, and
        ``decode`` available.
    pos_dim : int, default 32
        Maximum byte position. Tokens whose UTF-8 bytes exceed this are
        truncated UTF-8-safely.

    Returns
    -------
    byte_buffer : Tensor[uint8] of shape (vocab_size, pos_dim)
        Padded bytes per token id.
    length_buffer : Tensor[int16] of shape (vocab_size,)
        Effective UTF-8 byte length per token (post-truncation).
    """
    vocab = tokenizer.get_vocab()
    if not vocab:
        raise ValueError("tokenizer.get_vocab() returned an empty dict")
    vocab_size = max(vocab.values()) + 1

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    byte_buffer = np.zeros((vocab_size, pos_dim), dtype=np.uint8)
    length_buffer = np.zeros((vocab_size,), dtype=np.int16)

    for tid in range(vocab_size):
        try:
            raw = token_id_to_bytes(tokenizer, tid, special_ids)
        except Exception:
            raw = b""
        if len(raw) > pos_dim:
            raw = utf8_safe_truncate(raw, pos_dim)
        L = len(raw)
        if L > 0:
            byte_buffer[tid, :L] = np.frombuffer(raw, dtype=np.uint8, count=L)
        length_buffer[tid] = L

    return (
        torch.from_numpy(byte_buffer),
        torch.from_numpy(length_buffer),
    )


def coverage_stats(length_buffer: Tensor, pos_dim: int = 32) -> dict:
    """
    Summary of how many tokens hit the ``pos_dim`` ceiling.

    Returns a dict with vocab_size, n_truncated, pct_within_pos_dim,
    max_observed_length.
    """
    lengths = length_buffer.numpy() if isinstance(length_buffer, Tensor) else length_buffer
    n = int(lengths.shape[0])
    truncated_mask = lengths >= pos_dim
    n_truncated = int(truncated_mask.sum())
    return {
        "vocab_size": n,
        "pos_dim": int(pos_dim),
        "n_at_or_above_pos_dim": n_truncated,
        "pct_within_pos_dim": 100.0 * (n - n_truncated) / max(n, 1),
        "max_observed_length": int(lengths.max()) if n > 0 else 0,
    }
