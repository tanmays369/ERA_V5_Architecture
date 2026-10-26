"""
Kronecker codec — the pure-function math layer.

For a token whose UTF-8 byte sequence is b_1, ..., b_L (with L <= pos_dim,
truncated UTF-8-safely if longer), the codec computes:

    kappa(b) = (1 / sqrt(L)) * vec( sum_{p=1..L} c_{b_p} ⊗ p_p )

where c_v is the v-th standard basis vector in R^{char_dim} (one-hot for byte
value v) and p_p is the p-th standard basis vector in R^{pos_dim} (one-hot for
position p). The result is a vector of dimension D = char_dim * pos_dim
which is then z-normalized per-token.

This module exposes the math as pure functions so callers can use it without
constructing the full ``KroneckerEmbedding`` Module. See ``embedding.py`` for
the trainable wrapper.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor


def kronecker_codec(
    byte_sequences: Tensor,
    lengths: Tensor,
    char_dim: int = 256,
    pos_dim: int = 32,
    length_normalize: bool = True,
    z_normalize: bool = True,
    eps: float = 1e-6,
    out_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """
    Compute kappa(b) for a batch of byte sequences.

    Parameters
    ----------
    byte_sequences : Tensor[uint8 or long] of shape (B, pos_dim)
        Padded byte sequences. byte_sequences[i, :lengths[i]] holds the
        actual bytes; positions >= lengths[i] are ignored (treat as padding).
    lengths : Tensor[int16 or long] of shape (B,)
        Number of valid bytes per token. Values must satisfy 0 <= L <= pos_dim.
    char_dim : int, default 256
        Byte alphabet size.
    pos_dim : int, default 32
        Maximum byte position.
    length_normalize : bool, default True
        If True, divide by sqrt(L) (Equation 1 of the paper).
    z_normalize : bool, default True
        If True, apply per-token z-normalization (subtract mean, divide by
        std + eps) before return. Matches the production codec.
    eps : float, default 1e-6
        Numerical stabilizer for z-norm denominator.
    out_dtype : torch.dtype, optional
        Cast the final result to this dtype. Defaults to fp32.

    Returns
    -------
    Tensor of shape (B, char_dim * pos_dim).
    """
    if byte_sequences.dim() != 2:
        raise ValueError(
            f"byte_sequences must be (B, pos_dim); got {tuple(byte_sequences.shape)}"
        )
    if byte_sequences.size(1) != pos_dim:
        raise ValueError(
            f"byte_sequences.size(1)={byte_sequences.size(1)} != pos_dim={pos_dim}"
        )

    device = byte_sequences.device
    B = byte_sequences.size(0)
    D = char_dim * pos_dim

    bytes_long = byte_sequences.to(torch.long)
    lens_long = lengths.to(torch.long).to(device)
    pos = torch.arange(pos_dim, device=device).unsqueeze(0).expand(B, -1)  # (B, pos_dim)

    # Linear index into the flattened (char_dim, pos_dim) tensor.
    # Storage convention matches production: index = byte_value * pos_dim + pos.
    lin_idx = bytes_long * pos_dim + pos  # (B, pos_dim)

    # Mask out padding positions.
    valid = pos < lens_long.unsqueeze(1)  # (B, pos_dim) bool

    if length_normalize:
        # 1/sqrt(L); guard L=0 with clamp.
        scales = torch.rsqrt(lens_long.clamp_min(1).to(torch.float32))  # (B,)
        src = valid.to(torch.float32) * scales.unsqueeze(1)
    else:
        src = valid.to(torch.float32)

    # Scatter-add ones into the flat (B, D) output.
    out = torch.zeros((B, D), device=device, dtype=torch.float32)
    out.scatter_add_(dim=1, index=lin_idx, src=src)

    if z_normalize:
        mean = out.mean(dim=-1, keepdim=True)
        std = (out - mean).std(dim=-1, keepdim=True) + eps
        out = (out - mean) / std

    if out_dtype is not None:
        out = out.to(out_dtype)
    return out


def encode_single(
    byte_seq: bytes,
    char_dim: int = 256,
    pos_dim: int = 32,
    length_normalize: bool = True,
    z_normalize: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    """
    Convenience wrapper: encode a single bytes object directly.

    The bytes are NOT UTF-8-safe-truncated here; callers that need that
    should call ``utf8_safe_truncate`` (in tokenizer_utils) before passing.
    Bytes beyond pos_dim are simply ignored.
    """
    L = min(len(byte_seq), pos_dim)
    buf = torch.zeros(pos_dim, dtype=torch.uint8)
    if L > 0:
        buf[:L] = torch.frombuffer(bytearray(byte_seq[:L]), dtype=torch.uint8)
    return kronecker_codec(
        buf.unsqueeze(0),
        torch.tensor([L], dtype=torch.long),
        char_dim=char_dim,
        pos_dim=pos_dim,
        length_normalize=length_normalize,
        z_normalize=z_normalize,
        eps=eps,
    ).squeeze(0)


def codec_output_dim(char_dim: int = 256, pos_dim: int = 32) -> int:
    """Return D = char_dim * pos_dim. Useful for sizing the projection layer."""
    return char_dim * pos_dim
