"""
KroneckerEmbedding: a drop-in replacement for nn.Embedding.

Architecture::

    input_ids  ->  byte_buffer[input_ids]  ->  kappa(...)  ->  Linear(D, d_model)  ->  output

The codec ("kappa") is deterministic (no gradient). The only trainable
parameter is the projection. There is no normalization, residual, or
activation applied here — the host model can stack whatever it likes on
the output of this module.

Two computation modes are supported:

- ``"dynamic"`` (default): keep only the compact ``(V, pos_dim)`` byte
  buffer in memory (~4.5 MB for a 131K vocab at pos_dim=32). Recompute
  the codec on every forward pass. Slight per-step overhead (~1-4 ms at
  the scales tested in the paper). Use this at frontier scale.

- ``"cached"``: precompute the full ``(V, D)`` codec table once, then
  index into it on every forward pass. Zero forward compute. Memory cost
  scales as ``V * D * dtype_size`` bytes (gigabytes at frontier scale).
  Use this when V is small or you have memory to burn.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import math
import torch
import torch.nn as nn
from torch import Tensor

from .codec import codec_output_dim, kronecker_codec
from .tokenizer_utils import build_byte_buffer, utf8_safe_truncate


class KroneckerEmbedding(nn.Module):
    """
    Drop-in replacement for ``nn.Embedding`` using a byte-level Kronecker codec.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the tokenizer vocabulary.
    d_model : int
        Output dimension (matches the transformer body's hidden size).
    tokenizer : PreTrainedTokenizerBase, optional
        Tokenizer used to build the byte buffer. Required unless
        ``byte_buffer`` and ``length_buffer`` are passed directly.
    char_dim : int, default 256
        Byte alphabet size. Always 256 unless you have a strong reason.
    pos_dim : int, default 32
        Maximum byte position. Tokens longer than pos_dim are
        UTF-8-safe-truncated. 32 covers >=99.82% of tokens on the six
        tokenizers in the paper.
    mode : {"dynamic", "cached"}, default "dynamic"
        Computation strategy. See module docstring.
    byte_buffer : Tensor[uint8] of shape (vocab_size, pos_dim), optional
        Pre-built byte buffer. If None, built from ``tokenizer``.
    length_buffer : Tensor[int16] of shape (vocab_size,), optional
        Pre-built length buffer. If None, built from ``tokenizer``.
    projection_init : {"normal", "xavier"}, default "normal"
        Init scheme for the projection. ``"normal"`` uses std=1/sqrt(D),
        matching the paper.
    length_normalize : bool, default True
        Scale the codec by 1/sqrt(L). The paper's default.
    z_normalize : bool, default True
        Per-token z-norm the codec output before projection. The paper's
        default.

    Attributes
    ----------
    num_embeddings : int
        Alias for ``vocab_size``, matches ``nn.Embedding``.
    embedding_dim : int
        Alias for ``d_model``, matches ``nn.Embedding``.
    D : int
        Codec output dimension = ``char_dim * pos_dim``.

    Notes
    -----
    The codec has no learnable parameters; ``state_dict()`` only contains
    the projection. The byte and length buffers are reconstructed from the
    tokenizer at load time (or restored from the provided tensors).

    See Section 3 of the paper for the full method.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        tokenizer=None,
        char_dim: int = 256,
        pos_dim: int = 32,
        mode: Literal["dynamic", "cached"] = "dynamic",
        byte_buffer: Optional[Tensor] = None,
        length_buffer: Optional[Tensor] = None,
        projection_init: Literal["normal", "xavier"] = "normal",
        length_normalize: bool = True,
        z_normalize: bool = True,
    ):
        super().__init__()
        if mode not in ("dynamic", "cached"):
            raise ValueError(f"mode must be 'dynamic' or 'cached'; got {mode!r}")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.char_dim = char_dim
        self.pos_dim = pos_dim
        self.D = codec_output_dim(char_dim, pos_dim)
        self.mode = mode
        self.length_normalize = length_normalize
        self.z_normalize = z_normalize

        # Build byte buffer if not provided.
        if byte_buffer is None or length_buffer is None:
            if tokenizer is None:
                raise ValueError(
                    "Must pass either ``tokenizer`` or both ``byte_buffer`` "
                    "and ``length_buffer``."
                )
            bb, lb = build_byte_buffer(tokenizer, pos_dim=pos_dim)
            byte_buffer = bb
            length_buffer = lb

        # Validate shapes.
        if byte_buffer.shape != (vocab_size, pos_dim):
            raise ValueError(
                f"byte_buffer must have shape ({vocab_size}, {pos_dim}); "
                f"got {tuple(byte_buffer.shape)}"
            )
        if length_buffer.shape != (vocab_size,):
            raise ValueError(
                f"length_buffer must have shape ({vocab_size},); "
                f"got {tuple(length_buffer.shape)}"
            )

        self.register_buffer("_byte_buffer", byte_buffer.to(torch.uint8), persistent=False)
        self.register_buffer("_length_buffer", length_buffer.to(torch.int16), persistent=False)

        # Trainable projection D -> d_model.
        self.projection = nn.Linear(self.D, d_model, bias=False)
        if projection_init == "normal":
            nn.init.normal_(self.projection.weight, mean=0.0, std=1.0 / math.sqrt(self.D))
        elif projection_init == "xavier":
            nn.init.xavier_uniform_(self.projection.weight)
        else:
            raise ValueError(
                f"projection_init must be 'normal' or 'xavier'; got {projection_init!r}"
            )

        # Cached mode: precompute the full (V, D) codec table once.
        if mode == "cached":
            with torch.no_grad():
                table = kronecker_codec(
                    self._byte_buffer,
                    self._length_buffer,
                    char_dim=self.char_dim,
                    pos_dim=self.pos_dim,
                    length_normalize=self.length_normalize,
                    z_normalize=self.z_normalize,
                )
            self.register_buffer("_codec_table", table, persistent=False)

    # -------- nn.Embedding-compatible aliases --------
    @property
    def num_embeddings(self) -> int:
        return self.vocab_size

    @property
    def embedding_dim(self) -> int:
        return self.d_model

    # -------- core forward --------
    def _codec_lookup(self, input_ids: Tensor) -> Tensor:
        """Return ``(..., L, D)`` codec output for ``input_ids`` of shape ``(...,L)``."""
        flat_ids = input_ids.reshape(-1)
        if self.mode == "cached":
            codec_out = self._codec_table.index_select(0, flat_ids)
        else:
            # Dynamic: fetch bytes + lens, run codec.
            bytes_all = self._byte_buffer.index_select(0, flat_ids)
            lens_all = self._length_buffer.index_select(0, flat_ids)
            codec_out = kronecker_codec(
                bytes_all,
                lens_all,
                char_dim=self.char_dim,
                pos_dim=self.pos_dim,
                length_normalize=self.length_normalize,
                z_normalize=self.z_normalize,
            )
        return codec_out.view(*input_ids.shape, self.D)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Map ``input_ids`` of shape ``(..., L)`` to embeddings of shape
        ``(..., L, d_model)``. Matches ``nn.Embedding.forward`` semantics.
        """
        codec_out = self._codec_lookup(input_ids)
        return self.projection(codec_out.to(self.projection.weight.dtype))

    def forward_with_byte_override(
        self,
        input_ids: Tensor,
        position_byte_overrides: Optional[Dict[int, bytes]] = None,
    ) -> Tensor:
        """
        Like :meth:`forward` but override specific positions with arbitrary
        byte sequences. Single-row only (``input_ids`` shape ``(L,)`` or
        ``(1, L)``).

        Use case: forced-OOV inference (Section 6.12 of the paper). Replace
        a chosen position's bytes with a string that doesn't have a single
        vocab token (e.g. a hand-coined word) and let the codec compute its
        embedding from the byte sequence.

        Parameters
        ----------
        input_ids : Tensor of shape ``(L,)`` or ``(1, L)``
            Token ids; positions named in ``position_byte_overrides`` will
            be replaced.
        position_byte_overrides : dict, optional
            Mapping ``{position_in_sequence: utf8_bytes}``. The bytes are
            UTF-8-safe-truncated to ``self.pos_dim`` before codec.

        Returns
        -------
        Tensor of shape ``(L, d_model)`` or ``(1, L, d_model)`` matching the
        input shape.
        """
        if position_byte_overrides is None or len(position_byte_overrides) == 0:
            return self.forward(input_ids)

        squeezed = input_ids.dim() == 1
        ids = input_ids.unsqueeze(0) if squeezed else input_ids
        if ids.size(0) != 1:
            raise ValueError(
                "forward_with_byte_override supports single-row only; "
                f"got batch size {ids.size(0)}."
            )
        L = ids.size(1)

        # Start from the standard byte buffer view for this row.
        bytes_row = self._byte_buffer.index_select(0, ids.reshape(-1)).clone()  # (L, pos_dim)
        lens_row = self._length_buffer.index_select(0, ids.reshape(-1)).clone()  # (L,)

        for p, raw in position_byte_overrides.items():
            if not (0 <= p < L):
                raise IndexError(f"override position {p} out of range [0, {L})")
            if not isinstance(raw, (bytes, bytearray)):
                raise TypeError(f"override at position {p} must be bytes; got {type(raw)}")
            trunc = utf8_safe_truncate(bytes(raw), self.pos_dim)
            new_len = len(trunc)
            bytes_row[p].zero_()
            if new_len > 0:
                bytes_row[p, :new_len] = torch.frombuffer(
                    bytearray(trunc), dtype=torch.uint8
                )
            lens_row[p] = new_len

        codec_out = kronecker_codec(
            bytes_row,
            lens_row,
            char_dim=self.char_dim,
            pos_dim=self.pos_dim,
            length_normalize=self.length_normalize,
            z_normalize=self.z_normalize,
        )
        emb = self.projection(codec_out.to(self.projection.weight.dtype))  # (L, d_model)
        return emb.unsqueeze(0) if not squeezed else emb

    # -------- save / load --------
    def state_dict(self, *args, **kwargs):
        """Return only the projection weight; byte buffers are rebuildable."""
        sd = super().state_dict(*args, **kwargs)
        # Drop non-persistent buffers we already declared non-persistent;
        # nn.Module already handles this, so the returned dict will just
        # have ``projection.weight``.
        return sd

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, d_model={self.d_model}, "
            f"char_dim={self.char_dim}, pos_dim={self.pos_dim}, "
            f"D={self.D}, mode={self.mode!r}"
        )
