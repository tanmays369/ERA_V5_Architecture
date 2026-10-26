"""
kronecker_embeddings — byte-level structured token representations.

Quickstart::

    from kronecker_embeddings import KroneckerEmbedding
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    emb = KroneckerEmbedding(vocab_size=tok.vocab_size, d_model=512, tokenizer=tok)

    # Use it like nn.Embedding.
    ids = tok("hello world", return_tensors="pt").input_ids
    out = emb(ids)         # shape (1, L, 512)

See ``README.md`` for the full method description and the paper for math.
"""

from ._version import __version__
from .codec import codec_output_dim, encode_single, kronecker_codec
from .embedding import KroneckerEmbedding
from .tokenizer_utils import (
    build_byte_buffer,
    coverage_stats,
    token_id_to_bytes,
    utf8_safe_truncate,
)

__all__ = [
    "__version__",
    "KroneckerEmbedding",
    "kronecker_codec",
    "encode_single",
    "codec_output_dim",
    "build_byte_buffer",
    "coverage_stats",
    "token_id_to_bytes",
    "utf8_safe_truncate",
]
