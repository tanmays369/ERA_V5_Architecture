"""The tokenizer, and the small adapter the Kronecker embedding needs.

The Kronecker embedding builds its byte table from a Hugging Face (transformers) tokenizer: it calls
get_vocab(), convert_ids_to_tokens(), decode() and all_special_ids. Our test tokenizer is a plain
`tokenizers` JSON file (the session-2 BPE tokenizer for English, Hindi, Bengali and Telugu), so this
adapter offers those four calls on top of it. No transformers install is needed.

token_byte_lengths() gives the number of UTF-8 bytes each token stands for in running text, which turns a
loss in nats per token into bits per byte (the unit all G5 results are reported in, plan sheet week 2).
"""
import numpy as np
from tokenizers import Tokenizer

METASPACE = "▁"  # "▁", the word-start marker of a Metaspace tokenizer


class HFLikeTokenizer:
    def __init__(self, path):
        self.tk = Tokenizer.from_file(str(path))
        self.vocab_size = self.tk.get_vocab_size()
        self.all_special_ids = []

    # ---- the calls the Kronecker embedding uses ----
    def get_vocab(self):
        return self.tk.get_vocab()

    def convert_ids_to_tokens(self, token_id):
        return self.tk.id_to_token(int(token_id))

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return self.tk.decode(list(ids), skip_special_tokens=skip_special_tokens)

    # ---- for the harness ----
    def encode(self, text):
        return self.tk.encode(text).ids

    def token_byte_lengths(self):
        """UTF-8 bytes per token in running text; the Metaspace marker counts as one space byte."""
        out = np.zeros(self.vocab_size, dtype=np.int32)
        for tid in range(self.vocab_size):
            piece = self.tk.id_to_token(tid) or ""
            out[tid] = len(piece.replace(METASPACE, " ").encode("utf-8"))
        return out
