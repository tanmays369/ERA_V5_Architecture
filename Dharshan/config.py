# Name: dharshan
# Axiom email: dharshanthaigu@gmail.com
# Pack: ind-ta-fw2
"""Config loading for the harness.

load_config(path, sets) -> Cfg      reads the YAML, applies "--set a.b=value" overrides
Cfg.get("a.b")                      returns a value; fails for a key that is not in SCHEMA (typo or decoration)
                                    and for a value that is still "open"
Cfg.reads                           the keys that were read; train.py writes them to run_info.json
source_comments(path)               {key path: comment text} read from the lines of the YAML file
"""
import re

import yaml

# Every key a config must have. check_config.py reports "missing" and "unknown" against this list,
# and Cfg.get() refuses any key that is not here, so a key cannot exist without code that reads it.
SCHEMA = [
    "general.seed", "general.deterministic", "general.num_threads",
    "tokenizer.name", "tokenizer.url", "tokenizer.sha256", "tokenizer.vocab_size", "tokenizer.eos_id",
    "embedding.type", "embedding.char_dim", "embedding.pos_dim", "embedding.mode", "embedding.projection_init",
    "embedding.length_normalize", "embedding.z_normalize", "embedding.weight_decay", "embedding.ke2_output_shape",
    "model.n_layers", "model.d_model", "model.n_heads", "model.n_kv_heads", "model.d_ff", "model.norm_position",
    "model.norm_eps", "model.rope_theta", "model.qk_norm", "model.init_std", "model.ffn", "model.norm",
    "model.bias", "model.dropout", "model.tie_embeddings",
    "attention.window", "attention.global_every",
    "moe.enabled", "moe.num_experts", "moe.top_k", "moe.shared_experts", "moe.router_sharing", "moe.router_scoring",
    "moe.router_dtype", "moe.balancing", "moe.bias_update_speed", "moe.seq_aux_loss_coef", "moe.dropless",
    "head.z_loss_coef", "head.mtp_heads", "head.mtp_loss_weight",
    "optimizer.matrices_2d", "optimizer.other_params", "optimizer.lr_muon", "optimizer.lr_adamw",
    "optimizer.weight_decay", "optimizer.betas", "optimizer.grad_clip", "optimizer.muon_momentum",
    "optimizer.muon_ns_steps",
    "schedule.type", "schedule.warmup_frac", "schedule.decay_frac", "schedule.min_lr_frac",
    "tokens.seq_len", "tokens.global_batch_seqs", "tokens.micro_batch_seqs", "tokens.train_steps",
    "data.source", "data.paths", "data.weights", "data.text_column", "data.val_frac", "data.val_windows_per_source",
    "logging.eval_every",
    "precision.dtype",
    "safety.stop_on_nan",
]

OPEN = "open"
TAGS = ("KE", "paper", "run", "admin", "ours", "open")


def flatten(d, prefix=""):
    """{'a': {'b': 1}} -> {'a.b': 1}. A leaf is anything that is not a dict (lists stay lists)."""
    out = {}
    for k, v in d.items():
        p = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, p + "."))
        else:
            out[p] = v
    return out


class Cfg:
    def __init__(self, tree):
        self.tree = tree
        self.flat = flatten(tree)
        self.reads = set()

    def get(self, key):
        if key not in SCHEMA:
            raise KeyError(f"config key {key!r} is not in SCHEMA (typo, or a key without code behind it)")
        if key not in self.flat:
            raise KeyError(f"config key {key!r} is missing")
        v = self.flat[key]
        if v == OPEN or v is None:
            raise ValueError(f"config key {key!r} is {'open' if v == OPEN else 'empty'}; give it a value (--set {key}=...)")
        self.reads.add(key)
        return v


def parse_set(s):
    key, sep, val = s.partition("=")
    if not sep or not key:
        raise ValueError(f"override must look like a.b=value, got {s!r}")
    return key, yaml.safe_load(val)


def load_config(path, sets=()):
    with open(path, encoding="utf-8") as f:
        tree = yaml.safe_load(f) or {}
    for s in sets:
        key, val = parse_set(s)
        d = tree
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return Cfg(tree)


_LINE = re.compile(r"^( *)([A-Za-z_][A-Za-z0-9_]*):(.*)$")


def source_comments(path):
    """Read the YAML lines and return {key path: comment text or None} for every leaf and section.

    Also returns the list of the first lines of the file (for the Name / Axiom email / Pack header check).
    """
    comments, stack, head = {}, [], []
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    for line in lines[:3]:
        head.append(line)
    for line in lines:
        m = _LINE.match(line)
        if not m:
            continue
        indent, key, rest = len(m.group(1)), m.group(2), m.group(3)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        value, sep, comment = rest.partition(" #")
        path_key = ".".join([k for _, k in stack] + [key])
        if value.strip() == "":
            stack.append((indent, key))      # a section header: no value on the line
            continue
        comments[path_key] = comment.strip() if sep else None
    return comments, head
