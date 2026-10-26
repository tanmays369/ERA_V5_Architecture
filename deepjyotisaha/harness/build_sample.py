"""Build the mixed English + Indic training sample for the tiny test models.

Usage:
    python harness/build_sample.py [configs/tiny/tiny_dense.yaml]

Reads the "harness_sample" section of the config's data mix file (configs/data_mix.yaml):
  English  the 500 FineWeb-Edu documents of the week-1 diagnosis pack (data/en-fineweb-edu.parquet)
  Hindi, Bengali, Telugu   the cleaned session-2 text files
Each source is tokenized with the config's tokenizer and split into a training part and a held-out
validation part (the last validation_fraction of its text). The input files go in data/ (see data/README.md).
The result goes to work/sample/:
  <source>_train.npy, <source>_val.npy    token ids (uint16)
  token_bytes.npy                         UTF-8 bytes per token id, for bits per byte
  manifest.json                           sizes, token counts and the sha256 of every input file
The sample stays outside the lab repository; share manifest.json, not the data.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configlib import get, load_config, require  # noqa: E402
from tokenizer import HFLikeTokenizer  # noqa: E402

SOURCES = [("english", "english_pack"), ("hindi", "hindi_text"), ("bengali", "bengali_text"), ("telugu", "telugu_text")]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_source(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return "\n\n".join(pq.read_table(path).column("text").to_pylist())
    return path.read_text(encoding="utf-8")


def build(config_path):
    cfg = load_config(config_path)
    hs = cfg["data_mix"]["harness_sample"]
    tok = HFLikeTokenizer(require(cfg, "tokenizer", "path"))
    out = Path(hs["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"tokenizer": Path(cfg["tokenizer"]["path"]).name, "tokenizer_sha256": sha256(cfg["tokenizer"]["path"]),
                "vocab_size": tok.vocab_size, "validation_fraction": hs["validation_fraction"], "sources": {}}
    missing = [str(hs[key]) for _, key in SOURCES if not Path(hs[key]).exists()]
    if missing:
        raise SystemExit("These input files are missing (see data/README.md for where to get them):\n  "
                         + "\n  ".join(missing))
    for name, key in SOURCES:
        text = read_source(hs[key])
        cut = int(len(text) * (1 - hs["validation_fraction"]))
        cut = text.rfind("\n", 0, cut) + 1 or cut  # split at a line break
        parts = {"train": text[:cut], "val": text[cut:]}
        info = {"file": Path(hs[key]).name, "sha256": sha256(hs[key]), "share": hs[name + "_share"]}
        for split, t in parts.items():
            ids = np.array(tok.encode(t), dtype=np.uint16)
            np.save(out / f"{name}_{split}.npy", ids)
            info[f"{split}_chars"] = len(t)
            info[f"{split}_bytes"] = len(t.encode("utf-8"))
            info[f"{split}_tokens"] = int(ids.size)
        manifest["sources"][name] = info
        print(f"{name:8s} train {info['train_tokens']:>9,} tokens ({info['train_bytes']:>9,} bytes) · "
              f"val {info['val_tokens']:>7,} tokens · share {info['share']}")
    np.save(out / "token_bytes.npy", tok.token_byte_lengths())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"Wrote the sample to {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", nargs="?", default=str(Path(__file__).resolve().parents[1] / "configs" / "tiny" / "tiny_dense.yaml"))
    build(ap.parse_args().config)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
