# Name: Revant Gupta
# Pack: ind-hi-fw2
"""Stream each dataset source, tokenize, and write <name>.{val,train}.bin (uint32 LE) + manifest.json.

Usage: python prepare_data.py --config CFG [--set k=v ...]. Idempotent per source. No shuffle: val = first docs.
"""
import argparse
import hashlib
import itertools
import json
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import tok as tok_mod  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_tokenizer(cfg):
    path, want = cfg["paths"]["tokenizer"], cfg["tokenizer"]["sha256"]
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve(cfg["tokenizer"]["url"], path + ".part")
        os.replace(path + ".part", path)
    got = sha256(path)
    if got != want:
        sys.exit(f"tokenizer sha256 mismatch: {path} has {got}, config wants {want}")
    return path


def write_split(tok, texts, eos, path):
    """Tokenize texts (EOS after each doc), write uint32; return (tokens, utf8 bytes, sha256)."""
    ids = []
    for enc, text in zip(tok.encode_batch(texts, add_special_tokens=False), texts):
        ids.extend(enc.ids)
        ids.append(eos)
    arr = np.asarray(ids, dtype="<u4")
    arr.tofile(path)
    return len(arr), sum(len(t.encode("utf-8")) for t in texts), sha256(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    cfg = config.load_config(args.config, args.set)

    from tokenizers import Tokenizer
    os.environ.setdefault("HF_HOME", os.path.abspath("work/hf"))  # scratch stays under work/ (contract)
    import datasets

    tok = Tokenizer.from_file(ensure_tokenizer(cfg))
    out = cfg["paths"]["data"]
    os.makedirs(out, exist_ok=True)
    mpath = os.path.join(out, "manifest.json")
    manifest = {"tokenizer_sha256": cfg["tokenizer"]["sha256"], "datasets": {}}
    if os.path.exists(mpath):
        with open(mpath) as f:
            manifest = json.load(f)
        if manifest["tokenizer_sha256"] != cfg["tokenizer"]["sha256"]:
            sys.exit(f"{mpath} was built with another tokenizer; use a fresh paths.data")
    n_val, n_train = cfg["data"]["heldout_docs_per_source"], cfg["data"]["docs_per_source"]
    eos = cfg["tokenizer"]["eos_id"]

    for ds in cfg["datasets"]:
        name = ds["name"]
        tr, va = os.path.join(out, f"{name}.train.bin"), os.path.join(out, f"{name}.val.bin")
        entry = manifest["datasets"].get(name)
        if entry and entry["docs"] == [n_val, n_train] and os.path.exists(tr) and os.path.exists(va):
            print(f"{name}: exists, skipped")
            continue
        t0 = time.time()
        stream = datasets.load_dataset(ds["hf"], ds.get("subset"), split=ds["split"], streaming=True)
        texts = [row[ds["text_field"]] for row in itertools.islice(stream, n_val + n_train)]
        if len(texts) < n_val + n_train:
            sys.exit(f"{name}: only {len(texts)} docs, need {n_val + n_train}")
        val_tokens, val_bytes, val_sha = write_split(tok, texts[:n_val], eos, va)
        train_tokens, _, train_sha = write_split(tok, texts[n_val:], eos, tr)
        manifest["datasets"][name] = {
            "hf": ds["hf"], "subset": ds.get("subset"), "docs": [n_val, n_train],
            "train_tokens": train_tokens, "val_tokens": val_tokens, "val_bytes": val_bytes,
            "train_sha256": train_sha, "val_sha256": val_sha}
        with open(mpath + ".tmp", "w") as f:
            json.dump(manifest, f, indent=1)
        os.replace(mpath + ".tmp", mpath)  # manifest written after each source: a crash loses one source at most
        print(f"{name}: docs {n_val}+{n_train}, tokens val {val_tokens} train {train_tokens}, {time.time() - t0:.1f}s",
              flush=True)
    tok_mod.byte_buffers(cfg["paths"]["tokenizer"], cfg["embedding"]["pos_dim"])  # cache before torchrun ranks race on it


if __name__ == "__main__":
    main()
