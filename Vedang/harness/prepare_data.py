#!/usr/bin/env python3
"""prepare_data.py - build the mixed Hindi + English token file that train.py reads.

Usage (from teams/05-architecture):
    python harness/prepare_data.py                              # 6M tokens per language, Wikipedia
    python harness/prepare_data.py --tokens_per_source 300000   # small and fast
    python harness/prepare_data.py --offline                    # built-in sample text, no dataset download
    python harness/prepare_data.py --source bn=wikimedia/wikipedia:20231101.bn   # add another Indic language

How it avoids hanging
  * every language is downloaded by a separate child process. If it does not finish within --max_seconds,
    the main script kills it and keeps whatever documents it had already written to disk.
  * if a language ends with almost nothing (network down or throttled), it is replaced by a SYNTHETIC
    sample (simple Hindi / English sentences from a small grammar). Synthetic data is good enough to prove the
    harness runs and is deterministic. It is NOT good enough for experiment results; the script says so loudly
    and records it in mixed_stats.json ("synthetic": true).
  * the order of documents is fixed (no shuffling), so two runs build the same file.

Output: data/mixed_hi_en_tokens.npy (int32) and data/mixed_stats.json. Share mixed_stats.json in the team chat.
"""
import argparse
import faulthandler
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "20")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCES = ["hi=wikimedia/wikipedia:20231101.hi", "en=wikimedia/wikipedia:20231101.en"]
MIN_USEFUL_TOKENS = 1000   # a source with fewer tokens than this is treated as failed
SYNTHETIC_MAX_TOKENS = 500_000   # synthetic text is repetitive; more than this adds nothing

# ------------------------------------------------------------------ synthetic fallback text
_SYN = {
    "hi": dict(
        subj=["बिल्ली", "कुत्ता", "लड़की", "किसान", "शिक्षक", "चिड़िया", "मेरा दोस्त", "छोटा बच्चा"],
        obj=["गेंद", "किताब", "पेड़", "रोटी", "दरवाज़ा", "चाँद", "पुरानी घड़ी", "हरा पौधा"],
        when=["", "आज ", "कल ", "रात को ", "सुबह "],
        tpl=["{w}{s} को {o} पसंद है।\n", "{w}{s} के पास {o} है।\n", "{w}{s} को {o} चाहिए।\n"]),
    "en": dict(
        subj=["the cat", "a dog", "the bird", "my friend", "the teacher", "a small fox", "the farmer", "a young girl"],
        obj=["a ball", "the moon", "a red book", "the old tree", "some bread", "the green door", "an old clock"],
        when=["", "today ", "yesterday ", "at night ", "this morning "],
        tpl=["{w}{s} likes {o}.\n", "{w}{s} has {o}.\n", "{w}{s} wants {o}.\n"]),
}


def synthetic_rows(lang, seed=0):
    g = _SYN.get(lang, _SYN["en"])
    rng = random.Random(seed)
    while True:
        yield "".join(rng.choice(g["tpl"]).format(w=rng.choice(g["when"]), s=rng.choice(g["subj"]),
                                                  o=rng.choice(g["obj"])) for _ in range(40))


# ------------------------------------------------------------------ worker (runs in a child process)
def load_tokenizer(name):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def rows_for(spec):
    if spec["dataset"] == "synthetic":
        return synthetic_rows(spec["config"])
    from datasets import load_dataset
    print(f"[{spec['name']}] opening {spec['dataset']}:{spec['config']} (streaming) ...", flush=True)
    ds = load_dataset(spec["dataset"], spec["config"], split="train", streaming=True)
    print(f"[{spec['name']}] stream opened, reading documents ...", flush=True)
    return (r["text"] for r in ds)


def run_worker(spec, tok=None):
    """Tokenize documents and append them to <outdir>/<name>.tok/.len/.chr after every batch."""
    tok = tok or load_tokenizer(spec["tokenizer"])
    eos = [tok.eos_token_id] if tok.eos_token_id is not None else tok("\n\n", add_special_tokens=False)["input_ids"]
    out = Path(spec["outdir"])
    ft, fl, fc = (open(out / f"{spec['name']}.{x}", "wb") for x in ("tok", "len", "chr"))
    n_tok, n_docs, texts, t0 = 0, 0, [], time.time()

    def flush():
        nonlocal n_tok, n_docs
        ids_list = tok(texts, add_special_tokens=False)["input_ids"]
        arrs = [np.asarray(ids + eos, dtype=np.int32) for ids in ids_list]
        np.concatenate(arrs).tofile(ft)                      # tokens first, then lengths, then chars:
        np.asarray([len(a) for a in arrs], dtype=np.int32).tofile(fl)   # a document counts only when all three exist
        np.asarray([len(t) for t in texts], dtype=np.int32).tofile(fc)
        for f in (ft, fl, fc):
            f.flush()
        n_tok += sum(len(a) for a in arrs)
        n_docs += len(arrs)
        texts.clear()
        print(f"[{spec['name']}] {n_docs:,} docs, {n_tok:,}/{spec['budget']:,} tokens, {time.time() - t0:.0f}s", flush=True)

    for text in rows_for(spec):
        text = text.strip()[: spec["max_chars"]]
        if len(text) < spec["min_chars"]:
            continue
        texts.append(text)
        if len(texts) == 16:
            flush()
            if n_tok >= spec["budget"]:
                break
    if texts and n_tok < spec["budget"]:
        flush()
    for f in (ft, fl, fc):
        f.close()


# ------------------------------------------------------------------ parent
def worker_cmd(spec):   # separate function so it can be replaced in tests
    return [sys.executable, os.path.abspath(__file__), "--worker", json.dumps(spec)]


def _load(outdir, name, ext):
    path = Path(outdir) / f"{name}.{ext}"
    return np.fromfile(path, dtype=np.int32) if path.exists() else np.zeros(0, dtype=np.int32)


def read_source(outdir, name):
    tok, lens, chars = (_load(outdir, name, x) for x in ("tok", "len", "chr"))   # missing file = worker died early
    n = min(len(lens), len(chars))
    n = int((np.cumsum(lens[:n]) <= len(tok)).sum())         # drop a document cut off by the kill
    lens, chars = lens[:n], chars[:n]
    ends = np.cumsum(lens)
    docs = [tok[e - l:e] for e, l in zip(ends, lens)]
    return docs, int(lens.sum()), int(chars.sum())


def collect_source(spec, max_seconds):
    Path(spec["outdir"]).mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(worker_cmd(spec), timeout=max_seconds)
        if r.returncode != 0:
            print(f"[{spec['name']}] worker exited with code {r.returncode}; keeping what it wrote", flush=True)
    except subprocess.TimeoutExpired:
        print(f"[{spec['name']}] TIMEOUT after {max_seconds}s: killed the download, keeping what it wrote", flush=True)
    return read_source(spec["outdir"], spec["name"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--tokenizer", default="theschoolofai/BrahmicTokenizer-131K")
    ap.add_argument("--source", action="append", metavar="NAME=DATASET:CONFIG", help="repeatable; default Hindi + English Wikipedia")
    ap.add_argument("--tokens_per_source", type=int, default=6_000_000)
    ap.add_argument("--min_chars", type=int, default=200)
    ap.add_argument("--max_chars", type=int, default=4000, help="cut each document to this many characters")
    ap.add_argument("--max_seconds", type=int, default=240, help="hard time limit per source")
    ap.add_argument("--offline", action="store_true", help="use the built-in synthetic sample for every source")
    ap.add_argument("--no_fallback", action="store_true", help="fail instead of using synthetic text when a source gives nothing")
    ap.add_argument("--out", default="data/mixed_hi_en_tokens.npy")
    ap.add_argument("--measurements_dir", default="measurements", help="where measurement JSON files are written")
    args = ap.parse_args()

    if args.worker:
        run_worker(json.loads(args.worker))
        return

    sources = args.source or DEFAULT_SOURCES
    faulthandler.dump_traceback_later(args.max_seconds * len(sources) * 2 + 300, exit=True)  # last-resort kill
    print("loading tokenizer ...", flush=True)
    tok = load_tokenizer(args.tokenizer)
    kron_vocab = max(tok.get_vocab().values()) + 1
    print(f"tokenizer ok: vocab (max id + 1) = {kron_vocab}", flush=True)

    tmp = tempfile.mkdtemp(prefix="prep_")
    per_source, stats = {}, {"tokenizer": args.tokenizer, "vocab": kron_vocab, "sources": {}}
    for s in sources:
        name, rest = s.split("=", 1)
        dataset, config = rest.rsplit(":", 1)
        if args.offline:
            dataset, config = "synthetic", name
        spec = dict(name=name, dataset=dataset, config=config, tokenizer=args.tokenizer, budget=args.tokens_per_source,
                    min_chars=args.min_chars, max_chars=args.max_chars, outdir=os.path.join(tmp, "real"))
        if dataset == "synthetic":
            spec["budget"] = min(spec["budget"], SYNTHETIC_MAX_TOKENS)
        docs, n_tok, n_chars = collect_source(spec, args.max_seconds)
        synthetic = dataset == "synthetic"
        if n_tok < MIN_USEFUL_TOKENS:
            if args.no_fallback:
                sys.exit(f"[{name}] collected almost nothing and --no_fallback is set")
            print(f"\n*** [{name}] got {n_tok} tokens: network or dataset failed. Using SYNTHETIC sample text. "
                  f"Fine for smoke and determinism tests, NOT for experiment results. ***\n", flush=True)
            spec.update(dataset="synthetic", config=name, outdir=os.path.join(tmp, "synthetic"),
                        budget=min(args.tokens_per_source, SYNTHETIC_MAX_TOKENS))
            docs, n_tok, n_chars = collect_source(spec, max(60, args.max_seconds))
            synthetic = True
        per_source[name] = docs
        stats["sources"][name] = {"docs": len(docs), "tokens": n_tok, "chars": n_chars, "synthetic": synthetic,
                                  "chars_per_token": round(n_chars / max(1, n_tok), 3)}
        print(f"[{name}] done: {len(docs):,} docs, {n_tok:,} tokens, synthetic={synthetic}", flush=True)

    merged, idx, names = [], 0, list(per_source)
    while any(idx < len(per_source[n]) for n in names):
        for n in names:
            if idx < len(per_source[n]):
                merged.append(per_source[n][idx])
        idx += 1
    tokens = np.concatenate(merged).astype(np.int32)
    assert tokens.max() < kron_vocab, "token id outside tokenizer vocab"
    stats["total_tokens"] = int(len(tokens))
    stats["any_synthetic"] = any(v["synthetic"] for v in stats["sources"].values())

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, tokens)
    (out.parent / "mixed_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    faulthandler.cancel_dump_traceback_later()
    mdir = Path(args.measurements_dir) if Path(args.measurements_dir).is_absolute() else ROOT / args.measurements_dir
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "tokenizer_vocab.json").write_text(json.dumps({     # the measurement that model.vocab_size cites
        "id": "tokenizer_vocab", "date": time.strftime("%Y-%m-%d"), "tokenizer": args.tokenizer,
        "how": "max token id + 1 over tokenizer.get_vocab(), measured by harness/prepare_data.py",
        "chosen": {"model.vocab_size": kron_vocab}}, indent=2))
    print(f"\nsaved {len(tokens):,} tokens to {out}; wrote {mdir / 'tokenizer_vocab.json'}")
    if stats["any_synthetic"]:
        print("WARNING: part of this file is synthetic sample text. Do not report experiment results from it.")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
