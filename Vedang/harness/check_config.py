#!/usr/bin/env python3
"""check_config.py - report every setting with no value or no (valid) source.

Usage:
    python harness/check_config.py configs/config_v0.yaml
    python harness/check_config.py configs/config_v0.yaml --strict      # open sizes and pending measurements become errors

Each setting is one line:   key: value   # src: <kind> <detail>
Allowed source kinds (first word after "src:"):
    kronecker-code        the kronecker-embeddings package
    kronecker-paper       arXiv 2605.29459
    measurement:<id>      a measurement from our tiny run: measurements/<id>.json (written by sweep.py / prepare_data.py)
    harness-code          a fixed choice in harness/train.py (listed in the summary for the team to confirm)
    TBD_by_TSAI_31Oct     only with value null: a size TSAI fills on 31 Oct

ERROR    no value, no source, a placeholder source, an unknown source kind, a line that cannot be parsed,
         a value with a TBD marker, a measurement that does not cover the setting, or a default that differs
         from the value the measurement chose.
OPEN     value null with the TBD_by_TSAI_31Oct marker (error with --strict).
PENDING  source names a measurement whose file does not exist yet (error with --strict).
Exit code: 0 when there are 0 errors, 1 otherwise.
"""
import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OPEN_MARKER = "TBD_by_TSAI_31Oct"
STATIC_KINDS = {"kronecker-code", "kronecker-paper", "harness-code"}
PLACEHOLDER_SOURCES = {"", "tbd", "todo", "?", "??", "n/a", "na", "none", "-"}
KEY_RE = re.compile(r"^(?P<indent> *)(?P<key>[A-Za-z_][A-Za-z0-9_]*):(?P<rest>.*)$")
SRC_RE = re.compile(r"#\s*src:\s*(?P<src>.*)$")


def parse_sources(text):
    """Return ({dotted.path: source_or_None}, [line problems]). YAML parsers drop comments, so read them from the text."""
    sources, problems, stack = {}, [], []
    for lineno, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = KEY_RE.match(raw)
        if not m:
            problems.append(f"line {lineno}: cannot understand line: {raw.strip()!r}")
            continue
        indent, key, rest = len(m["indent"]), m["key"], m["rest"]
        while stack and stack[-1][0] >= indent:
            stack.pop()
        value_part = SRC_RE.sub("", rest).split("#")[0].strip()
        if value_part == "" and not SRC_RE.search(rest):
            stack.append((indent, key))  # a section header like "model:"
            continue
        path = ".".join([k for _, k in stack] + [key])
        sm = SRC_RE.search(rest)
        sources[path] = sm["src"].strip() if sm else None
    return sources, problems


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else str(k)
        out.update(flatten(v, p) if isinstance(v, dict) else {p: v})
    return out


def same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def check(path, strict=False, mdir=None):
    """Return (errors, opens, pendings, kind_counts, n_settings)."""
    mdir = Path(mdir) if mdir else ROOT / "measurements"
    errors, opens, pendings, kinds = [], [], [], Counter()
    text = Path(path).read_text()
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        return [f"YAML does not parse: {e}"], [], [], kinds, 0
    if not isinstance(data, dict) or not data:
        return ["file is empty or not a mapping"], [], [], kinds, 0

    values = flatten(data)
    sources, problems = parse_sources(text)
    errors.extend(problems)
    for p in sorted(set(values) - set(sources)):
        errors.append(f"{p}: YAML sees a setting here but the line is not 'key: value  # src: ...' (missing source or value?)")
    for p in sorted(set(sources) - set(values)):
        errors.append(f"{p}: line found in text but YAML did not read it as a setting")

    cache = {}
    for p in sorted(set(values) & set(sources)):
        val, src = values[p], sources[p]
        if src is None:
            errors.append(f"{p}: no source (add '# src: <kind> <detail>')")
            continue
        kind = src.split()[0].rstrip(",;:(") if src.split() else ""
        if src.lower() in PLACEHOLDER_SOURCES or kind.lower() in PLACEHOLDER_SOURCES:
            errors.append(f"{p}: source is a placeholder ({src!r})")
            continue
        is_empty = val is None or (isinstance(val, str) and val.strip() == "")

        if kind == OPEN_MARKER:
            kinds["TBD_by_TSAI_31Oct"] += 1
            if is_empty:
                (errors if strict else opens).append(f"{p}: size still open" + (" (--strict)" if strict else ", waiting for TSAI"))
            else:
                errors.append(f"{p}: has a value but source still says {OPEN_MARKER}; replace the source")
            continue
        if is_empty:
            errors.append(f"{p}: no value")
            continue

        if kind in STATIC_KINDS:
            kinds[kind] += 1
        elif kind.startswith("measurement:") and len(kind) > len("measurement:"):
            mid = kind.split(":", 1)[1]
            kinds["measurement"] += 1
            mfile = mdir / f"{mid}.json"
            if not mfile.exists():
                (errors if strict else pendings).append(f"{p}: measurement '{mid}' not run yet (missing {mfile.name})")
                continue
            if mid not in cache:
                try:
                    cache[mid] = json.loads(mfile.read_text())
                except json.JSONDecodeError as e:
                    cache[mid] = None
                    errors.append(f"measurements/{mid}.json is not valid JSON: {e}")
            js = cache[mid]
            if js is None:
                continue
            chosen = js.get("chosen", {})
            if p not in chosen:
                errors.append(f"{p}: measurement '{mid}' does not cover this setting (its chosen values: {sorted(chosen)})")
            elif not same(chosen[p], val):
                errors.append(f"{p}: default {val!r} differs from what measurement '{mid}' chose ({chosen[p]!r})")
            if js.get("tokenizer") and js["tokenizer"] != data.get("tokenizer", {}).get("name_or_path"):
                errors.append(f"{p}: measurement '{mid}' used tokenizer {js['tokenizer']!r}, config says "
                              f"{data.get('tokenizer', {}).get('name_or_path')!r}")
        else:
            errors.append(f"{p}: unknown source kind {kind!r}; use kronecker-code, kronecker-paper, "
                          f"measurement:<id>, harness-code or {OPEN_MARKER}")
    return errors, opens, pendings, kinds, len(values)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--strict", action="store_true", help="open sizes and pending measurements count as errors")
    ap.add_argument("--measurements", help="folder with measurement JSON files (default: <team folder>/measurements)")
    args = ap.parse_args()

    errors, opens, pendings, kinds, n = check(args.config, args.strict, args.measurements)
    for e in errors:
        print(f"ERROR    {e}")
    for o in opens:
        print(f"OPEN     {o}")
    for pnd in pendings:
        print(f"PENDING  {pnd}")
    by_kind = ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())) or "none"
    print(f"\nsources: {by_kind}")
    if kinds.get("harness-code"):
        print(f"note: {kinds['harness-code']} settings use 'harness-code'; the team should confirm that this counts as a source")
    print(f"{args.config}: {n} settings, {len(errors)} errors, {len(opens)} open sizes, {len(pendings)} pending measurements")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
