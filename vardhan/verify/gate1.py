"""Independent Gate-1 verifier for any team-05 submission.

The Saturday gate is: "config loads, check_config.py gives 0 errors, two
seed-1234 runs match, the loss goes down, no NaN, TSAI approves the plan."

Every member folder answers that with its own checker, its own config schema
and its own evidence layout. This tool answers it the same way for all of
them, so the lead can compare submissions instead of taking each one's word.

Two design rules:

1. **It never imports or executes code from the folder it is verifying.**
   The source audit re-implements the rule from the YAML; the run checks read
   the evidence files. A submission cannot pass by shipping a checker that
   prints "0 errors".
2. **It is schema-agnostic.** Both config shapes in use are handled: the flat
   `settings:` record list, and the nested mapping with trailing
   `# source:` comments. Five loss-file formats are handled (see parse_losses).

Usage:
    python verify/gate1.py ../shivman
    python verify/gate1.py --all ..
    python verify/gate1.py ../Udit --runs evidence/.../full_a evidence/.../full_b
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("needs pyyaml:  pip install pyyaml")

G8_METRICS = ["tokens_per_second", "loss_ntp", "loss_mtp", "loss", "lr", "gpu_alloc_gb"]
LOSS_NAMES = ("losses.txt", "loss.txt", "losses.tsv", "losses.csv",
              "run1.csv", "run2.csv", "total_loss.txt")
# A config may carry its sources in a separate file instead of per-line
# comments. That is a different convention, not a missing source.
SOURCE_FILES = ("sources.md", "source.md", "readme.md")


# ---------------------------------------------------------------- loss files

def _to_float(tok: str):
    """Accept decimal and C99 hex floats (0x1.76c18p+3)."""
    try:
        return float.fromhex(tok) if tok.lower().startswith(("0x", "-0x")) else float(tok)
    except ValueError:
        return None


def parse_losses(path: Path):
    """Return [(step, [values...])]. Handles the five layouts in this repo:
    bare float per line; `step hexfloat...`; `000001 12.3`; CSV with header;
    tab-separated `step<TAB>loss`."""
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows, header = [], None
    for i, line in enumerate(raw):
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"[,\t ]+", line)
        vals = [_to_float(p) for p in parts]
        if all(v is None for v in vals):
            if i == 0 and header is None:
                header = parts          # CSV/TSV header row
                continue
            continue                     # a stray non-numeric line
        if len(vals) == 1:
            rows.append((len(rows), [vals[0]]))
        else:
            # first column is a step index if it is a whole number
            first = vals[0]
            if first is not None and float(first).is_integer() and len(vals) > 1:
                rows.append((int(first), [v for v in vals[1:] if v is not None]))
            else:
                rows.append((len(rows), [v for v in vals if v is not None]))
    return rows, header


def pair_key(p: Path) -> str:
    """Collapse the a/b (or 1/2) twin marker so two runs of a pair share a key."""
    s = str(p)
    for a, b in ((r"run_?[ab]\b", "runX"), (r"_[ab]\b", "_X"),
                 (r"run[12]\b", "runX"), (r"\b[12]\.csv", "X.csv"),
                 # R1a / R1b, trial2a / trial2b: a trailing a/b on a dir name
                 (r"([A-Za-z]+\d+)[ab](/|$)", r"\1X\2")):
        s = re.sub(a, b, s, flags=re.I)
    return s


def find_run_pairs(root: Path):
    files = [p for p in root.rglob("*") if p.is_file() and p.name.lower() in LOSS_NAMES]
    groups: dict[str, list[Path]] = {}
    for f in files:
        groups.setdefault(pair_key(f.relative_to(root)), []).append(f)
    return [sorted(v)[:2] for v in groups.values() if len(v) >= 2], files


# ------------------------------------------------------------------- configs

def audit_config(path: Path):
    """Return (n_settings, errors, open_keys) without running their checker."""
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = yaml.safe_load(text)
    except Exception as e:
        return 0, [f"{path.name}: does not parse as YAML: {e}"], [], None
    if data is None:
        return 0, [f"{path.name}: empty"], [], None

    # Form A: flat record list under `settings:`
    if isinstance(data, dict) and isinstance(data.get("settings"), list):
        errs, open_keys, n = [], [], 0
        for item in data["settings"]:
            if not isinstance(item, dict):
                errs.append(f"malformed entry: {item!r}")
                continue
            n += 1
            k = item.get("key", "<no key>")
            is_open = bool(item.get("open"))
            if "value" not in item:
                errs.append(f"{k}: no value")
            elif item["value"] is None and not is_open:
                errs.append(f"{k}: null value not marked open")
            if is_open:
                open_keys.append(k)
            if not str(item.get("source", "")).strip():
                errs.append(f"{k}: no source")
        return n, errs, open_keys, None

    # Form B: nested mapping, sources as trailing `# source:` comments.
    # First pass decides the convention: if essentially nothing carries an
    # inline source but a SOURCES.md / README sits beside the config, the
    # author is using an external source table. That is a convention this
    # tool cannot grade, not a file full of unsourced settings.
    rows = []
    stack: list[str] = []
    for ln, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.strip().startswith("#"):
            continue
        m = re.match(r"^(\s*)([A-Za-z_][\w.-]*):(.*)$", line)
        if not m:
            continue
        indent, key, rest = len(m.group(1)), m.group(2), m.group(3)
        stack = stack[: indent // 2]
        stack.append(key)
        value_part = rest.split("#", 1)[0].strip()
        if not value_part:
            continue                      # a parent key, not a setting
        comment = rest.split("#", 1)[1] if "#" in rest else ""
        rows.append((ln, ".".join(stack), value_part, comment))

    n = len(rows)
    inline = sum(1 for _, _, _, c in rows if "source:" in c.lower())
    external = None
    if n and inline / n < 0.2:
        for cand in list(path.parent.iterdir()) + list(path.parent.parent.iterdir()):
            if cand.is_file() and cand.name.lower() in SOURCE_FILES \
                    and re.search(r"source", cand.read_text(errors="replace"), re.I):
                external = cand
                break

    errs, open_keys = [], []
    for ln, dotted, value_part, comment in rows:
        has_src = "source:" in comment.lower() or external is not None
        if not has_src:
            errs.append(f"{dotted} (line {ln}): no source")
        if value_part in ("null", "~"):
            # A null whose source explains the wait is a declared-open size,
            # not a missing value. Only an unexplained null is an error.
            if comment.strip() or external is not None:
                open_keys.append(dotted)
            else:
                errs.append(f"{dotted} (line {ln}): null value with no explanation")
    return n, errs, open_keys, external


def find_configs(root: Path):
    out = [p for p in root.rglob("*.yaml") if "config" in p.name.lower()
           and "small" not in p.parts and "tiny" not in p.name.lower()]
    return sorted(out)


# -------------------------------------------------------------------- checks

def check_runs(a: Path, b: Path):
    res = []
    ra, _ = parse_losses(a)
    rb, _ = parse_losses(b)
    ok_len = len(ra) == len(rb) and len(ra) > 0
    res.append(("two runs have the same number of steps",
                ok_len, f"{len(ra)} vs {len(rb)}"))
    if ok_len:
        same = all(x[1] == y[1] for x, y in zip(ra, rb))
        res.append(("the two loss series are numerically identical", same,
                    "identical" if same else
                    f"first differing step {next(i for i,(x,y) in enumerate(zip(ra,rb)) if x[1]!=y[1])}"))
        res.append(("the files are byte-identical", a.read_bytes() == b.read_bytes(),
                    "exact" if a.read_bytes() == b.read_bytes() else "differ (numerically equal is enough)"))
    flat = [v for _, vs in ra for v in vs if v is not None]
    nans = sum(1 for v in flat if math.isnan(v) or math.isinf(v))
    res.append(("no NaN or inf in the loss values", nans == 0, f"{nans} found"))
    if ra:
        first, last = ra[0][1][0], ra[-1][1][0]
        res.append(("the loss goes down", last < first, f"{first:.8f} -> {last:.8f}"))
    return res


# Known synonyms, so a renamed metric is reported as renamed rather than
# missing. The leaf name after any "namespace/" prefix is matched first.
G8_ALIASES = {
    "loss_ntp": ("ce_loss", "ce", "ntp", "loss_ce"),
    "loss_mtp": ("mtp", "mtp_loss"),
    "loss": ("total_loss", "total"),
    "lr": ("adamw_lr", "muon_lr", "learning_rate"),
    "gpu_alloc_gb": ("gpu_alloc", "gpu_mem_gb", "cuda_alloc_gb"),
    "tokens_per_second": ("tokens_per_sec", "tps"),
}


def check_metrics(run_dir: Path):
    """Report how the six names G8 was promised map onto what was logged.

    This is deliberately NOT a gate item: the Saturday gate is config, checker,
    matching seeds, falling loss and no NaN. The metric names were a Tuesday
    handoff to G8, so a mismatch is reported as a note for the integration.
    """
    m = run_dir / "metrics.jsonl"
    if not m.exists():
        return None
    rows = []
    for line in m.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not rows:
        return None
    keys = set(rows[0]).intersection(*[set(r) for r in rows[1:]]) if len(rows) > 1 else set(rows[0])
    leaves = {k.split("/")[-1]: k for k in keys}

    exact, renamed, absent = [], [], []
    for want in G8_METRICS:
        if want in keys:
            exact.append(want)
        elif want in leaves:
            renamed.append(f"{want} -> {leaves[want]}")
        else:
            hit = next((leaves[a] for a in G8_ALIASES.get(want, ()) if a in leaves), None)
            (renamed.append(f"{want} -> {hit}") if hit else absent.append(want))
    return len(rows), exact, renamed, absent


# --------------------------------------------------------------------- report

def verify(root: Path) -> bool:
    print(f"\n=== {root.name} " + "=" * max(0, 60 - len(root.name)))
    results, notes = [], []

    configs = find_configs(root)
    if not configs:
        results.append(("a config file is present", False, "none found"))
    for c in configs:
        n, errs, open_keys, external = audit_config(c)
        rel = c.relative_to(root)
        if external is not None:
            notes.append(f"{rel}: sources are in {external.name}, not per-line comments "
                         f"- this tool checks values only for that file")
        results.append((f"{rel} loads and every setting has a value and a source",
                        not errs, f"{n} settings, {len(errs)} errors, {len(open_keys)} declared open"))
        for e in errs[:5]:
            print(f"    - {e}")
        if len(errs) > 5:
            print(f"    - ... and {len(errs) - 5} more")
        # The lead's rule: "Size values stay empty. TSAI fills them on 31 Oct."
        if "small" not in rel.name.lower() and not open_keys:
            notes.append(f"{rel}: no setting is left open. config_v0 is supposed to keep "
                         f"the sizes empty until TSAI fills them on 31 Oct - worth a look")

    pairs, all_loss = find_run_pairs(root)
    if not pairs:
        results.append(("a pair of same-seed runs is present", False,
                        f"{len(all_loss)} loss file(s), no a/b pair found"))
    for a, b in pairs[:3]:
        print(f"  runs: {a.relative_to(root)}  vs  {b.relative_to(root)}")
        results.extend(check_runs(a, b))
        mc = check_metrics(a.parent)
        if mc:
            nsteps, exact, renamed, absent = mc
            bits = [f"{len(exact)}/6 exact over {nsteps} steps"]
            if renamed:
                bits.append("renamed: " + ", ".join(renamed))
            if absent:
                bits.append("NOT LOGGED: " + ", ".join(absent))
            notes.append(f"G8 metric names - " + "; ".join(bits))

    width = max(len(n) for n, _, _ in results) if results else 0
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")
    for note in notes:
        print(f"  [note] {note}")
    passed = all(ok for _, ok, _ in results)
    print(f"  -> {'GATE 1 CHECKS PASS' if passed else 'GATE 1 CHECKS INCOMPLETE'}")
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--all", action="store_true", help="verify every member folder under FOLDER")
    a = ap.parse_args()

    if a.all:
        members = [p for p in sorted(a.folder.iterdir())
                   if p.is_dir() and not p.name.startswith((".", "_"))]
        results = {m.name: verify(m) for m in members}
        print("\n" + "=" * 68)
        for k, v in results.items():
            print(f"  {'PASS' if v else 'INCOMPLETE'}  {k}")
        return 0
    return 0 if verify(a.folder) else 1


if __name__ == "__main__":
    sys.exit(main())
