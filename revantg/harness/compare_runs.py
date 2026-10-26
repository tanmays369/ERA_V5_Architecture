# Name: Revant Gupta
# Pack: ind-hi-fw2
"""Compare two runs' losses.tsv: `A B --exact` (byte-identical) or `A B --rtol R`; `--gate A` checks one run.
Exit 0 = pass, 1 = fail.
"""
import argparse
import math
import os
import sys


def losses(run):
    with open(os.path.join(run, "losses.tsv")) as f:
        rows = [ln.rstrip("\n").split("\t") for ln in f if ln.strip()]
    return [(int(s), v) for s, v in rows]


def gate(run):
    rows = losses(run)
    vals = [float(v) for _, v in rows]
    ok = True
    if not vals or not all(math.isfinite(v) for v in vals):
        print(f"FAIL non-finite or empty loss in {run}")
        return False
    ref = math.log(131072)
    print(f"step0 CE {vals[0]!r} vs ln(131072) {ref:.4f}")
    if abs(vals[0] - ref) > 0.5:
        print("FAIL step-0 CE not within 0.5 of ln(131072)")
        ok = False
    k = max(1, len(vals) // 10)
    first, last = sum(vals[:k]) / k, sum(vals[-k:]) / k
    print(f"mean CE first {k}: {first:.6f}  last {k}: {last:.6f}  steps {len(vals)}")
    if not last < first:
        print("FAIL loss did not fall")
        ok = False
    return ok


def compare(a, b, exact, rtol):
    ra, rb = losses(a), losses(b)
    if exact:
        for (sa, va), (sb, vb) in zip(ra, rb):
            if (sa, va) != (sb, vb):
                print(f"FAIL first difference at step {sa}: {va} vs {vb}")
                return False
        if len(ra) != len(rb) or not ra:
            print(f"FAIL length {len(ra)} vs {len(rb)}")
            return False
        print(f"exact: {len(ra)} steps byte-identical")
        return True
    db = dict(rb)
    common = [(s, float(v), float(db[s])) for s, v in ra if s in db]
    if not common:
        print("FAIL no common steps")
        return False
    mabs = max(abs(x - y) for _, x, y in common)
    s, x, y = max(common, key=lambda t: abs(t[1] - t[2]) / max(abs(t[2]), 1e-12))
    mrel = abs(x - y) / max(abs(y), 1e-12)
    print(f"{len(common)} common steps; max abs diff {mabs:.3e}; max rel diff {mrel:.3e} at step {s} ({x!r} vs {y!r})")
    if not mrel <= rtol:  # also fails on NaN
        print(f"FAIL max rel diff > rtol {rtol}")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--rtol", type=float, default=None)
    ap.add_argument("--gate", action="store_true")
    a = ap.parse_args()
    if a.gate:
        ok = all([gate(r) for r in a.runs])
    else:
        if len(a.runs) != 2 or a.exact == (a.rtol is not None):
            ap.error("give two runs and exactly one of --exact / --rtol R")
        ok = compare(*a.runs, a.exact, a.rtol)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
