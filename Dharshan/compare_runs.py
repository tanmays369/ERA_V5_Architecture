# Name: dharshan
# Axiom email: dharshanthaigu@gmail.com
# Pack: ind-ta-fw2
"""Compare the losses.txt files of finished runs. Exit code 0 = pass, 1 = fail.

    python harness/compare_runs.py RUN_A RUN_B --exact      same step numbers and the same bytes on every line
    python harness/compare_runs.py RUN_A RUN_B --rtol 1e-6  largest relative difference of the total loss
    python harness/compare_runs.py --gate RUN [RUN ...] --vocab 131072
                                                           finite, step-0 loss_ntp near ln(vocab), loss falls

A run folder is the --out folder of train.py. losses.txt lines are: step loss_ntp loss_mtp loss (hex floats).
An empty file fails every mode, so two missing runs can never "match".
"""
import argparse
import math
import os
import sys


def read(run):
    path = os.path.join(run, "losses.txt")
    if not os.path.isfile(path):
        return {}
    out = {}
    for line in open(path):
        p = line.split()
        if len(p) == 4:
            out[int(p[0])] = p[1:]
    return out


def exact(a, b):
    ra, rb = read(a), read(b)
    if not ra or not rb:
        return "FAIL: a losses.txt is missing or empty"
    if sorted(ra) != sorted(rb):
        return f"FAIL: steps differ ({min(ra)}..{max(ra)} vs {min(rb)}..{max(rb)})"
    for s in sorted(ra):
        if ra[s] != rb[s]:
            return f"FAIL: first difference at step {s}: {ra[s]} vs {rb[s]}"
    return f"PASS: {len(ra)} steps identical"


def close(a, b, rtol):
    ra, rb = read(a), read(b)
    common = sorted(set(ra) & set(rb))
    if not common:
        return "FAIL: no common steps"
    worst = max(abs(float.fromhex(ra[s][2]) - float.fromhex(rb[s][2])) / max(abs(float.fromhex(rb[s][2])), 1e-12) for s in common)
    return (f"PASS: " if worst <= rtol else "FAIL: ") + f"{len(common)} common steps, max relative difference {worst:.3e} (limit {rtol})"


def gate(run, vocab):
    r = read(run)
    if not r:
        return f"FAIL {run}: missing or empty losses.txt"
    ntp = [float.fromhex(r[s][0]) for s in sorted(r)]
    if not all(math.isfinite(v) for v in ntp):
        return f"FAIL {run}: non-finite loss"
    if abs(ntp[0] - math.log(vocab)) > 0.5:
        return f"FAIL {run}: step-0 loss {ntp[0]:.3f} is not within 0.5 of ln({vocab}) = {math.log(vocab):.3f}"
    k = max(1, len(ntp) // 10)
    first, last = sum(ntp[:k]) / k, sum(ntp[-k:]) / k
    return (f"PASS {run}: " if last < first else f"FAIL {run}: loss did not fall, ") + f"mean of first {k} = {first:.4f}, last {k} = {last:.4f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--rtol", type=float)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--vocab", type=int, default=131072)
    a = ap.parse_args()
    if a.gate:
        msgs = [gate(r, a.vocab) for r in a.runs]
    elif len(a.runs) == 2 and a.exact != (a.rtol is not None):
        msgs = [exact(*a.runs) if a.exact else close(*a.runs, a.rtol)]
    else:
        ap.error("give two runs and exactly one of --exact / --rtol, or use --gate")
    print("\n".join(msgs))
    sys.exit(0 if all(m.startswith("PASS") for m in msgs) else 1)


if __name__ == "__main__":
    main()
