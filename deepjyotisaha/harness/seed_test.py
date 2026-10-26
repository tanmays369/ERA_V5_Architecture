"""The week-1 gate test: two runs with seed 1234 must give the same loss at every step.

Usage:
    python harness/seed_test.py [--config configs/tiny/tiny_moe.yaml] [--steps 40]

It trains the same config twice, each time in a fresh Python process (so nothing is shared between the
runs), and compares the two losses.txt files line by line. losses.txt holds the step, the total loss, the
next-token loss, the MTP loss and the gradient norm as exact hexadecimal floats, so "the same" means
bit-for-bit identical, not "close". It then checks the two other gate conditions: the loss goes down,
and there is no NaN. The exit code is 0 only when all three hold.
"""
import argparse
import difflib
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(HERE.parent / "configs" / "tiny" / "tiny_moe.yaml"))
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or HERE.parent / "work" / "seed_test" / Path(a.config).stem)
    runs = []
    for name in ("run_a", "run_b"):
        d = out / name
        print(f"--- {name}")
        cmd = [sys.executable, "-X", "utf8", str(HERE / "train.py"), "--config", a.config, "--steps", str(a.steps),
               "--seed", str(a.seed), "--out", str(d)]
        subprocess.run(cmd, check=True)
        runs.append(d)
    la = (runs[0] / "losses.txt").read_text().splitlines()
    lb = (runs[1] / "losses.txt").read_text().splitlines()
    diff = list(difflib.unified_diff(la, lb, "run_a/losses.txt", "run_b/losses.txt", lineterm=""))
    metrics = [json.loads(l) for l in (runs[0] / "metrics.jsonl").read_text().splitlines()]
    ntp = [m["loss_ntp"] for m in metrics]
    k = max(1, len(ntp) // 5)
    first, last = sum(ntp[:k]) / k, sum(ntp[-k:]) / k
    finite = all(math.isfinite(m["loss"]) for m in metrics)
    same = not diff and len(la) == a.steps
    print("=" * 70)
    print(f"Identical losses at every step: {'YES (empty diff, ' + str(len(la)) + ' steps)' if same else 'NO'}")
    if diff:
        print("\n".join(diff[:12]))
    print(f"Loss goes down: {'YES' if last < first else 'NO'} (next-token loss, mean of first {k} steps "
          f"{first:.4f} -> last {k} steps {last:.4f})")
    print(f"No NaN or infinite loss: {'YES' if finite else 'NO'}")
    result = {"config": a.config, "steps": a.steps, "seed": a.seed, "identical": same, "diff_lines": len(diff),
              "loss_first": first, "loss_last": last, "finite": finite}
    (out / "seed_test_result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    sys.exit(0 if same and last < first and finite else 1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
