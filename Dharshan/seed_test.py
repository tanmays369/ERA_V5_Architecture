# Name: dharshan
# Axiom email: dharshanthaigu@gmail.com
# Pack: ind-ta-fw2
"""Gate tests 2 and 3, plus a NaN-stop check and a resume check. Every training run is a fresh process.

Run from the folder that holds configs/ and harness/:
    python harness/seed_test.py                 # prints PASS / FAIL, keeps nothing
    python harness/seed_test.py --evidence      # also copies the runs into evidence/seed_test_<time>/ (no checkpoints)
    python harness/seed_test.py --steps 30      # shorter (the loss-falls check needs at least 20 steps)

Checks:
    T2a  seed 1234 twice (two fresh processes)               losses.txt identical, and one row for every step
    T2b  control run, seed 1235                              differs from seed 1234 at EVERY step
    T2c  no NaN; loss falls; step-0 loss is close to ln(vocab size)
    T3a  window=0 and router_sharing=none set explicitly     equal to the default run, exactly
    T3b  attention.window=8, twice                           identical
    T3c  moe.router_sharing=all_layers, twice                identical
    T4   NaN injected at step 3                              run exits with code 2 and writes NAN_REPORT.json
    T5   stop at half-way, then --resume                     losses.txt equal to the uninterrupted run, exactly
Ten training runs in total. Exit code 0 only if every check passes.
"""
import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from config import load_config  # noqa: E402


def launch(config, out, sets, extra=()):
    cmd = [sys.executable, os.path.join(HERE, "train.py"), "--config", config, "--out", out, *extra]
    for s in sets:
        cmd += ["--set", s]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True), cmd


def rows(out):
    path = os.path.join(out, "losses.txt")
    return [l.split() for l in open(path)] if os.path.isfile(path) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_small.yaml")
    ap.add_argument("--steps", type=int, default=None, help="override tokens.train_steps")
    ap.add_argument("--evidence", action="store_true")
    a = ap.parse_args()
    cfg = os.path.join(ROOT, a.config)
    c = load_config(cfg)
    steps = a.steps or c.get("tokens.train_steps")
    ln_v = math.log(c.get("tokenizer.vocab_size"))
    tmp = tempfile.mkdtemp(prefix="seedtest_")
    extra_sets = [f"tokens.train_steps={a.steps}"] if a.steps else []

    def run(name, sets=(), extra=(), must_succeed=True):
        print(f"running {name} ...", flush=True)
        r, cmd = launch(cfg, os.path.join(tmp, name), list(sets) + extra_sets, extra)
        if must_succeed and r.returncode:
            print(r.stdout[-2000:], r.stderr[-2000:])
            raise SystemExit(f"run failed (exit {r.returncode}): {' '.join(cmd)}")
        return rows(os.path.join(tmp, name)) if must_succeed else r.returncode

    base1, base2 = run("base1"), run("base2")
    ctrl = run("control", ["general.seed=1235"])
    explicit = run("explicit_off", ["attention.window=0", "moe.router_sharing=none"])
    w1, w2 = run("win1", ["attention.window=8"]), run("win2", ["attention.window=8"])
    s1, s2 = run("sh1", ["moe.router_sharing=all_layers"]), run("sh2", ["moe.router_sharing=all_layers"])
    half = max(1, steps // 2)
    run("resume", extra=["--stop_after", str(half)])
    resumed = run("resume", extra=["--resume"])
    # NaN-free is judged on the normal runs, before the deliberate NaN run is made
    nan_free = not any(os.path.exists(os.path.join(tmp, n, "NAN_REPORT.json")) for n in os.listdir(tmp))
    code = run("nan_run", extra=["--nan_at", "3"], must_succeed=False)
    nan_stopped = code == 2 and os.path.isfile(os.path.join(tmp, "nan_run", "NAN_REPORT.json"))

    ntp = [float.fromhex(r[1]) for r in base1]
    total = [float.fromhex(r[3]) for r in base1]
    finite = nan_free and bool(total) and all(math.isfinite(v) for v in total)
    complete = len(base1) == steps and len(base2) == steps
    res = {
        f"T2a seed 1234 twice: identical losses, {steps} rows": complete and base1 == base2,
        "T2b control seed 1235 differs at every step": len(ctrl) == len(base1) and all(x[3] != y[3] for x, y in zip(ctrl, base1)),
        "T2c no NaN": finite,
        "T2c loss falls (last 10 steps below first 10)": finite and len(total) >= 20 and sum(total[-10:]) < sum(total[:10]),
        f"T2c step-0 loss_ntp within 0.5 of ln(vocab) = {ln_v:.2f}": bool(ntp) and abs(ntp[0] - ln_v) < 0.5,
        "T3a window=0 + router_sharing=none equals default exactly": complete and explicit == base1,
        "T3b attention.window=8 twice: identical": len(w1) == steps and w1 == w2,
        "T3c router_sharing=all_layers twice: identical": len(s1) == steps and s1 == s2,
        "T4 injected NaN stops the run (exit 2) and writes NAN_REPORT.json": nan_stopped,
        "T5 stop half-way + resume equals the uninterrupted run": complete and resumed == base1,
    }
    lines = [("PASS  " if v else "FAIL  ") + k for k, v in res.items()]
    if total:
        lines.append(f"loss first step {total[0]:.4f} -> last step {total[-1]:.4f}")
    print("\n".join(lines))
    if a.evidence:
        dest = os.path.join(ROOT, "evidence", "seed_test_" + time.strftime("%Y%m%d_%H%M%S"))
        shutil.copytree(tmp, dest, ignore=shutil.ignore_patterns("*.pt", "*.tmp"))
        with open(os.path.join(dest, "result.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print("evidence written to", dest, "(cite it as [run] evidence/<folder>/result.txt)")
    shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0 if all(res.values()) else 1)


if __name__ == "__main__":
    main()
