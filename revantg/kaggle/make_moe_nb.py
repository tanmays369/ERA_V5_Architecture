"""Kaggle T4x2 MoE iteration notebook: dense baseline + MoE arms, 1000 steps each, dp2, one session.

Usage: python make_moe_nb.py [BRANCH] > nb.json
"""
import json, sys

branch = sys.argv[1] if len(sys.argv) > 1 else "t5-harness-v0"
T = "teams/05-architecture"
H = f"{T}/harness"
C = f"{T}/configs/small"
COMMON = "--set tokens.train_steps=1000 --set checkpoint.every_steps=1000"
# name, config, overrides. Active FFN width per token = top_k*d_ff (+ shared*d_ff) = 768 in every MoE arm (dense d_ff 768).
ARMS = [
    ("D0_dense", "t4x2_dp2", ""),
    ("M1_e4k2", "t4x2_dp2_moe", ""),
    ("M2_e8k2", "t4x2_dp2_moe", "--set moe.num_experts=8"),
    ("M3_e16k4_fine", "t4x2_dp2_moe", "--set moe.num_experts=16 --set moe.top_k=4 --set model.d_ff=192"),
    ("M4_e8k1_shared1", "t4x2_dp2_moe", "--set moe.num_experts=8 --set moe.top_k=1 --set moe.shared_experts=1"),
    ("M5_e8k2_nobias", "t4x2_dp2_moe", "--set moe.num_experts=8 --set moe.bias_update_speed=0.0 --set moe.seq_aux_loss_coef=0.0"),
]

lines = ["set -u", "cd /kaggle/working/w", "export HF_HOME=/kaggle/working/hf PYTHONHASHSEED=0",
         "R=/kaggle/working/moe; mkdir -p $R"]
for name, cfg, ov in ARMS:
    lines.append(f"python {H}/check_config.py {C}/{cfg}.yaml {COMMON} {ov} --world-size 2 > $R/check_{name}.txt 2>&1; "
                 f"echo \"check {name} exit=$?\" | tee -a $R/status.txt")
    lines.append(f"torchrun --nproc_per_node=2 --master_port=$((29500 + RANDOM % 1000)) {H}/train.py --device cuda "
                 f"--config {C}/{cfg}.yaml {COMMON} {ov} --out $R/{name} > $R/{name}.log 2>&1; "
                 f"echo \"=== {name} exit=$?\" | tee -a $R/status.txt")
    lines.append(f"python {H}/compare_runs.py --gate $R/{name} >> $R/gates.txt 2>&1; echo \"gate {name} exit=$?\" >> $R/gates.txt")
lines += ["find $R -name checkpoints -type d -prune -exec rm -rf {} +",
          "tar czf /kaggle/working/moe.tgz -C /kaggle/working moe"]

summary = r'''import json, statistics, re, os
R = "/kaggle/working/moe"
rows = []
for name in %s:
    p = f"{R}/{name}/metrics.jsonl"
    if not os.path.exists(p):
        rows.append((name, "missing")); continue
    ms = [json.loads(l) for l in open(p)]
    ev = [m for m in ms if m.get("held_out_loss") is not None]
    params = re.search(r"params=(\d+)", open(f"{R}/{name}.log").read())
    tail = ms[-100:]
    rows.append(dict(arm=name, params=int(params.group(1)) if params else None,
        ce_last100=round(statistics.mean(m["loss_ntp"] for m in tail), 4),
        held_out=round(ev[-1]["held_out_loss"], 4), bpb=round(ev[-1]["bpb"], 4),
        held_out_curve=[(m["step"], round(m["held_out_loss"], 3)) for m in ev[::4]],
        step_ms=round(statistics.median(m["step_time_ms"] for m in ms[1:]), 1),
        mem_gb=round(max(m["gpu_alloc_gb"] for m in ms), 2),
        dead_max=max((m.get("dead_experts") or 0) for m in ms),
        top_share_last=tail[-1].get("top_expert_share"),
        top_share_max=max((m.get("top_expert_share") or 0) for m in ms),
        nan=any(m["loss"] != m["loss"] for m in ms)))
json.dump(rows, open(f"{R}/summary.json", "w"), indent=1)
for r in rows: print(r)
print(open(f"{R}/status.txt").read()); print(open(f"{R}/gates.txt").read())
''' % json.dumps([a[0] for a in ARMS])

cells = [
    f"!rm -rf /kaggle/working/w && git clone -q --depth 1 -b {branch} https://github.com/revantg/era-v5-lab.git /kaggle/working/w && git -C /kaggle/working/w log --oneline -1\n"
    "!pip install -q kronecker-embeddings 2>&1 | tail -1",
    f"%%bash\ncd /kaggle/working/w\n(time HF_HOME=/kaggle/working/hf python {H}/prepare_data.py --config {C}/t4_base.yaml) 2>&1 | tail -5",
    "%%bash\n" + "\n".join(lines),
    summary,
]
nb = {"metadata": {"kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4,
      "cells": [{"cell_type": "code", "source": c, "metadata": {}, "outputs": [], "execution_count": None} for c in cells]}
print(json.dumps(nb))
