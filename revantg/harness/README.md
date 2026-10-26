Name: Revant Gupta · Pack: ind-hi-fw2

# Team 5 harness v0

A tiny MoE decoder, built from scratch. It uses the Kronecker embedding (BrahmicTokenizer-131K), GQA attention with QK-norm and RoPE, and SwiGLU experts (a dense arm via `moe.enabled: false`). Optimizers are Muon for the 2D weights and AdamW for the rest. The schedule is WSD. It runs with DP, TP, PP or ZeRO-1 on 2 GPUs, and on CPU (gloo) for smoke tests. Two runs with seed 1234 give byte-identical `losses.tsv`.

Run everything from the folder that holds `configs/` and `harness/`: `teams/05-architecture/` in era-v5-lab, or `revantg/` in this repo. The `paths.*` settings resolve against the current directory.

```bash
pip install torch numpy pyyaml tokenizers datasets kronecker-embeddings
python harness/check_config.py configs/config_v0.yaml                 # 0 errors; sizes are open (TSAI, 31 Oct)
python harness/check_config.py configs/small/t4_moe.yaml --world-size 1
python harness/prepare_data.py --config configs/small/t4_moe.yaml     # about 2 min: streams 12 HF sources, about 9M tokens
torchrun --nproc_per_node=1 harness/train.py --config configs/small/t4x1_moe.yaml --out runs/a
torchrun --nproc_per_node=1 harness/train.py --config configs/small/t4x1_moe.yaml --out runs/b
python harness/compare_runs.py runs/a runs/b --exact                  # same seed, same loss at every step
python harness/compare_runs.py --gate runs/a                          # step-0 loss about ln(131072), loss falls, no NaN
```

- **Configs**
  - `configs/config_v0.yaml`: every setting has a source comment.
  - `configs/small/t4_moe.yaml`: the small config for G6 (4 experts, top-2).
  - `t4_base.yaml`: the dense arm.
  - `t4x*`, `cpu_smoke*`: the layouts.
- **Flags**
  - `--set a.b=v`: overrides a setting.
  - `--device cpu`: runs on CPU (gloo).
- **Outputs in `--out`**
  - `losses.tsv`: the gate diff file.
  - `metrics.jsonl`, with `metrics_schema.json` (the metric names for G8).
  - `run.json`, `train.log`, `checkpoints/`.
- **On NaN/Inf:** each worker exits 3 (`torchrun` reports 1), writes `NAN_REPORT.json`, saves no checkpoint, and prints the resume command. `--inject-nan STEP` tests this path.
