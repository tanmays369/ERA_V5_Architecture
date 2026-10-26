Name: Revant Gupta · Pack: ind-hi-fw2

# Team 5 harness v0

A tiny MoE decoder, built from scratch. It uses the Kronecker embedding (BrahmicTokenizer-131K), GQA attention with QK-norm and RoPE, and SwiGLU experts (a dense arm via `moe.enabled: false`). Optimizers are Muon for the 2D weights and AdamW for the rest. The schedule is WSD. It runs with DP, TP, PP or ZeRO-1 on 2 GPUs, and on CPU (gloo) for smoke tests. Two runs with seed 1234 give byte-identical `losses.tsv`.

Run everything from the `teams/05-architecture/` folder (the `paths.*` settings resolve against the current directory):

```bash
pip install torch numpy pyyaml tokenizers datasets kronecker-embeddings
python harness/check_config.py configs/config_v0.yaml                 # 0 errors; sizes are open (TSAI, 31 Oct)
python harness/check_config.py configs/config_small.yaml --world-size 1
python harness/prepare_data.py --config configs/config_small.yaml     # about 2 min: streams 12 HF sources, about 9M tokens
torchrun --nproc_per_node=1 harness/train.py --config configs/config_small.yaml --out runs/a
torchrun --nproc_per_node=1 harness/train.py --config configs/config_small.yaml --out runs/b
python harness/compare_runs.py runs/a runs/b --exact                  # same seed, same loss at every step
python harness/compare_runs.py --gate runs/a                          # step-0 loss about ln(131072), loss falls, no NaN
```

- **Configs**
  - `configs/config_v0.yaml`: every setting has a source comment.
  - `configs/config_small.yaml`: **the G6 file**, a self-contained tiny MoE (4 experts, top-2) beside `config_v0.yaml`; same values as `configs/small/t4x1_moe.yaml`.
  - `t4_base.yaml`: the dense arm.
  - `t4x*`, `cpu_smoke*`: the layouts.
- **Flags**
  - `--set a.b=v`: overrides a setting.
  - `--device cpu`: runs on CPU (gloo).
- **Outputs in `--out`**
  - `losses.tsv`: the gate diff file.
  - `metrics.jsonl`, with `metrics_schema.json` (the metric names for G8). `gpu_alloc_gb` is the peak allocated during the step (`max_memory_allocated`, reset each step); `ema_z` is null before step 10; `bpb` divides by the UTF-8 bytes of the evaluated val windows, and `held_out_loss_<source>` gives one value per dataset. MoE runs add `dead_experts`, `top_expert_share`, `moe_aux`, and every `per_layer_every` steps a `moe_layers` list (load, MaxVio, router entropy, top-k intervention rate, top-k score mass per layer).
  - `run.json`, `train.log`, `checkpoints/step_N/` (`rank{r}.pt` and `manifest.json`, written to `step_N.tmp/` and renamed, so a crash never leaves a half checkpoint).
- **On NaN/Inf:** each worker exits 3 (`torchrun` reports 1), writes `NAN_REPORT.json`, saves no checkpoint, and prints the resume command. `--inject-nan STEP` tests this path.
- **Resume (`--resume CKPT`, same `--out`):** refuses to start if the checkpoint's world size, pp/dp/tp, ZeRO stage or config hash differ from this run (it prints which), or if the frozen weights changed. It restores the weights, optimizer state and the spike/EMA state, so `losses.tsv` is byte-identical to an uninterrupted run and `spike_score`/`ema_z` continue without a gap. Resuming into a new `--out` writes `losses.tsv` from the resume step; `compare_runs A B --exact` then compares the steps both runs have (B's steps must be a contiguous tail of A's) and prints `exact: N aligned steps (from S)`.
