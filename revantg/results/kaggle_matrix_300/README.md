Name: Revant Gupta · Pack: ind-hi-fw2

# Harness v0 results: Kaggle 2× T4, 26 Oct 2026 (notebook version 6, final code)

Environment: Python 3.13.15, torch 2.11.0+cu128, NCCL 2.28.9, fp32 with deterministic algorithms on.
Each run is 300 steps of the `t4_base` sizes (4 layers, d_model 256, seq 256, global batch 16), seed 1234. The model has 38.8M parameters, most of them in the 131K-vocabulary head.
Data: `prepare_data.py` streamed 500 train + 20 val documents from each of 12 HF sources, about 9.0M train tokens, in 86 s. See `manifest.json` and `prepare_data.log`.
Files in this folder: `status.txt` (exit codes), `compare.txt` (every check), and one `losses.tsv` per run (the diff files). `R6a/metrics.jsonl` is a full MoE metrics log.

Command for each run: `torchrun --nproc_per_node=N harness/train.py --device cuda --config configs/small/<cfg>.yaml --out <run>`.

| Run | Config | Check | Result |
|---|---|---|---|
| R0 | `t4x1_ref` (dense, 1 GPU) | gate | step-0 CE 11.8437 (ln V = 11.7835); mean CE 11.15 over the first 30 steps → 7.01 over the last 30; no NaN |
| R1a/b | `t4x2_dp2` | exact | 300 steps byte-identical |
| R2a/b | `t4x2_tp2` | exact | 300 steps byte-identical |
| R3a/b | `t4x2_pp2` (1F1B, 4 micro-batches) | exact | 300 steps byte-identical |
| R4 | `t4x2_dp2_zero1` | vs R1a | max rel diff 0 |
| R5 | `t4x2_dp2 --inject-nan 150` | NaN path | stops at step 150 (worker exit 3); writes `NAN_REPORT.json`; no checkpoint after the NaN; prints the resume command |
| R5 resumed | `--resume checkpoints/step_000100` | exact vs R1a | 300 steps byte-identical |
| R6a/b | `t4x1_moe` (4 experts, top-2) | gate + exact | step-0 CE 11.8213; 11.14 → 7.02; byte-identical |
| R7a/b | `t4x2_dp2_moe` | gate + exact | byte-identical |
| R8 | `t4x2_pp2_moe` | vs R6a | max rel diff 8.4e-5 |

Comparison across layouts with the same seed: dense dp2, tp2, pp2 and zero1 differ from R0 by at most 2e-7 relative. That is the fp32 reduction order. MoE dp2 and pp2 differ from R6a by at most 9.3e-5 relative [INFERENCE: probably a near-tied router score flipping one expert choice; not proven].
These are re-runs on the final code, after the review fixes; the review fixes changed only logging and checkpointing.
Every `losses.tsv` is byte-identical to the earlier run of the same matrix (notebook version 3), so the result also holds across Kaggle sessions.

| Run | step ms (median) | tokens/s | peak alloc GB (max over steps) | final held-out CE | BPB (equal weight per source) |
|---|---|---|---|---|---|
| R0 dense 1 GPU | 477 | 8,592 | 3.67 | 7.444 | 1.734 |
| R1a dense dp2 | 295 | 13,879 | 3.82 | 7.444 | 1.734 |
| R2a dense tp2 | 729 | 5,621 | 3.39 | 7.444 | 1.734 |
| R3a dense pp2 | 418 | 9,808 | 0.34 | 7.444 | 1.734 |
| R4 dense dp2 ZeRO-1 | 286 | 14,323 | 3.80 | 7.444 | 1.734 |
| R6a MoE 1 GPU | 621 | 6,592 | 3.74 | 7.455 | 1.737 |
| R7a MoE dp2 | 413 | 9,918 | 3.90 | 7.455 | 1.737 |
| R8 MoE pp2 | 497 | 8,246 | 0.38 | 7.455 | 1.737 |

Peak memory is about 3.7 GB of the T4's 15 GB, mostly the 131K-vocab logits and their gradient. Under PP the head sits on rank 1, so `gpu_alloc_gb` is logged from rank 0.

Metric names for G8 (`metrics_schema.json`, one JSONL row per step): `loss, loss_ntp, loss_mtp, lr, tokens_per_second, gpu_alloc_gb`.
- **Extras:** `z_loss, grad_norm, step_time_ms, spike_score, ema_z, update_to_weight_ratio, max_attn_logit, held_out_loss, held_out_loss_<source>, bpb, dead_experts, top_expert_share, moe_aux`.
- **Per layer, every 10 steps:** `moe_layers[]` with load, MaxVio, router entropy, intervention rate and top-k mass.
- `loss_mtp` is null because v0 has no MTP head.
