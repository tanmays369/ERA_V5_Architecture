Name: Revant Gupta · Pack: ind-hi-fw2

# Harness v0 results: Kaggle 2× T4, 26 Oct 2026

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

Comparison across layouts with the same seed: dense dp2, tp2, pp2 and zero1 differ from R0 by at most 2e-7 relative. That is the fp32 reduction order. MoE dp2 and pp2 differ from R6a by at most 9.3e-5 relative, because a near-tied router score can flip one expert choice.

| Run | step ms (median) | tokens/s | peak alloc GB | final held-out CE | BPB | dead experts | top expert share |
|---|---|---|---|---|---|---|---|
| R0 dense 1 GPU | 478 | 8,573 | 0.68 | 7.444 | 1.901 | - | - |
| R1a dense dp2 | 300 | 13,659 | 0.84 | 7.444 | 1.901 | - | - |
| R2a dense tp2 | 739 | 5,547 | 0.53 | 7.444 | 1.901 | - | - |
| R3a dense pp2 | 442 | 9,276 | 0.13 | 7.444 | 1.901 | - | - |
| R4 dense dp2 ZeRO-1 | 297 | 13,797 | 0.81 | 7.444 | 1.901 | - | - |
| R6a MoE 1 GPU | 609 | 6,724 | 0.71 | 7.455 | 1.904 | 0 | 0.31 |
| R7a MoE dp2 | 401 | 10,228 | 0.88 | 7.455 | 1.904 | 0 | 0.31 |
| R8 MoE pp2 | 504 | 8,129 | 0.15 | 7.455 | 1.904 | 0 | 0.30 |

Metric names for G8 (`metrics_schema.json`, one JSONL row per step): `loss, loss_ntp, loss_mtp, lr, tokens_per_second, gpu_alloc_gb`. The extras are `z_loss, grad_norm, step_time_ms, spike_score, ema_z, update_to_weight_ratio, max_attn_logit, held_out_loss, bpb, dead_experts, top_expert_share, moe_aux`. `loss_mtp` is null because v0 has no MTP head.
