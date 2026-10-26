# Revant Gupta: Team 5 tiny MoE harness v0

**Open [`index.html`](index.html) for the visual summary** (download it and open it locally, or use any HTML preview). It has charts of every result below.

- **Author:** Revant Gupta (GitHub `revantg`). Diagnosis pack `ind-hi-fw2`. ERA V5 capstone, core Team 5 (Architecture).
- **Built:** 25–26 Oct 2026.
- **Status:** scratch version for the G5 comparison agreed on the 26 Oct call. The team cherry-picks from each member's folder into the skeleton for the main ERA-V5 repo.
- **Upstream PR with the same code:** [theschoolofai/era-v5-lab#55](https://github.com/theschoolofai/era-v5-lab/pull/55) (pr-guard passes).

## What the 26 Oct G5 call agreed (Udit, Deepjyoti, Shivanshu, Dharshan, Revant)

- **Due Tue 27 Oct:** MoE `config_v0.yaml`, `check_config.py`, the tiny-MoE harness `train.py`, and `plan_v0.md`.
  - Tomorrow's version is a skeleton: the keys and structure matter now.
  - The values come from TSAI (Rohan) on Fri/Sat.
- **This repo:** everyone pushes their own code under their own folder tonight. Shivanshu tracks progress and posts the key points in Telegram. The team cherry-picks the best parts and submits one skeleton by Tue evening.
- **Gaps in every config:** the data-mixture weights and the distributed sizes (dp/tp/pp). The SmolLM3 configs and nanotron are the references.
- **Seed 1234:** two runs must give the same loss at every step.

**How this folder covers those gaps:**
- **Data mixture:** SmolLM3-style `data_stages` (stable, stage2 and anneal at 0, 0.73 and 0.82 of training), weights over 12 sources: English, code, math and 8 Indic languages.
- **Parallelism:** nanotron `ParallelismArgs` names. DP, TP, PP (1F1B) and ZeRO-1 are each exercised on 2× T4.

## Folder

| Path | What it is |
|---|---|
| `configs/config_v0.yaml` | Every setting has a value and an inline `# source` comment: KE code / arXiv 2605.29459, a lesson section, an Admin ruling, or SmolLM3. Sizes, `moe.num_experts`/`top_k` and the KE2 shape are null and listed in `open_keys`. |
| `configs/small/` | Small configs that `extends:` a parent. `t4_moe.yaml` is the G6 file (4 experts, top-2). `t4_base.yaml` is the dense arm. `t4x*` are the 2-GPU layouts. `cpu_smoke*` runs on a laptop. |
| `harness/check_config.py` | Prints every setting with no value or no source, plus stage-weight, size and layout checks. Exits 1 on errors. `config_v0`: 0 errors, 15 open. Small configs: 0 errors, 1 open (KE2). |
| `harness/train.py` + `model.py`, `optim.py`, `parallel.py`, `data.py`, `tok.py`, `config.py` | The harness: deterministic training on one process or under `torchrun`, with DP/TP/PP/ZeRO-1, checkpoints and resume, and a NaN stop. |
| `harness/prepare_data.py` | Streams 500 train + 20 val docs from each of 12 HF sources (no token needed), tokenizes, writes uint32 `.bin` files + `manifest.json`. About 90 s on Kaggle. |
| `harness/compare_runs.py` | The gates: `--exact` (byte-identical `losses.tsv`), `--rtol R`, `--gate` (step-0 CE ≈ ln V, loss falls, no NaN). |
| `results/kaggle_matrix_300/` | Correctness of every layout on 2× T4 (300 steps): `compare.txt`, `status.txt`, one `losses.tsv` per run. |
| `results/moe_sweep_1000/` | The MoE iteration: 5 arms × 1,000 steps, DP=2, same notebook runtime. Full `metrics.jsonl` per arm plus `summary.json`. |
| `kaggle/` | Generators for the two Kaggle notebooks (`make_nb.py`: layout matrix; `make_moe_nb.py`: MoE sweep) and for the summary page (`build_index.py` + `index.template.html`). |
| `index.html` | The visual summary: interactive charts (toggle arms, hover for values), a sortable results table. Rebuild with `python kaggle/build_index.py index.html` from this folder. |

## Model

| Part | Choice |
|---|---|
| Tokens | BrahmicTokenizer-131K, pinned commit and sha256; EOS id 36 between documents |
| Embedding | Kronecker codec (char 256 × pos 32, dynamic mode) → trainable projection; no weight decay |
| Attention | GQA, QK-norm, RoPE θ = 50k, no bias, SDPA math backend (deterministic) |
| FFN | SwiGLU; MoE: sigmoid router in fp32, top-k over score + aux-free bias (DeepSeek-V3), sequence aux loss 1e-4, dropless, optional shared expert |
| Head | Untied; CE + z-loss 1e-4, in fp32 |
| Optimizer | Muon on 2D block weights and on each expert slice (Newton–Schulz in fp32 for T4); AdamW on the rest; WSD schedule (2% warmup, 10% decay); grad clip 1.0 |
| Metrics (G8) | `loss, loss_ntp, loss_mtp, lr, tokens_per_second, gpu_alloc_gb`, plus `z_loss, grad_norm, held_out_loss, bpb, update_to_weight_ratio, max_attn_logit, spike_score, ema_z, dead_experts, top_expert_share, moe_aux` |

## Results 1: layout correctness, Kaggle 2× T4, 300 steps

Environment: torch 2.11, fp32, seed 1234. Model: 4 layers, d_model 256, seq 256, global batch 16, 38.8M parameters.

| Check | Result |
|---|---|
| Same seed run twice: dense DP=2, TP=2, PP=2; MoE 1 GPU, MoE DP=2 | all **300 steps byte-identical** |
| NaN injected at step 150 | stops (worker exit 3), writes `NAN_REPORT.json`, saves no checkpoint, prints the resume command |
| Resume from step 100 | byte-identical to the uninterrupted run |
| Dense DP/TP/PP/ZeRO-1 vs 1 GPU | max relative gap 2e-7 (fp32 reduction order) |
| MoE DP/PP vs MoE 1 GPU | max relative gap 9e-5 (near-tied routing) |
| Gate | step-0 CE 11.84 (dense) and 11.82 (MoE) vs ln 131072 = 11.78; loss falls; no NaN |

| Layout | step ms | tokens/s | peak GB | held-out CE @300 |
|---|---|---|---|---|
| Dense 1 GPU | 478 | 8,573 | 0.68 | 7.4435 |
| Dense DP=2 | 300 | 13,659 | 0.84 | 7.4435 |
| Dense TP=2 | 738 | 5,547 | 0.53 | 7.4435 |
| Dense PP=2 | 442 | 9,276 | 0.13 | 7.4435 |
| Dense DP=2 + ZeRO-1 | 297 | 13,797 | 0.81 | 7.4435 |
| MoE 1 GPU | 609 | 6,724 | 0.71 | 7.4552 |
| MoE DP=2 | 400 | 10,228 | 0.88 | 7.4552 |
| MoE PP=2 | 504 | 8,129 | 0.15 | 7.4553 |

## Results 2: MoE iteration, 1,000 steps, DP=2, same notebook runtime

Every MoE arm keeps the active FFN width per token equal to the dense `d_ff` of 768 (top_k × d_ff + shared × d_ff). The comparison is therefore about routing and capacity at equal compute per token. Everything else is identical across arms.

| Arm | Params | Train CE (last 100) | Held-out CE @1000 | Δ vs dense | BPB | tokens/s | Dead experts (max) | Top expert share (max after step 50) |
|---|---|---|---|---|---|---|---|---|
| Dense (d_ff 768) | 38.8M | 5.612 | **6.3144** | +0.0000 | 1.6124 | 14,542 | - | - |
| MoE 4 experts, top-2 | 41.2M | 5.597 | **6.3009** | -0.0135 | 1.6089 | 10,555 | 0 | 0.49 |
| MoE 8 experts, top-2 | 45.9M | 5.573 | **6.2849** | -0.0295 | 1.6048 | 8,115 | 0 | 0.44 |
| MoE 16 fine experts, top-4 | 45.9M | 5.596 | **6.2991** | -0.0152 | 1.6085 | 5,428 | 0 | 0.21 |
| MoE 8 experts, top-1 + 1 shared | 47.1M | 5.572 | **6.2814** | -0.0330 | 1.6039 | 7,559 | 0 | 0.27 |

**Findings** (one seed, about 9M tokens; this is direction, not proof):
- **Every MoE arm beats dense** on held-out CE. The gap is consistent from about step 250 onward.
  - Best: 8 experts, top-1, plus 1 shared expert, at −0.033 nats.
  - Next: 8 experts, top-2, at −0.030.
- **Fine-grained experts** (16 experts, top-4) balance best, with the busiest expert at ≤ 0.21 share. They are 2.7× slower than dense on T4, because the reference expert loop is not fused.
- **No dead experts** in any arm. Aux-free bias balancing works at this scale.
- **The 6th arm (no balancing: bias speed 0, aux 0),** which tests for router collapse, crashed in a metrics logger when aux = 0. It is fixed in this code; its re-run is pending.

## Bugs found on real hardware (fixed here)

- **PP crashed on torch 2.11.** `PipelineStage` checks input strides, and `x[:, :-1]` is non-contiguous. Fixed with `.contiguous()`.
- **DDP over gloo** could not broadcast an int16 buffer. It is now int64.
- **`compare_runs --exact`** passed two empty loss files. It now fails.
- **`moe_metrics`** crashed when `seq_aux_loss_coef = 0`. Fixed.

## Run it

```bash
pip install torch numpy pyyaml tokenizers datasets kronecker-embeddings
python harness/check_config.py configs/config_v0.yaml
python harness/prepare_data.py --config configs/small/t4_moe.yaml
torchrun --nproc_per_node=1 harness/train.py --config configs/small/t4x1_moe.yaml --out runs/a
torchrun --nproc_per_node=1 harness/train.py --config configs/small/t4x1_moe.yaml --out runs/b
python harness/compare_runs.py runs/a runs/b --exact
python harness/compare_runs.py --gate runs/a
# laptop CPU: --device cpu with configs/small/cpu_smoke_moe.yaml
```

Run these from this folder: the `paths.*` settings resolve against the current directory.

## Candidates for the team skeleton

- **Config system:** `extends` + `--set` overrides, plus the source-comment checker.
- **Data order:** independent of the layout, so one config runs on any dp/tp/pp.
- **Gates:** `compare_runs.py` (exact / rtol / gate).
- **MoE:** aux-free bias balancing with Muon on each expert slice.

## Open

- **KE2 output shape:** from G4 (Tue 27 Oct).
- **G8 logger:** to integrate on Thu 29 Oct.
- **plan_v0 arms:** expert granularity, shared expert, Muon vs AdamW, the LR×WD law, and intra-document masking (SmolLM3).
- **Code data:** `the-stack-smol` needs an HF token, so codeparrot (Python only) is used for now.
