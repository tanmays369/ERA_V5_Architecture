Name: Revant Gupta
Pack: ind-hi-fw2

# Team 5 experiment plan v0 (for TSAI approval, Sat 31 Oct)

Legend: `[INFERENCE]` = estimate or reasoning, not measured. "S11 §14" = lesson session and section; "SmolLM3 f:N" = `stage1_8T.yaml:N` unless another file is named; "review §N" = the week-1 harness review.

## 1. Purpose and what this plan decides

Week 2 (1-14 Nov; break 5-11 Nov, queued runs only) runs the arms below. Their results fill `config_v1.yaml` and `decisions.md` (one result per setting) for the freeze on **Sat 14 Nov**.

| config v1 setting | decided by arm(s) | plan-sheet id |
|---|---|---|
| `moe.num_experts`, `moe.top_k`, `model.d_ff` (granularity at matched active params) | MO-1, HC-1 | EXP-3 |
| `moe.shared_experts`, `moe.first_dense_layers` | MO-2, MO-3 | EXP-3 |
| `moe.bias_update_speed`, `moe.seq_aux_loss_coef` (balancing) | MO-4, MO-5 | EXP-3 |
| `optimizer.matrices_2d` (Muon vs AdamW) | OP-1, PR-2, HC-2 | EXP-2 |
| `optimizer.lr`, `optimizer.weight_decay`, `adam_betas`, `schedule.warmup_frac` | OP-2 .. OP-5 | EXP-1 |
| `model.init_scaling`, `model.nope_every`, `model.qk_norm`, `head.z_loss_coef`, `data.doc_mask` | AR-1 .. AR-5 | attention arms |
| `data_stages.*.start_frac`, `schedule.decay_frac`, stage weights | SD-1, SD-2 | - |
| `precision`, `optimizer.muon_ns_dtype` | PR-1, PR-2 | - |
| `embedding.type` (KE vs dense control) | KE-1 | EXP-5 (with Team 4) |

Not decided here: `n_layers/d_model/n_heads/d_ff/seq_len/batch/steps` (TSAI, 31 Oct), the KE2 output shape (Team 4), the tokenizer (Team 3).

## 2. Protocol

| item | value | source |
|---|---|---|
| Base config **B** | `configs/small/t4x2_dp2_moe.yaml` + `--set moe.num_experts=8` (4 layers, d_model 256, E8 top-2, per-expert d_ff 384, seq 256, global batch 16, Muon + AdamW, WSD, lr 3e-4, wd 0.1) | sweep M2 (same `--set`); `configs/small/t4_moe.yaml` |
| Common `--set` | `--set tokens.train_steps=1000 --set checkpoint.every_steps=1000 --set general.seed=<S>`; val-set size stays at 4 windows per source (48 windows) until C1b lands, then add `--set data.val_windows_per_source=32` | sweep argv (`run.json`) |
| Command | `torchrun --nproc_per_node=2 harness/train.py --device cuda --config configs/small/t4x2_dp2_moe.yaml <sets> --out <run>` | results_v0 README |
| Seeds | **2 seeds (1234, 1235) per arm** that decides a v1 setting; A0 adds 1236. 1 seed only for diagnostics (marked) | lead rule "each arm has metric/pass bar/hours"; noise below |
| Steps | 1000 steps = 16 x 256 x 1000 = 4.10M tokens per run; the data set has 9.0M train tokens, so no epoch repeat | results_v0 README (9.0M tokens) |
| GPU | Kaggle 2xT4, fp32, dp2, deterministic algorithms, from 1 Nov. Shared H100 node from week 2 via TSAI's submit system (no keys); H100 arms are marked | overview ("shared H100 node comes in week 2") |
| **Primary metric** | held-out **BPB**, equal weight per source (12 sources, same number of windows each); logged now with per-source `held_out_loss_<src>`. Secondary: held-out CE, `step_time_ms` | plan sheet "report in bits per byte" (my-plan, week 2) |
| Noise floor | arm **A0** (below). `N0` = max - min of final BPB over seeds 1234/1235/1236 on B | A0 |
| **Pass bar (improve)** | `BPB(arm) - BPB(B) <= -max(0.01, N0)` on **both** seeds (paired by seed) and no stability flag | - |
| **Pass bar (non-inferior)** | `BPB(arm) - BPB(B) <= +max(0.01, N0)` on both seeds, no flag, and the arm is simpler or cheaper (used to adopt a cheaper/safer option) | - |
| BPB units | BPB/CE = 1.6124/6.3144 = 0.2554 (D0 final). So 0.01 BPB = 0.039 nats, and the ±0.02-nat noise estimate = ±0.005 BPB [INFERENCE: noise estimate from the sweep owner, not measured] | sweep D0 |

**Stability flags** (any one fails the arm; each is read from `metrics.jsonl`):

| flag | rule | source |
|---|---|---|
| NaN | non-finite loss or grad norm; run exits (code 3, torchrun 1) | results_v0 R5; S15 L363-365 |
| spike | `spike_score > 7` (`safety.spike_sigma`) | config_v0 `safety.spike_sigma: 7` |
| attention logit | `max_attn_logit > 100` | S15 L713 |
| dead experts | `dead_experts > 0` at any step after 50 | sweep: 0 for M1-M4 after step 50 (M5, balancing off, had 192/1000 steps with a dead expert from step 34); S15 L715 |
| MaxVio | mean of the last 100 steps > 1.0, per layer, from `moe_layers[].max_vio` (logged every `logging.per_layer_every` = 10 steps, so 10 samples in the last 100) [INFERENCE: adapts S15 L714 "above 1.0 for 1,000 steps" to a 1000-step run]. Sweep arms were read with the proxy `top_expert_share x E - 1` (section 4) | S15 L714; sweep proxy (section 4) |
| `ema_z` | NOT a flag: false alarms at step 1-10 (review §2.7) | review §2.7 |

**GPU-hours method.** `T4-GPU-h = runs x steps x (median ms/step / 1000) x 1.15 x 2 GPUs / 3600`. Median ms/step is measured in the sweep (D0 282, M1 388, M2 505, M3 755, M4 542 ms, dp2 on 2xT4). `1.15` is overhead for step 0 (1.2-1.5 s in the logs), eval every 50 steps on the larger val set, and checkpoint [INFERENCE]. `2` counts both T4s. Worked example, one B run: `1000 x 0.505 x 1.15 x 2 / 3600 = 0.323 T4-GPU-h` (0.161 h wall). Arms without a measured ms/step use B's 505 ms unless stated [INFERENCE]. H100: no step time measured yet; see section 5.

**Kaggle quota** (read with the Kaggle quota tool on 26 Oct): `total_time_allowed` 108000 s = 30 h/week, used 7393 s = 2.05 h, refresh 31 Oct 00:00 UTC. Whether a 2xT4 session counts 1x or 2x is unknown [INFERENCE: worst case 2x]. Week demand in section 5 fits either way.

### Code the arms need (none of it exists today unless marked)

| id | change | files | needed by |
|---|---|---|---|
| C1 | **Done:** per-source `held_out_loss_<src>` and equal-weight BPB (bytes of the evaluated target tokens) logged in `train.py`. **C1b, still open:** `data.val_windows_per_source` (hardcoded 4 at `data.py:12`; 48 windows) and a fixed `data.val_seed` (`data.py:57` seeds the val windows from `general.seed`, so a seed arm also changes the val set) | `data.py`, `check_config.py` | A0 noise floor, every arm |
| C2 | **Done:** `moe_layers` per-layer list logged every `logging.per_layer_every`: `max_vio`, `router_entropy`, `intervention_rate`, `topk_mass`, load, dead experts, aux | `train.py`, `model.py` | MO-3, MO-4, MO-5, flags |
| C3 | read `optimizer.matrices_2d` (today ignored; `optim.py:28` always sends block matrices to Muon): `adamw` puts them in the AdamW decay group | `optim.py` | OP-1 |
| C4 | `model.init_scaling: none\|num_layers` (std / sqrt(2 x n_layers) on `wo`, `w_down`, expert `w_down`; now `model.py:169-176` uses `init_std` everywhere) | `model.py` | AR-1 |
| C5 | `model.nope_every` (skip RoPE on every Nth layer) and `data.doc_mask` (position reset + block-diagonal attention at EOS; windows cross EOS today, `data.py:43`) | `model.py`, `data.py` | AR-2, AR-5 |
| C6 | `moe.first_dense_layers` + `moe.dense_d_ff` (every block is MoE today, `model.py:140`) | `model.py` | MO-3 |
| C7 | `precision: bf16` + `optimizer.accumulate_grad_in_fp32` (`check_config.py:111` rejects non-fp32) | `train.py`, `check_config.py` | PR-1 |
| C8 | `embedding.type: dense` control (model always builds `KroneckerEmbedding`, `model.py:154`) + KE2 shape from Team 4 | `model.py` | KE-1 |

## 3. Arms

"Now?" = can the current harness run the arm: **Y** = every key in `--set` is read today and the metric is logged; **C#** = needs that code (C1 and C2 are done, C1b only enlarges the val set). Cost cell = `runs x median ms/step -> T4-GPU-h` at 1000 steps, factor 1.15 x 2 / 3600 (H100 arms: section 5). All arms compare against **B** at the same seed unless a baseline is named. `--set` shows only the delta from B.

### 3.1 Noise and determinism

| id | question | `--set` | baseline | metric | pass bar | runs | cost | source | Now? |
|---|---|---|---|---|---|---|---|---|---|
| A0 | (a) same seed twice byte-identical on the current tree? (b) seed-to-seed BPB spread N0 of B | seed 1234 twice; `general.seed=1235`; `general.seed=1236` | - | per-step `losses.tsv` diff; final BPB per seed | (a) `compare_runs.py --exact` empty diff; (b) N0 = max - min BPB over 3 seeds, **reported** (it sets every other bar). Fails if (a) differs. Already seen: M2 rerun byte-identical in one session and across two Kaggle sessions (sweep, section 4); A0 repeats it on the final tree | 4 x 505 ms -> 1.29 | G5.md:34 (lead's gate); results_v0 R1a/b; Kaggle M2 reruns | Y |

### 3.2 MoE (EXP-3)

Granularity arms keep active FFN width `top_k x d_ff = 768` (= dense d_ff), as in the sweep.

| id | question | `--set` | baseline | metric | pass bar | runs | cost | source | Now? |
|---|---|---|---|---|---|---|---|---|---|
| MO-0 | does MoE beat the dense twin? | `moe.enabled=false model.d_ff=768` (sweep D0) | B | BPB; `step_time_ms` | improve bar for **B vs this arm** (B must win by max(0.01, N0) to justify MoE) | 2 x 282 -> 0.36 | sweep D0 vs M2; S14 §20 "dense or MoE?" | Y |
| MO-1a | fewer, larger experts | `moe.num_experts=4 moe.top_k=2 model.d_ff=384` (sweep M1) | B | BPB, MaxVio flag | non-inferior; adopt only if cheaper | 2 x 388 -> 0.50 | sweep M1; S14 §20 | Y |
| MO-1b | finer experts, k=4 | `moe.num_experts=16 moe.top_k=4 model.d_ff=192` (sweep M3) | B | BPB, MaxVio, `step_time_ms` | improve | 2 x 755 -> 0.96 | sweep M3; S16 L344 (best granularity rises with compute) | Y |
| MO-1c | finest, k=8 | `moe.num_experts=32 moe.top_k=8 model.d_ff=96` | B | BPB, MaxVio, `step_time_ms` | improve | 2 x 1255 -> 1.60 (1255 = 755 + 16 experts x 31 ms, linear extrapolation E8->E16 [INFERENCE]; perf cliff at E>=32, review §2.1) | S14 §20 (Kimi K2 chose 48 experts per active by matched-size runs); S16 L344 | Y |
| MO-2 | shared expert at matched active params | `moe.num_experts=8 moe.top_k=1 moe.shared_experts=1` (sweep M4) | B | BPB, dead experts | improve (else keep 0 shared: simpler) | 2 x 542 -> 0.69 | sweep M4; S14 L446; nanotron `nn_moe.py:193-197` | Y |
| MO-3 | first layer dense (routing settles slowly in layer 0) | `moe.first_dense_layers=1 moe.dense_d_ff=768` | B | BPB, layer-0 MaxVio | non-inferior and layer-0 MaxVio lower than B on both seeds | 2 x 505 -> 0.65 [INFERENCE: ms] | S14 L220-222; S15 L313-317 | C6 |
| MO-4 | bias vs aux vs both | bias only: `moe.seq_aux_loss_coef=0.0`; aux only: `moe.bias_update_speed=0.0`; neither: both 0.0 (sweep M5, already run once on seed 1234; run seed 1235 only) | B (= both) | dead-expert steps, MaxVio, BPB | adopt "both" unless a single-mechanism arm is non-inferior on BPB **and** has no more dead-expert steps and a clean MaxVio flag. Sweep M5 says BPB alone cannot pick: neither-arm BPB equals B within noise, but 192/1000 steps had a dead expert | 2 + 2 + 1 = 5 x 505 -> 1.61 (M5 seed 1234 could be reused: -0.32) | S14 §20; DeepSeek-V3 §2.1.2 (review §4); sweep M5 vs M2 (section 4) | Y |
| MO-5 | bias speed | `moe.bias_update_speed=1.0e-4`; `moe.bias_update_speed=5.0e-3` | B (1e-3) | MaxVio trailing mean, steps with a dead expert, BPB | lower MaxVio trailing mean than B on both seeds with BPB non-inferior | 4 x 505 -> 1.29 | S14 §20 (speed near 0.001); review §4 | Y |

### 3.3 Optimizer and schedule (EXP-1, EXP-2)

| id | question | `--set` | baseline | metric | pass bar | runs | cost | source | Now? |
|---|---|---|---|---|---|---|---|---|---|
| OP-1 | Muon vs AdamW-only, each given an LR pick | `optimizer.matrices_2d=adamw optimizer.lr=3.0e-4`; same with `optimizer.lr=1.0e-3` | B (Muon) | BPB | Muon stays unless AdamW beats it by max(0.01, N0) on both seeds at its better LR; tie goes to Muon only if non-inferior (S11 L625: Muon's edge falls to 1.1x at 1.2B, so confirm at H100 size: HC-2) | 4 x 505 -> 1.29 | S11 §14 (open: AdamW or Muon); S11 L625; Admin G6 answer 25 Oct | C3 |
| OP-2 | beta2 0.95 vs 0.999 (affects only AdamW groups: head, KE projection, router, norms; `optim.py:28-33`) | `optimizer.adam_betas=[0.9,0.95]` | B | BPB | improve | 2 x 505 -> 0.65 | SmolLM3 :215 (0.95) vs S11 L217 (0.999); review §3 | Y |
| OP-3 | peak LR | `optimizer.lr=1.0e-4`, `6.0e-4`, `1.0e-3`, `2.0e-3` (seed 1234 only), then the best point on seed 1235 | B (3e-4) | BPB curve, grad norm, flags | change lr only if the best point beats B by max(0.01, N0) on both seeds; always report the curve (optimum is broad: 2x error costs little, 10x a lot) | 5 x 505 -> 1.61 | S16 L428-429; S11 §2 (config `lr`); plan sheet "one arm of the LR sweep" per week-2 rotator | Y |
| OP-4 | LR x WD law `lr*wd = 17.04/steps` | `optimizer.weight_decay=0.0`, `1.0`, `10.0`, `56.8` at lr 3e-4 (law: wd = 17.04/(3e-4 x 1000) = 56.8, 568x today's 0.1) on seed 1234, then best-of-grid on seed 1235 | B (wd 0.1) | BPB, flags | law "holds" only if wd 56.8 beats B by max(0.01, N0) on both seeds; otherwise keep 0.1 and record that the law does not transfer at this size/steps (S16 L431-433: check on your own small runs; the MAI law is fitted on Muon recipes) | 5 x 505 -> 1.61 | S16 L419; config_v0 `weight_decay` comment | Y |
| OP-5 | warmup length | `schedule.warmup_frac=0.005`; `schedule.warmup_frac=0.10` | B (0.02) | BPB, max `update_to_weight_ratio` in steps 0-100, flags | improve, and no flag | 4 x 505 -> 1.29 | SmolLM3 :205-212 (2000 steps, 0.04%); S11 L449, L456-457; S15 L250 | Y |

### 3.4 Architecture

| id | question | `--set` | baseline | metric | pass bar | runs | cost | source | Now? |
|---|---|---|---|---|---|---|---|---|---|
| AR-1 | depth-scaled init | `model.init_scaling=num_layers` (new key) | B | BPB, max grad norm steps 0-100, step-0 CE | non-inferior; adopt only if grad-norm peak is lower. 4 layers under-states the effect of depth [INFERENCE] | 2 x 505 -> 0.65 | nanotron `models_config.py:15`, `parametrization.py` (NUM_LAYERS); SmolLM3 :166-167 | C4 |
| AR-2 | NoPE every 4th layer | `model.nope_every=4` (new key; with 4 layers = layer 4 only) | B | BPB at seq 256, flags | non-inferior (its benefit is long context, not measurable at seq 256) | 2 x 505 -> 0.65 | SmolLM3 :201; nanotron `models_qwen.py:272`; S8 §16 | C5 |
| AR-3 | QK-norm off | `model.qk_norm=false` | B | BPB, `max_attn_logit` | **keep ON** unless OFF beats B by the improve bar and `max_attn_logit` <= 100; at 4 layers instability may not show [INFERENCE] | 2 x 505 -> 0.65 | S15 L277; S15 §18 | Y |
| AR-4 | z-loss 0 vs 1e-4 | `head.z_loss_coef=0.0` | B (1e-4) | held-out BPB (CE only, `train.py:268`), `z_loss`, `max_attn_logit` | non-inferior to remove; otherwise keep | 2 x 505 -> 0.65 | S15 L213; SmolLM3 :199-200 (off, 1e-5); nanotron `models_qwen.py:888-892`; S9 §23 | Y |
| AR-5 | document masking | `data.doc_mask=true` (new key) | B | BPB, `step_time_ms` | improve; unmasked kept if cost > +15% ms [INFERENCE: threshold] | 2 x 560 -> 0.72 (560 = 505 x 1.11 masked-SDPA overhead [INFERENCE]) | SmolLM3 :173 (`_use_doc_masking`), :141 | C5 |

### 3.5 Schedule and data

| id | question | `--set` | baseline | metric | pass bar | runs | cost | source | Now? |
|---|---|---|---|---|---|---|---|---|---|
| SD-1 | anneal stage aligned with LR decay start (SmolLM3 0.889 / 0.111; ours 0.82 / 0.10) | `data_stages.2.start_frac=0.889 schedule.decay_frac=0.111` | B | BPB, per-source BPB | non-inferior (removes a free parameter) | 2 x 505 -> 0.65 | SmolLM3 stage3_9T_11T.yaml:484,541-542: 4198001/4720000 = 0.8894, 522000/4720000 = 0.1106; review §2.4 | Y |
| SD-2 | more code in stage 2 and anneal | `data_stages.1.weights.code=0.34 data_stages.1.weights.fineweb_edu=0.11 data_stages.2.weights.code=0.36 data_stages.2.weights.fineweb_edu=0.12` (each stage still sums to 1) | B | code BPB, Indic mean BPB, equal-weight BPB | code BPB improves by max(0.01, N0_code) on both seeds **and** Indic mean and overall BPB non-inferior. Harness-size signal only; the real mix is the data teams' call [INFERENCE] | 2 x 505 -> 0.65 | S5 ("a proportion is an opinion until a proxy run has tested it"); config_v0 `data_stages` comments (S15 §19) | Y (per-source via `held_out_loss_<src>`) |

### 3.6 Embedding (EXP-5, with Team 4)

| id | question | `--set` | baseline | metric | pass bar | runs | cost | source | Now? |
|---|---|---|---|---|---|---|---|---|---|
| KE-1 | Kronecker input vs dense-table control | `embedding.type=dense` (new key) plus Team 4's KE2 shape (`embedding.ke2_output_shape`) | B | BPB (all results reported in bits per byte) | KE stays unless dense beats it by the improve bar; the parameter and memory saving is the reason to keep KE on a tie | 2 x 505 -> 0.65 | S7 §13 ("dense table as the control arm"); plan sheet EXP-5 | C8 + Team 4 |

### 3.7 Precision and H100 arms

H100 cost is per **single-GPU** run; see section 5 for the arithmetic. These arms need the shared H100 node.

| id | question | `--set` | baseline | metric | pass bar | runs | cost | source | Now? |
|---|---|---|---|---|---|---|---|---|---|
| H0 | H100 calibration: fp32 B, byte-identical on H100? ms/step? | seeds 1234 twice, 1235; B on 1 GPU (`parallelism.dp=1`) | B on T4 (BPB within noise) | `losses.tsv` diff, median `step_time_ms`, BPB | repeat byte-identical; |BPB(H100) - BPB(T4)| <= max(0.01, N0). **Step time replaces every H100 estimate below** | 3 x 0.16 = 0.48 H100-h (ceiling, section 5) | review §6 (determinism proven only on one Kaggle session) | Y (node access) |
| PR-1 | bf16 + fp32 accumulation vs fp32 | `precision=bf16 optimizer.accumulate_grad_in_fp32=true` | H0 fp32, same seed | BPB, tokens/s, byte-identical rerun (seed 1234 twice) | non-inferior on both seeds, rerun identical, speed-up >= 1.5x [INFERENCE: threshold] | 3 x 0.16 = 0.48 | S10 §16 (precision open: short run in each); SmolLM3 :203,250,253 (bf16, fp32 accumulation) | C7 + node |
| PR-2 | Muon Newton-Schulz in bf16 (torch default) vs fp32 | `optimizer.muon_ns_dtype=bf16` (key read, `optim.py:100`; T4 has no bf16, so H100 only) | H0 fp32 | BPB, `step_time_ms` | non-inferior on both seeds | 2 x 0.16 = 0.32 | review §2.3 (torch Muon uses bf16 NS) | Y (node) |
| HC-1 | confirm MoE granularity at the TSAI size: top 2 from MO-1 | the two MO-1 winners' `moe.*` + `model.d_ff`, at the 31 Oct sizes | the other winner | BPB, MaxVio | improve, same bar | 4 x 0.94 = 3.74 | S14 §20; S16 §13 (proxy before full) | node + sizes |
| HC-2 | confirm Muon vs AdamW at the TSAI size | `optimizer.matrices_2d=muon` vs `adamw` (best LR from OP-3) | AdamW | BPB | Muon keeps its place only if it wins by the improve bar (S11 L625: edge shrinks with size) | 4 x 0.94 = 3.74 | S11 §14, L625 | node + sizes + C3 |
| V1 | all adopted winners together vs B | union of the passed arms' `--set` | B | BPB, flags | no flag; BPB <= B - max(0.01, N0) on both seeds, else drop the last-added arm | 2 x 505 -> 0.65 | G5 week 2 (config_v1 + decisions.md) | after arms |

## 4. What the MoE sweep shows and does not show

Sweep: 1000 steps, dp2 on 2xT4, 9M-token mix, seed 1234, one run per arm, val set 48 windows x 256 tokens (the sweep owner's noise estimate ±0.02 nats [INFERENCE]). Raw `metrics.jsonl` per run sit in the scratch area (not committed). All five runs finished (`status.txt` exit 0).

| arm | params | active FFN | held-out CE | BPB | dCE vs D0 | dBPB vs D0 | median ms | vs D0 | top share max (step >= 50) | MaxVio proxy: steps > 1.0 / last-100 mean |
|---|---|---|---|---|---|---|---|---|---|---|
| D0 dense | 38.8M | 768 | 6.314 | 1.6124 | - | - | 282 | 1.00x | - | - |
| M1 E4 k2 | 41.2M | 2 x 384 | 6.301 | 1.6089 | -0.014 | -0.0035 | 388 | 1.38x | 0.488 | 0 / 0.21 |
| M2 E8 k2 | 45.9M | 2 x 384 | 6.285 | 1.6048 | -0.030 | -0.0076 | 505 | 1.79x | 0.438 | 170 / 0.41 |
| M3 E16 k4 fine | 45.9M | 4 x 192 | 6.299 | 1.6085 | -0.015 | -0.0039 | 755 | 2.68x | 0.213 | 213 / 0.48 |
| M4 E8 k1 + 1 shared | 47.1M | 384 + 384 | 6.281 | 1.6039 | -0.033 | -0.0085 | 542 | 1.92x | 0.273 | 3 / 0.30 |
| M5 E8 k2, balancing off | 45.9M | 2 x 384 | 6.2871 | 1.6054 | -0.027 | -0.0070 | n/a | n/a | 0.50 (last-100 mean 0.250; M2 0.177) | first dead expert at step 34; 192/1000 steps with >= 1 dead expert |

MaxVio proxy = `top_expert_share x E - 1` (max over layers), computed by me from the logged share over steps 50-999 (950 steps) [INFERENCE: a proxy; the harness did not log MaxVio then]. M5 = M2 with `moe.bias_update_speed=0.0 moe.seq_aux_loss_coef=0.0` (rerun on Kaggle, exit 0; M5 CE/BPB and dead-expert numbers from the sweep owner, dCE/dBPB vs D0 computed here; M5 ms not read). The sweep BPB used the old token-weighted bytes per token (~1% different in the CPU check; the review measured ~10% on the Kaggle manifest), so **compare sweep arms by held-out CE**.

**Shows**
- All four MoE layouts train stably next to dense: no NaN, 0 dead experts after step 50, step-0 CE 11.82-11.85 against ln V 11.78.
- Matched-active MoE is **not worse** than dense at this size, and every MoE arm is slightly better (-0.014 to -0.033 nats).
- Balancing is not equal across layouts: E8 and E16 ran above MaxVio-proxy 1.0 in 18% and 22% of steps (156 and 182 of those before step 300, last one at step 874 for both); M4 (shared expert) and M1 (E4) stayed near or below 1. Dead = 0 after step 50 does not mean balanced.
- **Balancing off collapses routing early, and BPB does not see it** (M5 vs M2, same shape): held-out CE 6.2871 vs 6.2849 (+0.002 nats, noise), but the first step with a dead expert is 34, 192 of 1000 steps have >= 1 dead expert, up to 7 of 8 experts are dead in one step (summed over 4 layers), top-share max 0.50 vs 0.49, last-100 mean 0.250 vs 0.177. It recovers by the end (last-100 mean dead experts 0). So at 4 layers and 1000 steps the balancing is a stability guard, not a quality gain: MO-4 must be judged on dead-expert steps and MaxVio.
- Determinism: the M2 rerun in the same Kaggle session is byte-identical (1000 steps), and M2 is byte-identical across two separate Kaggle sessions (first sweep vs M5 rerun session).
- Step time on T4 rises with expert count (1.38x to 2.68x of dense for the same active FFN).

**Does not show**
- Any ranking. One seed, 48 val windows. M2 (-0.030) and M4 (-0.033) sit at the edge of the ±0.02 estimate; M1 and M3 are inside it. **No arm meets the plan's own bar** (0.039 nats = 0.01 BPB). Treat "MoE beats dense" as unproven until MO-0.
- Anything at 4.1M tokens about behaviour at the real width, depth or token count.
- Per-layer behaviour (only max/sum over layers was logged in the sweep), seed-to-seed variance (single seed per arm), or whether balancing-off collapse persists at larger E or longer runs (M5 recovered by step 1000).
- The step-time ratios are partly the Python per-expert loop (`model.py:114`, review §2.1) and T4 fp32, not a cost of the architecture on H100.
- BPB in the sweep uses the old token-weighted bytes per token (see above); CE is on the same windows, so use CE to compare sweep arms. The harness now logs equal-weight BPB and `held_out_loss_<src>`; new runs are not comparable to sweep BPB.

**Follow-up arms it motivates:** A0 first (nothing else can be read without N0); MO-0 (a second seed for D0 vs B); MO-1a/1c (extend the E4-E16 range, E32 not yet run); MO-2 (M2 vs M4 is the closest pair); MO-4 (seed 1235 for M5 and the single-mechanism arms; judge by dead-expert steps and MaxVio, not BPB) and MO-5 (E8/E16 imbalance); MO-3 (layer-level load now logged, so the layer-0 effect can be read); HC-1/HC-2 (transfer to the TSAI size).

## 5. GPU-hour budget and order of execution

H100 arithmetic [INFERENCE: not measured; replaced by H0's step time on day one]:
- Harness-size run on 1 H100: ceiling = wall time of one 2xT4 B run = 0.161 h (an H100 is not slower than 2xT4 for this model). 3 + 3 + 2 runs = 8 x 0.16.
- HC proxy run: 1e18 FLOPs / (0.30 MFU x 989e12 FLOP/s bf16 dense peak) / 3600 = **0.94 H100-h** (1/10 of ladder rung 1e19, which is 9.36 H100-h under the same assumption; the ladder is outside this plan). 989 TFLOPS = NVIDIA H100 SXM spec; 0.30 MFU is an assumption.

| phase | arms | runs | cost (T4-GPU-h) | cost (H100-h) |
|---|---|---|---|---|
| 0: noise | A0 | 4 | 1.29 | - |
| 1: runnable now | MO-0, MO-1a/b/c, MO-2, MO-4, MO-5, OP-2, OP-3, OP-4, OP-5, AR-3, AR-4, SD-1, SD-2 | 43 | 14.76 | - |
| 2: needs code (C3-C8) | MO-3, OP-1, AR-1, AR-2, AR-5, KE-1 | 14 | 4.59 | - |
| 3: combine | V1 | 2 | 0.65 | - |
| H100: calibrate and precision | H0, PR-1, PR-2 | 8 | - | 1.28 |
| H100: confirm at TSAI size | HC-1, HC-2 | 8 | - | 7.49 |
| **Total** | | 63 T4 + 16 H100 | **21.29** | **8.77** |
| With 20% reserve for failed or repeated runs [INFERENCE] | | | **25.54** | **10.52** |

T4 check: 21.29 GPU-h = 10.6 h of 2xT4 wall time. Phases 0+1 = 16.05 GPU-h in the first week against the 30 h/week quota (fits even if counted 2x). T4 and H100 hours are separate pools and are not added.

| date | action |
|---|---|
| Sat 31 Oct | TSAI approves this plan; TSAI sizes arrive. Start C1b (val-set size and fixed val seed) |
| Sun 1 Nov | C1b lands; run A0 (0.65 h wall). Set N0, post it in G5 |
| 1-4 Nov | Phase 1 on Kaggle (7.4 h wall), order: OP-3 (rotators take one LR point each), MO-1, MO-4/MO-5, then the rest. C3-C8 PRs in parallel. Ask TSAI for H100 time |
| by Wed 4 Nov | **Queue** phase 2 as Kaggle committed runs (the Kaggle save-and-run call runs a notebook top to bottom unattended; session cap ~12 h [INFERENCE: verify]) and H0, PR-2 on the H100 node |
| 5-11 Nov (break, queued only) | Phase 2: 4.59 T4-GPU-h (2.3 h wall); H100: H0, PR-1, PR-2 (1.28 H100-h), then HC-1/HC-2 (7.49 H100-h) once sizes and MO-1/OP-1 winners are known (HC-1 needs MO-1 results, available 4 Nov) |
| Mon 9 Nov onward | read break results; no new code |
| Thu 12 Nov | V1 (0.65 T4-GPU-h); fill `decisions.md` per setting |
| Fri 13 Nov | review with TSAI; any pass-bar ties go to TSAI |
| **Sat 14 Nov** | `config_v1.yaml` frozen |

## 6. Open questions for TSAI / lead

1. **Sizes (31 Oct).** Please also fix the HC proxy size (I propose 1e18 FLOPs per run, 0.94 H100-h) and the token count per run; HC-1/HC-2 and the transfer check depend on it.
2. **H100 access.** From which date, how many H100-hours for Team 5 (plan asks 8.77, 10.52 with reserve), is a single-GPU job allowed, and does the queue accept a fixed seed and deterministic flags?
3. **Source policy.** The lead's rule names KE code, the paper, or a measurement from our tiny run. These arms cite lessons, SmolLM3 and nanotron files for settings KE does not cover. Is that accepted, or should each arm's source be a measurement (the arm itself becomes the source once it runs)?
4. **Pass bar.** Is `max(0.01 BPB, N0)` on both of 2 seeds acceptable, or do you want 3 seeds for the arms that change the optimizer or MoE shape (+50% cost)?
5. **LR x WD law.** The law at 1000 steps asks for wd 56.8 (568x today's 0.1). Do you want OP-4 run as written, or only after a units check with the MAI recipe?
6. **Data for HC.** The harness data set is 9.0M tokens (4.1M used per run). HC runs at 1e18 FLOPs need more; do we use the data teams' cleaned shards, and from when?
7. **Kaggle quota.** One account has 30 h/week. Is the team pooling several accounts, and does a 2xT4 session count 1x or 2x?
