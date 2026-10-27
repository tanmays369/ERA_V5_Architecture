Name: dharshan
Axiom email: dharshanthaigu@gmail.com
Pack: ind-ta-fw2

# plan_v0: G5 experiment plan (draft for TSAI approval on Sat 31 Oct)

Status: draft. No arm has been run, so every GPU-hours cell says "to measure". TSAI approves on Saturday.

## How an arm is judged

- Metric for every arm: the final `held_out_loss` in `metrics.csv` (next-token loss, mean over data sources with equal weight per source). `loss` is not used because it also contains the MTP, z-loss and balance terms, which differ between arms.
- Noise first: arm A0 runs the baseline with seeds 1234, 1235 and 1236. `noise range` = max minus min of the three final held-out losses.
- Pass bar = `max(0.01, noise range)`, applied on BOTH seeds 1234 and 1235. Every arm below therefore has 2 runs (A0 has 3).
- An arm that adds or changes something passes if its loss is lower than the baseline run with the same seed by more than the pass bar, on both seeds. An arm that removes something (MO-3, AR-2, HE-1) shows the feature is worth keeping only if removing it makes the loss worse by more than the pass bar on both seeds.
- A result inside the pass bar is reported as "no difference found", not as better or worse.
- The pass bars are fixed before any run and are not changed after seeing results. If a bar turns out to be wrong, the change is written down with the reason and the arm is re-run.
- Every arm reports total and active parameters (`run_info.json`: `params_total`, `params_active`) next to the loss, so a gain from a bigger model is not mistaken for a gain from the idea.
- Same seed 1234 for the wiring checks; seeds 1234 and 1235 for every quality comparison, as in A0.

## Arms

Exact `--set` values are for the tiny config `configs/config_small.yaml` (2 layers, d_model 64, 4 experts, top_k 2, d_ff 128, seq_len 64, 60 steps). Values for the real sizes are open until TSAI gives the sizes (Sat 31 Oct). Several `--set` values are given in one run as repeated `--set` flags. All values I propose are [ours]; the arm ids come from the comments in config_v0.yaml.

| id | question | exact --set | baseline | metric | pass bar | runs | GPU hours | source |
|---|---|---|---|---|---|---|---|---|
| A0 | How noisy is one run? | `general.seed=1235`, `general.seed=1236` (1234 is the default) | config_small.yaml | final held_out_loss | defines the bar | 3 | to measure | [ours] |
| MO-1 | Finer MoE granularity at equal active size? | `moe.num_experts=8 moe.top_k=4 model.d_ff=64` | A0, seeds 1234 and 1235 | final held_out_loss | max(0.01, noise range) on both seeds | 2 | to measure | [ours] |
| MO-2 | Does one shared expert help? | `moe.shared_experts=1` | same | final held_out_loss | same | 2 | to measure | [ours] |
| MO-3 | Is expert balancing needed? | `moe.bias_update_speed=0 moe.seq_aux_loss_coef=0` | same | final held_out_loss; also `max_vio`, `dead_experts` | same | 2 | to measure | [ours] |
| MO-4 | Does sharing one router across layers hurt? | `moe.router_sharing=all_layers` | same | final held_out_loss; also `max_vio` | same | 2 | to measure | [ours] |
| HE-1 | Does the MTP head help the next-token loss? | `head.mtp_heads=0` | same | final held_out_loss | same | 2 | to measure | [ours] |
| AR-1 | Pre-norm or post-norm? | `model.norm_position=post` | same (pre) | final held_out_loss | same | 2 | to measure | [ours] |
| AR-2 | Does QK-norm help? | `model.qk_norm=false` | same (on) | final held_out_loss | same | 2 | to measure | [ours] |
| AR-3 | Does a sliding window hurt? | `attention.window=32` | same (0 = full) | final held_out_loss | same | 2 | to measure | [ours] |
| OP-1 | Muon or AdamW for the matrix weights? | `optimizer.matrices_2d=adamw` | same (muon) | final held_out_loss | same | 2 | to measure | [ours] |
| OP-2 | Which learning rates? (week 2: one value per student) | `optimizer.lr_muon=0.01 optimizer.lr_adamw=0.0015`, then `0.04` and `0.006` | same (0.02 and 0.003) | final held_out_loss | same | 2 per value | to measure | [ours] |
| OP-3 | Weight decay and betas? | `optimizer.weight_decay=0.1`, `optimizer.betas=[0.9,0.99]` (two arms) | same (0.0 and [0.9,0.95]) | final held_out_loss | same | 2 per arm | to measure | [ours] |
| OP-4 | Warm-up and decay lengths? | `schedule.warmup_frac=0.2`, `schedule.decay_frac=0.5` (two arms) | same (0.1 and 0.3) | final held_out_loss | same | 2 per arm | to measure | [ours] |
| KE-2 | Does decay on the KE projection hurt? | `embedding.weight_decay=0.1` | same (0.0) | final held_out_loss | same | 2 | to measure | [ours] |
| KE-1 | KE or a dense vocab x d_model embedding (control)? | open: needs a dense-embedding code path (not written; `embedding.type` accepts only `kronecker`) and G4's KE2 output shape | A0 | final held_out_loss | same | 2 | to measure | open: G4, Tue 27 Oct |

Notes:
- AR-3: on the tiny model seq_len is 64, so window 32 is half the context. The sliding-window source stays [ours] until a source is verified.
- OP-1: Muon and AdamW each need their own learning rate; a fair comparison needs an AdamW learning-rate check first. That check is not in this plan and is open for the lead to decide.
- KE-1: parameter counts differ (a vocab-sized table against a projection), so the arm compares loss only.
- The tiny synthetic source uses an alphabet of 64 tokens, so these arms test the harness and the switches, not language quality. Real arms need the parquet source and the real sizes (open: TSAI and the data teams).

## GPU hours

GPU hours = measured seconds per step x steps x runs / 3600. Seconds per step = `tokens.global_batch_seqs x tokens.seq_len / tokens_per_second` from one real run's `metrics.csv`. Until that run exists every cell says "to measure". Nothing here comes from memory.

Total GPU hours = sum of the cells above = to measure.

## Order of work

1. A0 first: every pass bar depends on it.
2. MO, HE, AR, KE-2 and OP arms in any order, one arm per person in weeks 2 and 3.
3. KE-1 after G4 gives the KE2 shape and a dense-embedding path exists.

## Open items

| item | owner | date |
|---|---|---|
| Model sizes (layers, d_model, heads, kv heads, d_ff, experts, top_k, seq_len, batches, steps) | TSAI | Sat 31 Oct |
| KE2 output shape | G4 | Tue 27 Oct |
| Learning rates, weight decay, betas, warm-up and decay fractions (real model) | arms OP-2 to OP-4 | Sat 31 Oct |
| Data source, paths and mixture weights | Data A and Data B teams | date not given |
| Trainer logger and alarms | G8 | Thu 29 Oct |
| Approval of this plan | TSAI | Sat 31 Oct |

## References and limits

- Sources named in this plan: the KE code (github.com/theschoolofai/kronecker-embeddings) and the paper arXiv 2605.29459 for the embedding; every other value is [ours]. The ideas behind aux-free bias balancing, Muon and QK-norm are not cited here because I have not read the sources; they are to verify.
- One harness, one synthetic data source, two seeds: results show direction for the switches, not language quality.
- A0 noise comes from 3 seeds of the tiny config; a real-size noise range must be measured again once TSAI gives the sizes.
- No arm has been run. The harness gate tests have not been run by me yet (see `harness/README.md`, section 11).
