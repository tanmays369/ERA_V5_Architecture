# Team 5 week-1 harness review (PR #55, `revantg:t5-harness-v0`)

Reviewer model: claude-proxy/claude-opus-5-5. Read-only on `teams/` and `logs/`; probes live in `work/review/` (`wrap_train.py`, `wrap_bias.py`, `unused_keys.py`, `runs/`).
Note: `harness/model.py` was modified at 22:04:22 IST while I was reviewing (adds the `moe_metrics` None guard, line 243). Line numbers below are the working tree at 22:05 IST. Check that PR #55 has this change.

Legend: met / partial / missing / wrong. `[INFERENCE]` = not run, reasoned from code or docs.

## 1. Requirements trace

| # | Ask (source) | Evidence | Status |
|---|---|---|---|
| A1 | `configs/config_v0.yaml`; each setting has a default + a comment naming the source (Admin 24 Oct, G5.md:25) | 144 lines, 121 leaves, every leaf has a ` #` comment; `check_config` gives `0 errors, 15 open` | **met** (letter) |
| A2 | "A source is the KE code, its paper arXiv 2605.29459, or a measurement from our tiny run" (lead 26 Oct, G5.md:449) | Only 2 of 121 leaves cite only those kinds (`tokenizer.sha256`, `tokenizer.eos_id`); 6 more cite KE+lesson. 36 cite a lesson, 13 SmolLM3, 10 nanotron, 4 `torch.optim.Muon`, 18 "contract / harness size / open", 11 have no named source (see §1.1) | **partial / risk** |
| A3 | Size values stay empty, TSAI fills 31 Oct (Admin, lead) | `model.{n_layers,d_model,n_heads,n_kv_heads,d_ff}`, `moe.{num_experts,top_k}`, `tokens.*`, `data.*`, `logging.eval_every` null and in `open_keys` | **met** |
| A4 | KE2 output shape empty until G4 replies (lead 26 Oct 13:38) | `embedding.ke2_output_shape: null`, `check_config` prints `open: ... (G4 Tue 27 Oct)` | **met** |
| A5 | "No V4 yaml, V4 code, V4 LR settings" (lead G5.md:469) | No V4 values copied. But `config_v0.yaml:135` comment says "V4 dashboard list" | **partial** (wording; §1.2) |
| A6 | First lines = Name, Axiom email, Pack (lead G5.md:472; contract amendment) | Files have `# Name` + `# Pack` only; no `# Axiom email` line in any file under `teams/` (grep: 0 hits). PR body has all three | **partial** (see §1.3) |
| A7 | One PR to theschoolofai/era-v5-lab | PR #55 open | **met** |
| B1 | `check_config.py` "reports/prints each setting that has no value or no source" (Admin G5.md:26, lead G5.md:454) | `check_config.py:64-75`: null + not in `open_keys` -> error; null + open -> `open: key (who)`; missing ` #` -> `error: key has no '# source' comment`. Exit 1 on errors | **met**, with blind spots (E8) |
| B2 | Saturday: "check_config gives 0 errors" | config_v0: `0 errors, 15 open`; all 12 small configs `0 errors, 1 open` (ke2) (run, §2.9) | **met** |
| B3 | Output wording "no value or no source" | Prints `open:` / `error: ... is null and not in open_keys` / `has no '# source' comment`. The word "value" never appears | **partial** (cosmetic) |
| C1 | `harness/train.py`, tiny MoE (Admin G5.md:27, lead G5.md:455-456) | `train.py` 397 lines; `t4_moe`/`cpu_smoke_moe` | **met** |
| C2 | Same seed twice -> same loss | R1a/b, R2a/b, R3a/b, R6a/b, R7a/b `exact: 300 steps byte-identical` (`results_v0/compare.txt`) | **met** (Kaggle 2xT4; one machine/session) |
| C3 | Loss falls | R0 11.15 -> 7.01, R6a 11.14 -> 7.02 (first 30 / last 30 mean CE) | **met** (weak gate, see §2.8) |
| C4 | No NaN | gate: all finite; NaN path tested R5 | **met** |
| C5 | "One small config beside config_v0, tiny sizes, Colab/Kaggle/laptop. This is the file G6 gets. It is not config_v0" (lead G5.md:452) | 12 files in `configs/small/`; G6's file is `small/t4_moe.yaml` and has `extends: t4_base.yaml -> ../config_v0.yaml`. Not "beside", not one, not self-contained | **partial** (P0, §5) |
| C6 | Harness + small config to G6 on 27 Oct (G6.md:38) | Exists. G6 also needs fp64 reference / importable model (Admin 25 Oct to G6). `model.build_model(cfg, tok_path)` is importable | **met** |
| D1 | Metric names to G8: `tokens_per_second, loss_ntp, loss_mtp, loss, lr, gpu_alloc_gb` (lead G5.md:458) | `results_v0/metrics_schema.json` + `train.py:39-61` + every `metrics.jsonl` row has all 6 | **met** (but `gpu_alloc_gb` semantics, §2.7) |
| D2 | G8 logger runs inside our harness (G8 gate Sat) | Harness writes its own `metrics.jsonl` every step, no logger hook | **missing, by design** (Thu 29 Oct) |
| D3 | `experiments/plan_v0.md`, each arm metric + pass bar + GPU hours (Sat) | `teams/05-architecture/experiments/` does not exist | **missing** (Sat; P1) |
| E1 | G7 needs config_v0 on 27 Oct, sizes empty | Delivered. `checkpoint:` has only `every_steps/strict_load/keep_last`; checkpoint format is per-rank `torch.save` of local shards (`train.py:101-110`), layout-bound | **partial** (G7 needs this documented) |
| E2 | G7: CPU restore -> bit-equal loss | R5 resumed byte-identical (Kaggle). I re-proved on CPU for ZeRO-1 dp2 and TP2 (E7): byte-identical | **met** |
| F1 | Admin 25 Oct to G6: BrahmicTokenizer-131K, Muon on 2D + AdamW on the rest | `config_v0.yaml:11-16, 58-59`; `optim.py:20-34` | **met** |
| F2 | Gate command in lead post: "same seed -> same loss" | `compare_runs.py --exact` / `--gate` | **met** |
| F3 | Do not commit data/scratch | `.gitignore` covers `work/ tokenizer/ data/ runs/ hf/`; `results_v0` holds only losses.tsv, 1 metrics.jsonl, run.json | **met** |

### 1.1 Source-comment policy risk (A2)
Measured with a script over the 121 leaves in `config_v0.yaml`:
```
leaves 121
leaves whose comment names ONLY lead-approved source kinds (KE code/arXiv/measurement): 2
36 lesson | 18 ours/contract/open | 13 SmolLM3 | 11 NONE | 10 nanotron | 8 lesson+ours | 6 admin/lead+lesson
6 KE/arXiv+lesson | 5 admin/lead | 4 torch | 2 measurement | 1 admin/lead+measurement | 1 SmolLM3+lesson
```
The Admin's own wording is only "a comment with its source" (G5.md:25). The lead narrowed it on 26 Oct 13:30 to three kinds. Lesson / SmolLM3 / nanotron / torch-default citations are real, checkable references. For a setting that KE or the paper does not cover (LR, WD, clip, betas, norm_eps, init) there is no KE/arXiv source and no tiny-run measurement exists yet, so the lead's rule cannot be met literally. They are a risk only if the lead reads the rule strictly. Mitigation, cheapest first:
1. Ask the lead in G5 one line: "settings outside KE (lr, betas, wd, clip, init, eps) cite a lesson section or SmolLM3/nanotron config line; OK, or do you want 'measured' arms in plan_v0?" Post it before Tue 23:59.
2. Prefix every comment with a source class tag (`[KE]`, `[paper]`, `[run]`, `[lesson S11]`, `[ref SmolLM3 stage1:215]`, `[default]`) so the lead can grep. The current free text ("harness size, T4 fits", "harness default") passes `check_config` but is not a source.
3. Where we do have a measurement (step-0 CE 11.84 vs ln V, `tokenizer.sha256`, eos id, held-out BPB), cite `results_v0/...` explicitly.
11 leaves have no named source at all (`general.project`, `general.run`, `datasets.3` ... `datasets.11`: code and Indic dataset rows say "FineWeb-2 Indic subset" or similar, which names a dataset, not a rationale). Flag them or cite the dataset card URL.

### 1.2 "V4"
`config_v0.yaml:135`: `# first 6: lead G5 26 Oct (V4 dashboard list, G8 hand-off)`. The lead's rule is that V4 yaml, code, and LR stay out. This comment copies nothing, but a skim-reviewer will read "V4" in a source column. Reword to `lead G5 26 Oct (the six names confirmed at the 8pm call)`. Lesson lines like `s11.md:512 "V4's batch of 56"` are not used by the config. I checked.

### 1.3 Header lines
`logs/contract.md:3` requires `# Axiom email:`. No file under `teams/05-architecture/` has it. Likely a deliberate choice: worklog line 27 notes the lab pr-guard regex flags email-shaped text in a patch. The lead's wording ("The first lines are Name, Axiom email, and Pack", G5.md:472) sits in the "How we write the team files" block, ahead of "One pull request", so it is ambiguous between file header and PR body. PR body has all three. Recommendation: keep out of files, say so in the PR body ("file headers carry Name+Pack only: the pr-guard flags emails"), or ask the lead.

## 2. Correctness review

Environment: torch 2.14 CPU in `.venv`, `cpu_smoke*.yaml`. Kaggle evidence read from `results_v0/`.

### 2.1 Determinism
- `train.py:75-83`: `use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG` set before CUDA init (`:8`), TF32 off, SDPA forced to math. Good.
- MoE `index_add_` (`model.py:117`), `gather` backward, `index_select` backward, `repeat_interleave` backward: all appear in the "act deterministically when the flag is on" list of `torch/__init__.py:1774-1805` (`scatter_add_`, `gather`, `index_add`, `index_select`, `repeat_interleave`). Not an error under the flag; the Kaggle MoE runs R6/R7 are byte-identical, which is the empirical proof. [INFERENCE for other GPUs / torch versions: the deterministic `index_add_` path on CUDA is slower; the per-expert Python loop with `nonzero()` (`model.py:114`) forces a host sync per expert per layer. Fine for E=4, a perf cliff for E>=32 (P2).]
- Data order: `data.py:42` `default_rng([seed, step])`, independent of rank/layout. Val stream `[seed, 2**32-1]` cannot collide with a train step. Good. Sampling is with replacement at random offsets, windows cross EOS (no doc masking). Fine for v0, see §3.
- DDP `broadcast_buffers` default True: buffers incl. `expert_bias`, the byte buffer and rope tables are re-broadcast each step (`model.py:157` comment confirms buffers matter to DDP). `expert_bias` is already identical across dp (load is all-reduced). Setting `broadcast_buffers=False` removes ~MBs of redundant comm. [INFERENCE: not measured; P2.]
- Cross-layout: dense layouts differ from R0 by <=2e-7 rel (fp32 reduction order); MoE dp2/pp2 vs 1 GPU 9.3e-5 / 8.4e-5 rel. The README attributes this to "a near-tied router score can flip one expert choice". That is plausible, but I could not confirm it. My PP check (E4 below) shows the step-0 bias update is the same sign pattern either way, so the PP load overcount is NOT the cause.

### 2.2 Loss normalisation under grad-accum / PP / DP
- Non-PP: `train.py:309` `(tot / ga).backward()`, equal-size micro-batches, no padding -> exactly the global-batch token mean. DDP averages over dp, so the gradient is the mean over `G` sequences. Correct.
- PP: `schedule` default `scale_grads=True` (checked: `Schedule1F1B.__init__(..., scale_grads: bool = True)`), so grads are the mean over `pp_microbatches`. Correct, and `check_config.py:118-122` forces `grad_accum == 1`.
- Experiment E6 (aux coef 0.5, 3 steps, `cpu_smoke_moe`): 
```
ga=4 vs ga=1:   11.786722183 / 11.786723137 ; 11.796762466 / 11.796762466 ; 11.786811828 / 11.786812782
pp2  vs ga=1:   11.786723137 / 11.786723137 ; 11.796764373 / 11.796762466 ; 11.78680419  / 11.786812782
```
Equal to fp32 noise (<=1e-6). So aux scaling (`aux_scale = 1/ga`, or 1.0 under PP because the schedule divides) is consistent. The aux gradient is NOT invariant by construction only if someone changes `scale_grads`; add a comment/assert (P2).
- z-loss: `model.py:251` `ce + c * mean(lse^2)` = PaLM form, same as nanotron `LossWithZLoss` (`models_qwen.py:888-892`). Coefficient 1e-4 (S15:213). SmolLM3 has `z_loss_enabled: false`, coef 1e-5 (stage1:199-200). Keep ours (explicit lesson source), note the difference.
- Held-out uses CE only (`train.py:268`, coef 0.0). Correct.

### 2.3 Muon (`optim.py`)
Compared line by line with `torch.optim.Muon` semantics:
- `lerp_` momentum + Nesterov (`:74-75`), `lr*0.2*sqrt(max(A,B))` for `match_rms_adamw`, `sqrt(max(1,A/B))` for `original` (`:84-88`), decoupled WD with base lr (`:90`). Matches.
- DTensor (TP) branch (`:76-78`): `full_tensor()` -> NS on the full matrix -> `distribute_tensor`. Correct (Muon needs the whole matrix), costs an all-gather per param per step. R2 (tp2) vs R0 <= 1.5e-7 rel confirms. P2: `redistribute`-based or per-rank-owner scheme if TP grows.
- 3D experts (`:79-80`): per-slice NS, and `A,B = p.shape[-2:]` gives the per-expert out/in for the lr adjust. Correct. Expert slices `[E, d_ff, d_model]` and `[E, d_model, d_ff]` get adjust factors from their own shapes.
- Router stays on AdamW, no WD (`:30-33`, S14 §20). Head AdamW with WD. Embedding projection AdamW no WD. Reasonable; Moonlight/Keller keep embed + head off Muon.
- NS dtype fp32 default (T4 no bf16); torch's own Muon uses bf16 NS. Different numerics from the reference: flag as an experiment arm (`muon_ns_dtype: bf16` on H100) and note `eps=1e-7` equals torch's.
- ZeRO-1 + Muon: each rank steps whole params, then broadcast. Correct; R4 vs R1a max rel diff 0.
- Not tested: Muon + `adjust_lr_fn` when `lr` schedule hits 0 (last step: no update, harmless).

### 2.4 WSD schedule (`optim.py:116-126`)
- Warmup `lr*(step+1)/W`, `W = ceil(0.02*T)`. Flat. Decay `lr + (lo-lr)*(step-(T-D)+1)/D`, so the last step (`T-1`) has lr exactly `lo = 0.0`. R6a confirms: step 299 `lr: 0.0`. The last optimizer step is a no-op (Adam/Muon update scaled by 0, WD factor 1). Wasted step; P2: use `(step-(T-D)+1)/(D+1)`.
- Anneal/decay misalignment. Config: data stage `anneal` at 0.82 (`config_v0.yaml:118`), decay starts at `1 - 0.10 = 0.90`. SmolLM3: stage3 `start_training_step` 4198001 = decay `lr_decay_starting_step` 4198001; of `train_steps` 4720000 that is **0.8894**, and decay is 522000/4720000 = **0.1106** (stage3 yaml:484, 541-542). The comment "SmolLM3 8T/9T/11T boundaries (9/11)" gives 0.818, but the actual step boundaries are 0.731 and 0.889. In SmolLM3 the anneal data stage and the LR decay start together; ours do not (0.82 vs 0.90). Either set `anneal.start_frac` to 0.89 and `decay_frac` to 0.11, or keep and cite `[ours]`. The current citation is wrong.
- Warmup: SmolLM3 2000/4.72M = 0.04% (stage1:210); nanotron tiny MoE example `lr_warmup_steps: 2`. Ours 2% = 6 of 300 steps. Fine for harness, a plan_v0 arm for real runs.

### 2.5 Gradient clipping under TP/PP/ZeRO (`parallel.py:129-158`)
`global_sq`: `Shard` DTensors summed over tp, replicated counted once, then pp all-reduce. `q_norm/k_norm` grads are all-reduced over tp before (`sync_tp_grads`, `:73-78`) so they are identical and counted once. `coef = max_norm/(norm+1e-6)`, applied when `<1`: same rule as `clip_grad_norm_`. Under DDP and ZeRO-1 grads are full and averaged before clipping. R2/R3/R4 vs R0 <= 2e-7 rel over 300 steps is the evidence. Correct.

### 2.4b MoE router and bias (see §4 for design). Correctness checks run:
- **E3, PP step-0 load overcount (bug, low severity).** Probe `work/review/wrap_train.py` prints each MoE layer's load sum before the bias update. Expected per layer per step = G*L*k = 8*64*2 = 1024.
```
PP  (pp2, 2 mb)  step0 [1536.0]  step1 [1024.0]  step2 [1024.0]
1 proc           step0 [1024.0, 1024.0]  step1 [1024.0, 1024.0]  step2 [1024.0, 1024.0]
```
1536 - 1024 = 512 = one extra pipeline micro-batch (4 seqs * 64 * 2). The first `schedule.step()` runs a shape-inference forward (`train.py:296` comment says it calls `loss_fn` once on uninitialised buffers) with grad enabled, so `Transformer.forward` (`model.py:204`: `counting = training and is_grad_enabled()`) counts it. Effect: the step-0 `expert_bias` update and the step-0 `dead_experts/top_expert_share` use a polluted load. E4 shows the step-0 bias came out the same on both layouts (`[-0.05,-0.05,0.05,0.05]` then zeros, speed 0.05) because the update is `sign(...)`, so the overcount did not flip a sign here. Fix: in `train.py`, zero `moe.load` after the first `pp_step` (or run shape inference with `torch.no_grad()`/`counting=False` by an explicit flag) (P1).
- **E1, `seq_aux_loss_coef: 0` crash (reproduced, now fixed in working tree).** My first run with `--set moe.seq_aux_loss_coef=0` exited 1 at step 0 with an empty `metrics.jsonl` (train.log stops after the layout line). Cause: `moe_metrics` did `float(moe.last_aux)`, and `last_aux` stays `None` when `aux_coef` is 0 (`model.py:123`). The file changed at 22:04:22 and now has `aux=0.0 if moe.last_aux is None else ...` (`:243`); 3 later runs exit 0 with `moe_aux: 0.0`. Needed for the "bias-only" arm, so keep it in the PR.
- Aux loss correctness: formula matches DeepSeek-V3 Eq. (sequence-wise balance): `per_seq*E/(k*L)` times mean router prob per sequence, mean over batch (`model.py:119-125`); E6 shows gradient scaling is layout-invariant.
- Bias update (`model.py:227-233`): all-reduce over dp, `bias += speed*sign(mean - load)`, `load` zeroed. Sums over all micro-batches of the step, and `counting` is off in checkpoint recompute (`:204-211`), shape inference of held-out (`no_grad`). Matches DeepSeek-V3 §2.1.2. Under PP each stage owns disjoint layers, so no cross-stage reduce is needed. Correct.
- Expert bias is a registered buffer (`model.py:96`) -> in `state_dict`, so it is checkpointed (see §2.6).

### 2.6 Checkpoint / resume completeness
Saved (`train.py:104`): step, model `state_dict` (includes `expert_bias`; rope/KE byte buffers are non-persistent and rebuilt), optimizer states (Muon momentum, Adam moments; ZeRO shard via `.optim`), `frozen_sha256`. LR is a pure function of step; data is a pure function of (seed, step). No dropout, so RNG state is not needed.
- Resume proved on CPU (E7, 20 steps, resume at 10, `compare_runs` on aligned steps): ZeRO-1 dp2 `byte-identical on aligned steps: True`; TP2 `True`. R5 (Kaggle) dp2 also. (`compare_runs --exact` as invoked on the whole file reports `FAIL first difference at step 0` only because the resumed run's `losses.tsv` starts at step 10, so the exact check must align by step; document this, §2.8.)
- **Not saved (metrics, not loss): spike/EMA state** (`hist`, `ema`, `ema_var`, `train.py:274-275, 349-358`). E5 (window 10, resume at 10):
```
step 10  loss eq True | spike full -1.98 res None | ema_z full -3.87 res None
step 11  loss eq True | spike full -1.51 res None | ema_z full -3.13 res 353813171.4
step 12  loss eq True | spike full -3.31 res None | ema_z full -5.26 res -978.6
```
So after every resume, `spike_score` is null for `spike_window` steps and `ema_z` restarts with a spurious huge value. G8 alarms on these will fire on every restart (P1). Fix: save `hist`, `ema`, `ema_var` in the checkpoint dict.
- No layout/config fingerprint in the checkpoint: `load_ckpt` checks only key-set equality (`:116`); resuming on a different world size loads `rank{r}.pt` blindly. G7 needs a manifest (world/pp/dp/tp, config hash, torch version) (P1).
- `torch.save` writes straight to the final path (`:104`); a crash mid-write leaves a partial `step_N` that `keep_last` rotation and `last_good` may later pick. Write to `.tmp` then `os.replace` (P2).
- `frozen_sha256` is saved but never verified on resume (P2).

### 2.7 NaN path and metric schema vs reality
- NaN: `bad` = non-finite loss or grad norm, all-reduced MAX (`:319-320`), exit 3 before the optimizer step, no checkpoint written, `NAN_REPORT.json` + resume command. R5 on Kaggle: `status.txt` `R5 exit=1` (torchrun maps worker exit 3 to 1; README says so). Correct. `losses.tsv` receives the NaN row before exit (`:322`), by design.
- **`ema_z` is wrong at step 1 (bug, shipped in evidence).** `train.py:351,355-358`: `ema_var` starts 0, so step 1 divides by ~1e-12. `results_v0/R6a/metrics.jsonl`:
```
step 1 ema_z 12944221496.58   step 2 -5.73   step 3 -23.64
max |ema_z| 1.29e10 ; |ema_z| > 2.5 (the configured ema_z_threshold) on 21 of 300 steps
```
The loss is falling smoothly, so these are false alarms by construction. Fix: return null until `step >= N` (e.g. 10) and ideally bias-correct the EMA variance (P0: cheap, and G8 reads it).
- `gpu_alloc_gb` = `torch.cuda.memory_allocated()` read after `zero_grad(set_to_none=True)` (`:348`), i.e. a post-step snapshot. `results_v0/README.md` columns are headed "peak alloc GB" (R6a values 0.711, 0.71 constant). Not a peak. Use `max_memory_allocated()` + `reset_peak_memory_stats()` per step, or rename the column (P0 for wording, P1 for semantics; G6 reads memory).
- `update_to_weight_ratio` is one global number over all params (`train.py:359-362`); S15 §16 dashboard asks per-layer. `max_attn_logit` is max over layers of the last micro-batch only (documented).
- `dead_experts` is a sum over layers, `top_expert_share` a max over layers (`:366-375`). Per-layer `load` is computed (`model.py:241`) and dropped. S15 requires five routing metrics per layer and MaxVio (S15:315,642,694,714-715). Not present: MaxVio, router entropy, top-k intervention rate, selected top-k mass, per-layer view (P1).

### 2.8 Held-out loss / BPB math
- `held_out()` (`train.py:259-272`): chunked CE mean over equal-length rows, same value on all ranks. Correct per token.
- **BPB mixes two weightings (bug, ~10%).** `data.py:31`: `val_bytes_per_token = sum(val_bytes)/sum(val_tokens)` over all sources = token-weighted (code has 128k of ~330k val tokens). But `val_batch()` takes 4 windows per source (`data.py:59-66`), so the CE is an equal-weight source mean. Measured from `manifest.json`:
```
token-weighted bytes/token 5.650 ; equal-weight mean over sources 6.213
{fineweb_edu 4.88, dclm 4.40, finemath 3.26, code 3.95, hin 8.05, ben 7.48, tam 7.71, tel 6.18, kan 7.05, mal 7.67, mar 7.08, guj 6.86}
bpb used 1.9037 ; bpb with equal-weight bytes/token 1.7310
```
BPB is the cross-tokenizer comparison metric the plan will use, so a 10% bias matters. Fix: compute bytes per window (decode the actual val windows, or use per-source bytes/token) and divide the summed nats by summed bytes of the evaluated windows; also log `held_out_loss_<source>` per source (Indic vs code is the point of this tokenizer) (P1). EOS tokens add tokens with zero bytes (`prepare_data.py:44-51`), a small extra bias.
- Held-out set = 48 windows x 256 = ~12k tokens: noisy (+-0.02?). [INFERENCE] Increase `VAL_WINDOWS_PER_SOURCE` for plan arms.
- Gate weakness: `compare_runs.py:25` hardcodes `ln(131072)`, and "loss falls" is `last10% < first10%`. A dead-fast unigram fit passes it. Use config vocab and also require a held-out improvement (P2).
- Kaggle evidence: step-0 CE 11.8437 vs ln V 11.7835 (std 0.02 untied head, plausible); final held-out dense 7.444 vs MoE 7.455 (BPB 1.901 / 1.904). At 300 steps and 4 experts, MoE == dense within noise: this is a harness check, not an architecture result. The README "peak alloc GB" label and "near-tied router flip" explanation are the two claims I could not confirm.

### 2.9 check_config blind spots (E8)
Edited copies under `work/review/runs/cc/` all return `0 errors, 15 open` except one:
```
junk.yaml  (lr comment replaced by "# x")        -> 0 errors      (any ' #' passes as a "source")
del.yaml   (optimizer.grad_clip line deleted)    -> 0 errors      (missing required key never reported; train.py KeyErrors later)
unk.yaml   (seed typo "sead:")                   -> 0 errors      (unknown key silently accepted, seed missing)
typo.yaml  (open_keys has model.n_layerz)        -> 1 error       (caught only indirectly: n_layers null and not open)
w.yaml     (data weights sum 1.0000001)          -> 0 errors      (check allows 1e-6; numpy rng.choice rejects at 1e-7)
```
`numpy`: `Generator.choice(p=[0.5, 0.5+1e-7])` -> `ValueError: Probabilities do not sum to 1`. So a stage that passes the checker can crash `data.py:45` at the first step. Also: 34 leaves in `config_v0.yaml` are never read by the harness (list from `work/review/unused_keys.py`):
```
general.{project,run,deterministic} tokenizer.revision embedding.{type,ke2_output_shape} model.{ffn,norm,norm_position,position,bias,dropout,tie_embeddings}
moe.{router_dtype,balancing,dropless} head.mtp_heads optimizer.{matrices_2d,other_params,no_decay,activation_grad_clip} schedule.type precision
parallelism.{tp_mode,expert_parallel_size,context_parallel_size} safety.{on_nan,checkpoint_after_nan,spike_sigma,ema_z_threshold}
logging.{format,scalars_every,metrics}
```
Several are "decoration": `general.deterministic: false` would still run deterministically; `optimizer.no_decay` is ignored (hardcoded in `optim.py:20-34`); `logging.metrics` does not select what is logged; `safety.spike_sigma/ema_z_threshold` are consumed only by G8. A reader of config_v0 (G7, TSAI) will assume these are knobs. Either wire them, or have `check_config` fail on a key the harness never reads and mark the rest `# fixed in v0 (not read)` (P1).

## 3. Config review vs SmolLM3 stage configs and nanotron schema

Refs: SmolLM3 `text/pretraining/smollm3/stage1_8T.yaml` (local copy `work/review/ref/stage1_8T.yaml`), `stage3_9T_11T.yaml`; nanotron `src/nanotron/config/config.py`, `models_config.py`, `examples/config_tiny_llama.yaml`, `examples/moe`, `scaling/parametrization.py` (local copies in `work/review/ref/nanotron/`).

| Setting | Reference does | We do | Recommendation |
|---|---|---|---|
| init method | `init_method.std: 0.02` (stage1:166-167); nanotron `scaling_method: NUM_LAYERS` default (`models_config.py:15`): row-linear (`o_proj`, `down_proj`) std divided by `sqrt(2*num_layers)` (`parametrization.py:~96-108`); grouped MLP `merged_down_proj` likewise | `init_std: 0.02` for every Linear incl. `wo`, `w_down`, experts (`model.py:169-176`), no depth scaling | **Experiment arm** `init_scaling: none | num_layers` (affects step-0 CE and early stability); keep ours for v0 and say so |
| rope theta / scaling | `rope_theta: 50000`, `rope_scaling: null`, `rope_interleaved: false` (stage1:192-194) | 50000, rotate-half, none | **Keep** (same). Context-extension (long_context yamls) is out of week-1 scope |
| NoPE layers | `no_rope_layer: 4` = every 4th layer skips RoPE (stage1:201; `models_qwen.py:272`) | none | **P2 arm** `nope_every: 0 | 4` (SmolLM3's long-context choice) |
| document masking / intra-doc attention | `_use_doc_masking: true`, `return_positions: true` (stage1:173,141) | none; windows cross EOS (`data.py:46-47`) | **P2 arm** `doc_mask: off | on`; needs `position_ids`/varlen SDPA. Note determinism cost on math SDPA |
| LR schedule | linear warmup 2000 steps (0.04%), `lr_decay_style: linear`, decay 522000 steps = 11% from step 4198001, `min_decay_lr: 0` (stage1:205-212) | WSD, warmup 2%, decay 10%, min 0 | **Keep**, but fix the anneal/decay alignment (§2.4) and add `lr_warmup_steps` as an alternative to `warmup_frac` |
| peak LR | `2e-4` (3B, AdamW) | `3e-4` (S11 §2) for both Muon and AdamW | **Keep, make LR an arm** (the W2 LR sweep is a rotating-member task) |
| Adam betas / eps | `0.9 / 0.95`, eps `1e-8` (stage1:214-216); nanotron MoE example also 0.95 | `0.9 / 0.999` (S11: "almost always 0.999", s11.md:217), eps 1e-8 | **Experiment arm**: beta2 0.999 vs 0.95. Short runs and large batch favour 0.95 |
| weight decay | `0.1`; `weight_decay_exclude_named_params: [".*token_embedding.*"]` only (stage1:219-221). Norms ARE decayed | 0.1, no WD on norms/biases/embedding/router (`optim.py:20-34`) | **Keep ours** (S15 §18, OLMo2). Note that it differs. `optimizer.no_decay` is not read (wire it) |
| grad clip | `clip_grad: 1.0` | 1.0 | match |
| grad accumulation | `batch_accumulation_per_replica: 1`, `micro_batch_size: 3`, `accumulate_grad_in_fp32: true` (stage1:203,250,253) | `grad_accum` derived = `G/(dp*micro)`; fp32 everywhere | **Keep**; `accumulate_grad_in_fp32` becomes relevant when bf16 arrives (P2) |
| precision | `dtype: bfloat16` | fp32 (T4 has no bf16) | **Keep** for Kaggle; add `precision: bf16` path before H100 (G6 asks) |
| z-loss | `z_loss_enabled: false`, coef 1e-5 (stage1:199-200); nanotron MoE yaml `0.0001`, disabled | enabled, 1e-4 (S15:213) | **Keep**; arm `z_loss_coef: 0 | 1e-4` |
| data stage format | list of `{name, start_training_step, data: {dataset_folder, dataset_weights, ...}}`; separate data `seed` from `general.seed` (stage1:148,157) | `data_stages` with `start_frac` + weights; one seed | **Keep** `start_frac`; **adopt a separate `data.seed`** (week-3 "repeat with a new seed" task needs to change data order without changing init) |
| checkpoint / resume flags | `checkpoint_interval, checkpoints_path, save_initial_state, save_final_state, resume_checkpoint_path, load_lr_scheduler, load_optimizer` (stage1:1-9; nanotron `CheckpointsArgs`) | `every_steps, strict_load, keep_last`; resume is a CLI flag; final step always saved | **Adopt** `save_final_state`, `resume_path`, `load_optimizer`, `load_lr_scheduler` as keys (G7 asked for config_v0 on 27 Oct and checkpoint design is theirs) |
| tokenizer max length | `tokenizer_max_length: 4096` = `sequence_length` | none; `tokens.seq_len` open | **Adopt** `tokenizer.max_length: =tokens.seq_len` derived (not an independent knob) |
| GQA ratio | 16 heads / 4 kv = 4:1 (stage1:186,188); nanotron tiny: 4 / 4 | t4_base 4 / 2 = 2:1; sizes open | sizes are TSAI's; note 4:1 as the SmolLM3 default for the TSAI table |
| tie embeddings | `tie_word_embeddings: true` | `false`, KE has no V x d table | **Keep** (KE) |
| QK-norm | none | yes (S15) | **Keep** |
| norm eps | 1e-6 | 1e-6 | match |
| parallelism block | `dp, tp, pp, pp_engine, tp_mode, tp_linear_async_communication, tp_recompute_allgather, zero_stage, recompute_layer, expert_parallel_size, context_parallel_size` | same names; `tp_mode`, `expert_parallel_size`, `context_parallel_size` are never read | Keep names (good for later), mark unread ones |
| ZeRO | SmolLM3 `zero_stage: 0`, nanotron default 0 | `zero_stage: 0|1` | match |
| seed | `general.seed: 6` | 1234 (brief) | match brief |
| tied vocab padding | `make_vocab_size_divisible_by: 1` | vocab 131072 | match |

## 4. MoE review vs nanotron MoE example and S14

Refs: `work/review/ref/nanotron/nn_moe.py`, `config_qwen_with_moe.yaml`, `ex_moe.py`; S14 §12-14, §20.

| Aspect | nanotron (Qwen2-MoE) | Ours (`model.py:81-130`) | Verdict |
|---|---|---|---|
| router | Linear `W in R^{E x d}`, fp32, **softmax** over all E then top-k (`nn_moe.py:52-59`) | Linear, fp32 **sigmoid**, top-k on `score + bias`, weights = chosen scores renormalised (`:108-111`) | Ours = DeepSeek-V3/GLM style (S14:382). Keep. Note: no routed scaling factor (DS-V3 2.5); with shared=0 and renorm the weights sum to 1 so it is a no-op |
| load balancing | none in the example (no aux loss, no bias) | aux-free bias (sign update, speed 1e-3) + sequence-wise aux 1e-4 | Ours matches DS-V3 §2.1.2-2.1.3 and S14 §13. Good. The bias speed 1e-3 over 300 steps moves a bias by <=0.3; fine at harness scale |
| balancing scope | n/a | bias load is over the whole global batch (all micro-batches, dp all-reduced) | Good (S14 §14: balance over the whole batch). The aux loss is per-sequence by design (DS-V3 uses a small one) |
| granularity | `moe_intermediate_size` per expert; E=8, k=1 | `d_ff` per expert; t4_moe E=4, k=2, `d_ff 384` so `k*d_ff = 768` = dense | Matched active FFN params, fine for the harness. Real arms: E in {8,16,32,64} with `d_ff` shrunk to hold active params (S14 granularity, Kimi K2 48 experts per active) |
| shared expert | `enable_shared_expert` + sigmoid gate (`nn_moe.py:193-197`) | `shared_experts: 0` supported (`MLP` of `n_shared*d_ff`, no gate) | Config key exists and the path exists; **never exercised** by any run/test. Run one smoke before claiming support; P2 arm `shared: 0|1` |
| capacity / dropless | grouped GEMM, no capacity (dropless) via `ops.permute` | Python loop, dropless (S14 §11: modern models drop none) | Good. Perf only (loop + `nonzero` sync); P2: grouped GEMM or sort-based dispatch |
| expert init | `merged_down_proj` std/`sqrt(2L)`; gate/up std; router std (parametrization.py) | all std 0.02, router 0.02 | See §3 init arm. Router std 0.02 with sigmoid gives scores ~0.5 +- small, so initial routing is near-uniform: OK |
| first-k dense layers | n/a | every layer is MoE | S14:220-222 and the S15 "silent routing fault in layers 0-1" (s15.md:310-322) say first layers route poorly. **P2 arm** `moe.first_dense_layers: 0 | 1` |
| TP / EP | EP via `expert_parallel_size` | rejected (`parallel.py:61-62`, `check_config.py:96-99`) | Fine for v0; G7/G6 should know |
| aux in PP / grad-accum | n/a | `_AddAux` autograd seed, scale 1/ga or 1.0 (PP) | Verified equal to fp32 noise (E6) |
| what to log | n/a | `dead_experts` (sum), `top_expert_share` (max), `moe_aux` | **Missing vs S15**: per-layer MaxVio, per-layer dead experts, router entropy, top-k intervention rate (share of picks where bias changed the choice), selected top-k mass. Cheap to add in `moe_metrics`: `load` is already per layer; intervention = `topk(scores)` vs `topk(scores+bias)`; entropy from `scores/sum` |

## 5. Improvements

### P0: before Tue 27 Oct 23:59 IST (cheap, visible to the lead / G6 / G8 / G7)
1. **G6 small config is not "one file beside config_v0".** `configs/small/t4_moe.yaml:4` `extends: t4_base.yaml` -> `../config_v0.yaml`; 12 files in a subfolder. Fix: add `configs/config_small.yaml` (flattened, self-contained, tiny MoE, passes `check_config`, 1 open = ke2) beside `config_v0.yaml`; keep layout files but say in the README that `config_small.yaml` is the G6 file. Better: replace the 8 layout files by `--set` lines in the README. Evidence: lead G5.md:452; `ls configs/small` = 12 files.
2. **`ema_z` false alarms in shipped metrics.** `train.py:351,355-358`; `results_v0/R6a/metrics.jsonl` step 1 `ema_z=1.29e10`; 21/300 steps above the 2.5 threshold in a smooth run. Fix: `ema_z=None` until `step >= ema_warmup` (10) or `ema_var` bias-corrected. G8 alarms consume this tomorrow. Regenerate/trim `R6a/metrics.jsonl` (or mark the note).
3. **Source-policy ambiguity (A2).** Post one question to the lead now, and tag comment classes. Reword `config_v0.yaml:135` "V4 dashboard list". Fix the wrong SmolLM3 citation for the stage boundaries (`config_v0.yaml:112,115,118`: actual 0.731 and 0.889, not 8/11 = 0.727 and 9/11 = 0.818) or relabel as ours. Name a source for the 11 leaves that have none.
4. **README column "peak alloc GB" is not peak** (`train.py:348` `memory_allocated`). Rename the column or fix (G6 reads this).
5. **PR body hygiene.** State that the Axiom email is only in the PR body because the pr-guard flags emails in patches, and which file is the G6 file. Make sure the working-tree fix in `model.py:243` (aux None) is in the PR.

### P1: before the Sat 31 Oct gate
1. **BPB weighting bias ~10%** (`data.py:31` vs `:59-66`, `train.py:376-378`): 1.9037 vs 1.7310. Use bytes of the evaluated windows (or per-source bpt), add per-source `held_out_loss_<src>`. Evidence in §2.8.
2. **Spike/EMA state lost on resume** (`train.py:104,274-275`): after resume `spike_score` null for `spike_window` steps, `ema_z` spurious (E5). Save `hist/ema/ema_var` in the checkpoint. G8 alarms fire on every restart otherwise.
3. **PP step-0 MoE load overcount** (`train.py:293-297`, `model.py:204`): 1536 vs 1024 slots (E3). Reset `moe.load` after the first pp step, or count only under an explicit flag. Pollutes step-0 bias and metrics.
4. **`check_config` blind spots** (`check_config.py:50`): any ` #` passes (E8 junk), deleted required keys not reported, unknown keys accepted, weight-sum tolerance 1e-6 vs numpy's tighter check (`data.py:45` crash on 1+1e-7). Fix: (a) require a recognised source tag (`[KE]|[paper]|[run]|[lesson ..]|[ref ..]|[admin]|[default]`), (b) a required-keys list or compare against the leaf set of `config_v0.yaml`, (c) error on unknown keys, (d) normalise weights or tolerance 1e-9, (e) reword outputs to "no value: key" / "no source: key" (lead's sentence), (f) fail on a config key the harness never reads (34 today, §2.9).
5. **Wire or mark the 34 unread keys** (`work/review/unused_keys.py`). Especially `optimizer.no_decay`, `general.deterministic`, `logging.metrics`, `moe.router_dtype`, `moe.balancing`, `moe.dropless`, `head.mtp_heads`, `precision`.
6. **MoE health metrics per layer** (S15:315,642,694,714-715): MaxVio, per-layer dead experts, router entropy, top-k intervention rate, selected top-k mass in `model.moe_metrics`/`train.py:366-375`; the per-layer `load` is already computed and discarded.
7. **Checkpoint manifest for G7**: write `manifest.json` next to `rank*.pt` (world/pp/dp/tp, config sha256, torch version, `frozen_sha256`), verify it in `load_ckpt`; atomic write via `.tmp` + `os.replace`. Add `checkpoint.{save_final_state,resume_path,load_optimizer,load_lr_scheduler}` keys to config_v0 (nanotron names).
8. **`plan_v0.md` missing** (`teams/05-architecture/experiments/`). Arms from §3/§4 (P2) with metric, pass bar, GPU hours.
9. **Exactness check ergonomics.** `compare_runs --exact` on a resumed run fails at step 0 (the resumed `losses.tsv` starts at the resume step; I saw `FAIL first difference at step 0` for ZeRO and TP resumes whose aligned steps are identical). Add `--from-step` or align by step key when lengths differ; README example for the resume test.
10. **Anneal/decay alignment** (`config_v0.yaml:118` anneal 0.82 vs decay start 0.90): set both to SmolLM3's 0.889 / 0.111 or document as ours.

### P2: plan_v0 experiment arms and later hardening
| Arm | Metric | Setting | Source |
|---|---|---|---|
| init scaling | step-0 CE, loss @N steps | `none | num_layers` (std/sqrt(2L) on `wo`, `w_down`, experts) | nanotron `parametrization.py` |
| Adam beta2 | held-out BPB | 0.999 vs 0.95 | SmolLM3 stage1:215 vs S11 |
| LR / warmup | held-out BPB, grad norm | peak LR sweep; `warmup_frac` vs absolute steps (2000 in SmolLM3) | SmolLM3 stage1:206-210 |
| NoPE every 4th layer | BPB at seq_len | `nope_every: 0|4` | SmolLM3 `no_rope_layer: 4` |
| doc masking | held-out BPB | `doc_mask: off|on` | SmolLM3 `_use_doc_masking` |
| z-loss | max logit, BPB | `0 | 1e-4` | S15; SmolLM3 off |
| WD exclusions | BPB | norms decayed (SmolLM3) vs not (ours) | stage1:219-221 |
| MoE granularity | BPB at matched active params | E in {4, 8, 16, 32}, k in {1, 2, 4} | S14 |
| shared expert | BPB, dead experts | 0 vs 1 (path exists, untested) | nanotron `enable_shared_expert` |
| first-k dense | MaxVio layers 0-1 | 0 vs 1 dense layer | S14:220, S15:310-322 |
| balancing | MaxVio, BPB | bias only (`seq_aux_loss_coef: 0`, now runnable) vs bias+aux vs aux only | DS-V3 |
| bias speed | MaxVio | 1e-3 vs 1e-4 vs 5e-3 | S14 |
| Muon NS dtype | BPB | fp32 vs bf16 (torch default) | torch.optim.Muon |
| tokenizer-aware data | per-source BPB | stage weights (all weights are "hypothesis" per S5) | S5 |
Hardening: `broadcast_buffers=False` in DDP; loss-parallel (sharded) cross-entropy under TP instead of gathering 131k logits (`parallel.py:69`, nanotron `sharded_cross_entropy`); grouped GEMM / sorted dispatch to remove the per-expert `nonzero` sync; WSD last-step no-op (`(step-(T-D)+1)/(D+1)`); verify `frozen_sha256` on resume; gate that reads vocab from config and checks held-out; bf16 + `accumulate_grad_in_fp32` path; larger val set for plan arms; run the shared-expert path once.

## 6. Reading of the Kaggle evidence (`results_v0/`)
- Byte-identical pairs (R1-R3, R6, R7) and the resumed R5r are genuine proof of run-to-run determinism on that Kaggle session. They say nothing about cross-session, cross-GPU-model or cross-torch determinism.
- "Differs by at most 2e-7 rel (fp32 reduction order)" for dense layouts: consistent with the code. The MoE 9.3e-5 / 8.4e-5 cross-layout gap is attributed to a tied router score; unproven, and not caused by the PP overcount (E4). To prove it, dump `topi` for step 292 on both layouts.
- MoE vs dense at 300 steps: held-out 7.455 vs 7.444, BPB 1.904 vs 1.901. No quality claim is supported; the table is a harness check.
- Throughput: tp2 739 ms vs dp2 300 ms is expected (full-logits gather over PCIe + full-matrix Muon all-gather). PP2 442 ms. Nothing wrong, but G6 should not read tp2 as representative.
- "Peak alloc GB" is a post-step snapshot (§2.7). `ema_z` column in `R6a/metrics.jsonl` has the 1.29e10 artefact (§2.7).
- R5: `status.txt` shows `R5 exit=1` while README says the worker exits 3. That is torchrun's mapping, not an inconsistency.
- 9.0M train tokens vs ~1.2M consumed in 300 steps (16 x 256 x 300): no epoch repeat; val = first 20 docs, train = next 500: no overlap.

## 7. Experiments run (all under `work/review/`)
| ID | Question | Result |
|---|---|---|
| E1 | `moe.seq_aux_loss_coef=0` | first run crashed (exit 1, empty `metrics.jsonl`), fixed in tree at 22:04:22; reruns exit 0, `moe_aux 0.0` |
| E3 | MoE load per step, PP vs 1 proc | PP step0 1536 vs 1024 (extra micro-batch), later steps 1024 |
| E4 | step-0/1 `expert_bias` PP vs 1 proc (speed 0.05) | identical `[-0.05,-0.05,0.05,0.05]` then zeros |
| E5 | resume metrics state | `spike_score` null x10, `ema_z` 3.5e8 at step 11 |
| E6 | aux loss grad-accum / PP invariance (coef 0.5) | equal to <=1e-6 |
| E7 | resume ZeRO-1 dp2, TP2 | byte-identical on aligned steps |
| E8 | check_config mutations | junk comment, deleted key, unknown key, weight 1+1e-7 all pass |
| E9 | SmolLM3 fractions | stage2 0.7309, stage3/decay start 0.8894, decay 0.1106, warmup 0.00042 |
| E10 | unread config keys | 34 |
| numpy | `choice(p sum 1+1e-7)` | `ValueError: Probabilities do not sum to 1` |
Not run (stated, not guessed): CUDA determinism of `index_add_` beyond the docs list and the Kaggle runs; Muon vs `torch.optim.Muon` numeric diff; shared-expert path.
