Name: dharshan
Axiom email: dharshanthaigu@gmail.com
Pack: ind-ta-fw2

# G5 tiny harness (team 05-architecture)

## 2. What it delivers and to whom

| File | For | What |
|---|---|---|
| `configs/config_v0.yaml` | G7, whole team | every setting with a source tag; sizes, KE2 shape, data weights are `open` |
| `configs/config_small.yaml` | G6 | self-contained tiny MoE (no inheritance); only the KE2 shape is `open` |
| `harness/check_config.py` | everyone | checks values and sources, exit code 1 on errors |
| `harness/config.py, model.py, data.py, optim.py, train.py` | G6, G8 | plain training loop, metric names fixed |
| `harness/seed_test.py` | lead, G6 | gate tests 2 and 3, a NaN-stop check and a resume check |
| `harness/compare_runs.py` | everyone | compares finished runs: `--exact`, `--rtol`, `--gate` |
| `experiments/plan_v0.md` | TSAI (approves Sat 31 Oct) | table of arms |
| `evidence/` | all | only files from real runs (empty until then) |

G4 gives the KE2 output shape. G8 gets the metric names (section 8) and later delivers the logger and spike/EMA alarms (Thu 29 Oct).

## 3. Quick start

Where the files live (team repo `udisinghania/ERA_V5_Architecture`, one folder per student):

```
Dharshanthaigu/
  configs/      config_v0.yaml, config_small.yaml
  harness/      README.md, check_config.py, config.py, model.py, data.py, optim.py, train.py, seed_test.py, compare_runs.py
  experiments/  plan_v0.md
  evidence/     only files from real runs (empty until then)
```

Run everything from the folder that holds `configs/` and `harness/` (`Dharshanthaigu/` in the team repo). On Windows use `py` (Python 3.11).

```
py -m pip install pyyaml numpy torch pandas
py -m pip install git+https://github.com/theschoolofai/kronecker-embeddings
py harness\check_config.py configs\config_small.yaml
py harness\train.py --config configs\config_small.yaml --out runs\a
py harness\train.py --config configs\config_small.yaml --out runs\b
py harness\compare_runs.py runs\a runs\b --exact
py harness\seed_test.py --evidence
```

Expected output (the first two are what a yaml-only test showed; the rest is **not run yet**, fill in after your own run):

```
check_config small : 73 ok, 1 open, 0 errors     source tags: KE=8, admin=7, ours=58, open=1
check_config v0    : 53 ok, 21 open, 0 errors    source tags: KE=8, admin=7, ours=38, open=21
train.py           : prints "device=... params=... active=..." then a line every 10 steps   (not run)
seed_test.py       : ten PASS lines and exit code 0                                          (not run)
```

Note: both configs only pass after the `<FILL>` header lines are replaced with the real Axiom name and email.

The tokenizer is downloaded only when `data.source` is `parquet`; the synthetic source (used by `config_small.yaml` and the seed test) needs no download.

## 4. Config rules

- Every line is `key: value  # [tag] reason`. Tags: `KE` (Kronecker-embeddings code), `paper` (arXiv 2605.29459), `run` (our tiny run; must start with a path of an existing file in `evidence/`), `admin` (lead/admin decision), `ours` (our own default with a one-line reason), `open`.
- `open` values need `# [open] who, when`. Open keys in v0: KE2 shape (G4, Tue 27 Oct); model sizes, experts, top_k, sequence/batch/steps (TSAI, Sat 31 Oct); learning rates, weight decay, betas, warm-up/decay (arms OP-2 to OP-4); data source, paths, weights (Data A/B teams, date not given).
- The key list is `SCHEMA` in `config.py`. A key outside it is "unknown"; a key code reads but the file lacks is "missing".
- Override at run time: `--set a.b=value`, repeatable, e.g. `--set attention.window=8 --set moe.router_sharing=all_layers`. An `open` key that is still open when read makes the run stop.
- No V4 yaml, V4 code or V4 learning rates are used.

## 5. check_config output and exit code

Lines start with `no value`, `no source`, `unknown`, `missing`, `bad value`, `header` or `open (who, when)`, then a summary count and a count per source tag. `--list` also prints every good setting. Checks include `d_model % n_heads`, `n_heads % n_kv_heads`, even head size, `top_k <= num_experts`, `norm_position` in pre|post, `router_sharing` in none|all_layers, window and global_every >= 0. Exit code 1 if there is any error, else 0.

## 6. Model design

Kronecker input embedding (trainable projection, no weight decay) -> N blocks -> final norm -> untied output head.
Attention: grouped-query, optional QK-norm, RoPE; `attention.window` (0 = full) and `attention.global_every`.
FFN: SwiGLU experts; MoE with sigmoid router in fp32, top-k, dropless, aux-free bias balancing plus a small sequence aux loss, optional shared expert; `moe.router_sharing` none|all_layers (one router matrix, per-layer bias kept).
Norm: RMSNorm, `model.norm_position` pre (default) | post.
Heads: one MTP head (t+2), z-loss.
Optimizer: Muon on matrices, AdamW on the rest; no weight decay on norms, biases, router. Schedule: warm-up / flat / linear decay.
Tokenizer: BrahmicTokenizer-131K, pinned by commit and sha256 in the config.

## 7. Determinism measures

Seed python, numpy and torch from `general.seed`; `torch.use_deterministic_algorithms(True)`; cuDNN deterministic, benchmark off; `CUBLAS_WORKSPACE_CONFIG`; fixed thread count (`general.num_threads`); math SDPA backend; data batch k depends only on `(seed, step, micro)`, so order is the same after a resume; losses written as exact hex floats to `losses.txt`.

## 8. Metrics

Written to `metrics.csv`, one row per optimiser step unless noted.

| Name | Unit | When |
|---|---|---|
| tokens_per_second | tokens/s | every step |
| loss_ntp | nats/token | every step |
| loss_mtp | nats/token | every step (empty if MTP off) |
| loss | nats | every step (ntp + z + aux + weighted mtp) |
| lr | factor x base | every step (`lr_muon` column too) |
| gpu_alloc_gb | GB, peak allocated in the step | every step; empty on CPU |
| grad_norm | norm before clipping | every step |
| z_loss | nats | every step |
| held_out_loss | nats/token, equal weight per source | every `logging.eval_every` steps and the last |
| dead_experts | count, summed over layers | every step |
| top_expert_share | fraction of routed tokens | every step (max over layers) |
| max_vio | (max load - mean) / mean | every step (max over layers) |
| per-layer MoE stats | same three | `moe_layers.csv`, every `eval_every` steps |

Definitions to confirm with G8: `gpu_alloc_gb` is the peak allocated during the step (`torch.cuda.max_memory_allocated`, null/empty on CPU); another team member reads the allocation after the step instead. `loss_mtp` is the loss of one extra block that predicts token t+2; another team member uses a second linear head. Units and cadence are open until G8 confirms.

No spike/EMA alarms: G8 delivers them.

## 9. Gate tests and evidence layout

1. `check_config.py` gives 0 errors on both configs.
2. `seed_test.py`: seed 1234 twice gives identical `losses.txt` with one row per step; seed 1235 differs at every step; no NaN; loss falls; step-0 loss is within 0.5 of ln(131072).
3. `window=0` and `router_sharing=none` equal the default run exactly; each switch on twice still matches.
4. Extra checks: an injected NaN stops the run with exit code 2 and writes `NAN_REPORT.json`; a run stopped half-way and resumed from `checkpoint.pt` gives the same `losses.txt` as an uninterrupted run (checkpoints are written to a temporary file and renamed, and are never copied into `evidence/`).

Evidence: `evidence/seed_test_<time>/` with `result.txt` and the run folders, created by `seed_test.py --evidence`. Cite as `[run] evidence/<folder>/result.txt`. Never add files by hand.

**Status: gate 1 tested on yaml only; gates 2 and 3 not run.**

## 10. The five topic decisions

- **Pre-norm**: real switch `model.norm_position`, pre default, one post arm (AR-1).
- **RoBERTa embedding model**: comparison only (table below).
- **Sliding window**: config keys, mask path, one arm (AR-3); source tag `[ours]` until a source is verified.
- **Shared router**: config key and code switch, one arm (MO-4).
- **PyTorch Lightning**: not used. Exact metric names, the bit-for-bit seed test and G8's custom logger fit a plain loop.

| | RoBERTa | Ours |
|---|---|---|
| Direction | bidirectional encoder | causal decoder |
| Norm placement | post-LN | pre-norm (post switch) |
| Positions | learned absolute | RoPE |
| Activation | GELU | SwiGLU |
| Objective | masked LM | next-token plus MTP loss |

## 11. Limits

Not done or not run:
- Gate tests 2 and 3 and every training run: **not run**. No `evidence/` exists.
- `data.source: parquet` and tokenizer download: **not run**.
- Atomic checkpoint save and resume: written but **not run**; the check is T5 in `seed_test.py`.
- No dense-embedding path, so arm KE-1 is blocked (also needs the KE2 shape from G4).
- No spike/EMA alarms (G8).
- Only float32 is implemented.

The harness code was compiled (syntax) only. PyTorch could not be installed where it was written, so no model code has been executed.

Unverified:
- Tokenizer sha256, vocab size 131072, eos id 36 and the pinned commit are marked "measured by me" in the configs: re-check before pushing.
- `KroneckerEmbedding` and `build_byte_buffer` arguments match a teammate's vendored copy of the KE code (commit d07cb54, said to be unmodified). I have not compared that copy with the GitHub repo.
- Tokenizer values are the same in several teammates' configs, which is agreement, not a check against the source.
- A teammate's review says the lab's pr-guard may flag email-shaped text in a patch. The brief asks for the Axiom email in every file header; ask the lead whether it belongs in the files or only in the PR body. This is not confirmed.
- Newton-Schulz coefficients in `optim.py` are from memory.
- Muon and the "loss-free balancing" idea are tagged `[ours]` ("to verify"); the Muon-on-2D choice comes from teammates, lead to confirm.
- Parameter-to-optimiser grouping is my reading of the lead's rule, to confirm.
