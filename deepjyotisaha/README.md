# G5 Architecture: config v0, the config check and the tiny-model harness

**Author:** Deepjyoti Saha (G5 Architecture team, ERA V5 capstone) · **Built:** 25–26 Oct 2026 · **Status:**
scratch version for the G5 comparison call. The team compares each member's version, then splits the work.

This folder is self-contained. With Python and the input text described in [`data/README.md`](data/README.md),
everything below can be rerun from here, and every result quoted below has its raw output in
[`evidence/`](evidence/).

---

## Contents

1. [What this is](#1-what-this-is)
2. [Results at a glance](#2-results-at-a-glance)
3. [Quick start](#3-quick-start)
4. [Folder structure](#4-folder-structure)
5. [Design](#5-design): configs, the check, the data, the model, the loss, determinism, metrics
6. [How one training step flows through the code](#6-how-one-training-step-flows-through-the-code)
7. [Evidence: what each file proves and how to check it](#7-evidence)
8. [Findings](#8-findings)
9. [What this is not yet](#9-what-this-is-not-yet)
10. [Sources and provenance](#10-sources-and-provenance)

---

## 1. What this is

ERA V5 is the large model the course trains from 28 Nov. G5 owns its configuration. In week 1, G5 delivers three
things, and this folder contains all three:

| # | Deliverable | File | Done when |
|---|---|---|---|
| 1 | **Config v0**: every setting of the V5 model and run, each with a default and the reason for it | [`configs/config_v0.yaml`](configs/config_v0.yaml) | every setting has a value (or `open`) and a source |
| 2 | **The config check**: lists every setting with no value or no source | [`harness/check_config.py`](harness/check_config.py) | 0 errors on 31 Oct |
| 3 | **The harness**: trains a tiny version of the model from a config and logs the agreed metrics | [`harness/train.py`](harness/train.py) and helpers | two runs with seed 1234 give the same loss at every step |

The tiny model is a small copy of the real design, used as a test bench: Kronecker input embedding,
grouped-query attention with rotary positions, mixture-of-experts layers, and an extra multi-token-prediction
head, trained on mixed English and Indic text. It is **not** the V5 model; its sizes are laptop sizes.

Two decisions from the G5 call on 25 Oct shaped it: always test with the Kronecker embedding, and train on
Indic and English text together from the first step.

---

## 2. Results at a glance

All on a laptop CPU, seed 1234 unless stated.

| What was checked | Result | Evidence |
|---|---|---|
| Config check on config v0 | **88 ok, 16 waiting** (`open`, with a source), **0 errors** | rerun `python harness/check_config.py` |
| Seed test, `tiny_moe`: two fresh processes, 40 steps | **identical `losses.txt` at every step** (bit for bit); next-token loss 8.5671 → 6.6143 (mean of first / last 8 steps); no NaN | [`evidence/seed_test/tiny_moe/`](evidence/seed_test/tiny_moe/) |
| Seed test, `tiny_dense` | identical at every step; 8.5744 → 6.6105; no NaN | [`evidence/seed_test/tiny_dense/`](evidence/seed_test/tiny_dense/) |
| Control: `tiny_moe` with seed 1235 against seed 1234 | **all 40 lines differ**, so the test can tell runs apart | [`evidence/seed_test/tiny_moe/control_seed1235/`](evidence/seed_test/tiny_moe/control_seed1235/) |
| 200 steps, `tiny_moe` (5,407,744 parameters, 3,343,360 active per token) | held-out bits per byte 2.683 → **1.559**; expert load 0.52–1.39 × an even share at the last step | [`evidence/runs/tiny_moe_seed1234/`](evidence/runs/tiny_moe_seed1234/) |
| 200 steps, `tiny_dense` (3,337,216 parameters) | held-out bits per byte → **1.567** | [`evidence/runs/tiny_dense_seed1234/`](evidence/runs/tiny_dense_seed1234/) |

The 200-step runs show that the harness trains both kinds of model correctly and repeatably. They are far too
short to say anything about dense against MoE.

---

## 3. Quick start

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

Put the four input files in `data/` (see [`data/README.md`](data/README.md)), then, from this folder:

```bash
python harness/check_config.py                                       # config v0: expect 0 errors, 16 waiting
python harness/build_sample.py                                       # once: builds work/sample/
python harness/seed_test.py --config configs/tiny/tiny_moe.yaml      # the gate test, about a minute
python harness/train.py --config configs/tiny/tiny_moe.yaml          # 200 steps, about 2 minutes on a laptop CPU
```

| Command | Useful options |
|---|---|
| `check_config.py [config]` | `--list` prints every setting with its value, its source and the file it comes from |
| `train.py` | `--config`, `--steps N` (also caps the warm-up at N/10 steps), `--seed`, `--device cuda`, `--out DIR` |
| `seed_test.py` | `--config`, `--steps` (default 40), `--seed` (default 1234), `--out DIR` |

Everything the harness writes goes to `work/` (git-ignored): `work/sample/`, `work/runs/<config>_seed<seed>/`
and `work/seed_test/<config>/`.

---

## 4. Folder structure

```
deepjyotisaha/
├── README.md                    this file
├── requirements.txt             tested versions
├── configs/
│   ├── config_v0.yaml           DELIVERABLE 1: the V5 config, every setting sourced
│   ├── lr_schedule.yaml         side file: the WSD learning-rate schedule
│   ├── data_mix.yaml            side file: production mix (open) + the harness sample
│   └── tiny/
│       ├── tiny_dense.yaml      inherits config_v0; laptop sizes, dense
│       ├── tiny_moe.yaml        inherits tiny_dense; 16 experts, 2 per token, 1 shared
│       ├── tiny_moe_g6.yaml     inherits tiny_moe; larger, for G6 to time
│       └── lr_tiny.yaml         the WSD schedule shortened to 200 steps
├── harness/
│   ├── configlib.py             reads configs: values, source comments, inherits, side files
│   ├── check_config.py          DELIVERABLE 2: the config check
│   ├── tokenizer.py             test tokenizer + the adapter the Kronecker embedding needs
│   ├── build_sample.py          builds the mixed English/Hindi/Bengali/Telugu sample
│   ├── data.py                  serves training windows in a fixed, seeded order
│   ├── model.py                 the tiny model
│   ├── train.py                 DELIVERABLE 3: the training loop and metric log
│   ├── determinism.py           seeds and deterministic settings
│   └── seed_test.py             the gate test: two runs, compared bit for bit
├── third_party/kronecker_embeddings/   the lab's Kronecker code, vendored unmodified (Apache 2.0)
├── assets/tokenizer_session2_enhibnte.json   test tokenizer (7,911 tokens; en, hi, bn, te)
├── references/hf_configs/       the six open MoE config.json files used as sources
├── evidence/                    run outputs behind the results table (local paths removed)
├── data/                        input text (git-ignored; see data/README.md)
└── work/                        harness outputs (git-ignored)
```

**Suggested reading order (about an hour):** this README → `config_v0.yaml` → `tiny/tiny_dense.yaml` and
`tiny/tiny_moe.yaml` → `check_config.py` with `configlib.py` → `build_sample.py`, `data.py`, `tokenizer.py` →
`model.py` (start with `TinyModel`, then `Block`, `Attention`, `SwiGLU`, `MoE.forward`,
`MoE.update_expert_bias`) → `train.py` (`run()` top to bottom) → `determinism.py`, `seed_test.py`. Every
Python file starts with a docstring in plain words. Tags like `[S14]` in comments name the course session a
rule comes from.

---

## 5. Design

### 5.1 Config v0: every setting has a value and a source

Every line of [`config_v0.yaml`](configs/config_v0.yaml) has the same shape:

```yaml
key: value      # source: where the value comes from (open: what may still change it)
```

For example:

```yaml
embedding:
  type: kronecker    # source: S7 (factored input, 93.75% saving); call 25 Oct (always test with Kronecker)
moe:
  bias_update_rate: 0.001    # source: S14 (update speed near 0.001)
```

The source codes are explained in the file's header: `S7`…`S14` are course sessions; `brief` is the G5
week-1 brief and the lab plan sheet; `call 25 Oct` is the team call; `V4 prod` is the V4 production config
shared in the G5 group; `refs` are the six open MoE configs in [`references/hf_configs/`](references/hf_configs/);
`KE repo` is the Kronecker embedding repository.

| Section | What it controls | Settled defaults (examples) |
|---|---|---|
| `meta` | name, version, status | config version 0, draft |
| `tokenizer` | which tokenizer | BrahmicTokenizer-131K (vocabulary 131,072); path `open` until G3 freezes it |
| `embedding` | how a token becomes a vector | Kronecker, byte alphabet 256, `pos_dim` 32 |
| `model` | the network's shape | sizes `open`; RMSNorm, SiLU (SwiGLU), first layer dense, dropout 0, no weight tying |
| `attention` | how tokens see each other | rotary positions (θ = 10⁷), GQA, V4's long-context choices (linear state, DDDGDDDG) as defaults |
| `moe` | the expert layers | sizes `open`; 1 shared expert, sigmoid router in fp32, bias balancing (rate 0.001), sequence loss 1e-4, dropless |
| `output_head` | prediction and loss | dense head (KE2 waits for G4), full softmax, chunked cross-entropy, 1 MTP head with weight 0.3, z-loss 1e-4 |
| `optimizer` | AdamW and clipping | β = 0.9 / 0.95, weight decay 0.1 (not on norms, biases, router), clip at 1.0 |
| `training` | the run | seed 1234, bf16, per-token loss, batch and sequence `open` |
| `parallel` | how the real run uses GPUs | ZeRO-2, tensor and expert parallel 8, sequence parallel |
| `checkpoint`, `observability` | saving and logging | every 780 steps, keep 3; the metric list (5.8) |

**The 16 settings still `open`**, each with a source saying who gives it and when:

| Who | Settings |
|---|---|
| TSAI, 31 Oct | `model.num_hidden_layers`, `hidden_size`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `intermediate_size`, `max_position_embeddings`; `moe.num_experts`, `num_experts_per_tok`, `moe_intermediate_size`; `training.global_batch_tokens`, `sequence_length`; `schedule.total_steps` (follows from the above); `production.mix` (the data mix) |
| G3, 14 Nov | `tokenizer.path` (the frozen tokenizer with its sha256) |
| G4, 27 Oct | `output_head.ke2_output_shape` |

### 5.2 Side files and inheritance

* **Side files.** As V4 did, the schedule and the data mix are separate files that the config points to:
  `optimizer.lr_schedule_file` → [`lr_schedule.yaml`](configs/lr_schedule.yaml) (WSD: peak 2e-4, 4,000
  warm-up steps, linear decay over the last 10% to 5% of peak) and `training.data_mix_file` →
  [`data_mix.yaml`](configs/data_mix.yaml).
* **Inheritance.** A tiny config starts with `inherits: ../config_v0.yaml` and writes only what differs.
  `tiny_moe` = config v0, then `tiny_dense`'s changes, then its own. Every setting not written keeps its value
  and its source from config v0, so a change to config v0 reaches every test model.
  `check_config.py configs/tiny/tiny_moe.yaml --list` shows which file each setting comes from.
* **Paths** inside a config (settings named `path` or ending in `_file`, `_dir`, `_pack`, `_text`) are
  resolved relative to the file that sets them, so the configs work from any folder.
* The `harness:` section at the end of `tiny_dense.yaml` holds settings only the test program uses (device,
  threads, micro-batch size, evaluation interval).

| Config | Layers | Width | Heads (query / key-value) | Feed-forward | Sequence | Tokens per step | Parameters |
|---|---|---|---|---|---|---|---|
| `tiny_dense` | 4 | 128 | 4 / 2 | dense SwiGLU, width 384 | 128 | 2,048 | 3,337,216 |
| `tiny_moe` | 4 | 128 | 4 / 2 | layer 1 dense; layers 2–4: 16 experts (width 128), 2 per token, + 1 shared | 128 | 2,048 | 5,407,744 (3,343,360 active) |
| `tiny_moe_g6` | 6 | 256 | 8 / 2 | 32 experts (width 256), 4 per token, + 1 shared | 256 | 8,192 | about 39.2M (11.6M active), from a 5-step run |

### 5.3 The config check

[`check_config.py`](harness/check_config.py) reads the config, every config it inherits from and its side
files **line by line**, so it sees the comments that YAML readers drop
([`configlib.read_settings`](harness/configlib.py)). Every setting lands in one group
(`classify()` in `check_config.py`):

| Group | Rule | Allowed in config v0? |
|---|---|---|
| **ERROR** | no value, or no `# source:` comment | no; the command exits with 1 |
| **WAITING** | value `open`, and the source says who gives it and when | yes |
| **OK** | a value and a source | yes |

It prints the WAITING and ERROR lines (`group`, `section.key`, `file:line`, the source text) and a total.
When a tiny config overrides a setting, only the last value counts. A test file with one setting missing its
source and one missing its value gave 2 errors and exit code 1.

### 5.4 The training text

[`build_sample.py`](harness/build_sample.py) builds the sample once. It tokenizes each source with the
test tokenizer and keeps the last 5% of each as held-out text:

| Source | Training bytes | Training tokens | Bytes per token | Share of windows |
|---|---:|---:|---:|---:|
| English (500 FineWeb-Edu documents) | 2,332,759 | 883,318 | 2.6 | 50% |
| Hindi | 426,682 | 61,672 | 6.9 | 25% |
| Bengali | 451,528 | 69,557 | 6.5 | 15% |
| Telugu | 174,639 | 26,807 | 6.5 | 10% |

The model reads **windows** of 131 consecutive tokens: 128 to read, plus 1 and 2 more for the next-token and
MTP targets. For each window, [`data.py`](harness/data.py) (`MixedSample._rows`) draws the source by its
share, then a random start inside it. Both draws use a generator seeded with the run's seed, so two runs see
the same windows in the same order. Held-out windows are drawn once, from a second generator (seed + 1).
`work/sample/manifest.json` (copy in [`evidence/sample_manifest.json`](evidence/sample_manifest.json)) records
the sha256 of every input file and of the tokenizer.

The test tokenizer is a BPE tokenizer built in course session 2 for English, Hindi, Bengali and Telugu
(7,911 tokens). It stands in until G3 freezes BrahmicTokenizer-131K.

### 5.5 The tiny model ([`model.py`](harness/model.py))

```
token ids ─▶ Kronecker embedding ─▶ [Block] × 4 ─▶ RMSNorm ─┬─▶ next-token head (Linear → 7,911 scores)
                                                             └─▶ MTP head (Linear → 7,911 scores, token after next)
Block:  x + Attention(RMSNorm(x))  then  x + FeedForward(RMSNorm(x))
        FeedForward = SwiGLU in layer 1; mixture of experts in layers 2–4 (tiny_moe)
```

**Kronecker embedding.** A normal model keeps one learned vector per token (7,911 × 128 = 1,012,608 weights
here). The Kronecker embedding turns each token's UTF-8 bytes into a fixed pattern (256 byte values × 16
positions = 4,096 numbers) and multiplies it by one learned matrix of 4,096 × 128 = 524,288 weights. Tokens
that share bytes share structure, across scripts. The code is the lab's package, vendored unmodified in
`third_party/`. The tiny configs use `mode: cached`: the codec table (7,911 × 4,096) is built once, which also
avoids the `scatter_add` of the default mode that a GPU refuses in deterministic mode.

**Attention.** Grouped-query attention (4 query heads share 2 key/value heads), rotary positions, causal
mask, through PyTorch's `scaled_dot_product_attention` (the plain math kernel on a GPU, for determinism).

**Mixture of experts** (`MoE.forward`), per token:

1. **Score.** A router (Linear from the width to 16 experts, computed in fp32) gives a sigmoid score per expert.
2. **Choose.** Each expert's *bias* is added to its score, only for choosing; the 2 highest are kept.
3. **Weigh.** The chosen experts' original scores are rescaled to sum to 1 (0.61 and 0.58 become 0.51 and 0.49).
4. **Compute, dropless.** Every (token, expert) pair is computed; nothing is skipped. Each result goes into
   its own slot of a buffer (`index_put`), and a token's two results are added with their weights. Writing
   each slot once, instead of adding into a shared place, keeps two runs identical.
5. **Shared expert.** One more expert that every token uses, added on top.
6. **Balance** (`MoE.update_expert_bias`, after each optimizer step). Each layer counts the tokens each expert
   got over the whole step and moves its bias by `rate × sign(average − load)`: up for under-used experts,
   down for over-used ones. A sequence-level balance loss (coefficient 1e-4) pushes the same way.

### 5.6 Loss, schedule and bits per byte

```
loss = loss_ntp + 0.3 × loss_mtp + 1e-4 × z-loss + MoE balance loss
```

* `loss_ntp` is the next-token cross-entropy, `loss_mtp` the same for the token after next. Both are computed
  in chunks of the sequence (`chunked_cross_entropy`) so the full score matrix never exists at once. The
  z-loss keeps the softmax normaliser from drifting.
* **Per-token normalisation.** A step is 16 windows in 2 micro-batches of 8. Each micro-batch's summed loss is
  divided by the token count of the *whole step* before its backward pass, so two micro-batches give exactly
  the gradient of one big batch (S10).
* **Schedule** (`wsd_lr` in `train.py`): linear warm-up, flat at the peak, linear decay at the end. Tiny runs
  use [`lr_tiny.yaml`](configs/tiny/lr_tiny.yaml): peak 3e-3, 20 warm-up steps, decay over the last 20% to
  10% of the peak. With `--steps N` the warm-up is capped at N/10 steps.
* **Optimiser.** AdamW; no weight decay on norms, biases or the router; gradient norm clipped at 1.0.
* **Bits per byte** = next-token loss in bits ÷ UTF-8 bytes of the predicted text, using a table of bytes per
  token (the Metaspace marker ▁ counts as one space byte). A Hindi token here is about 7 bytes and an English
  one about 2.6, so a loss per token would not compare the two; bits per byte does, and it does not depend on
  the tokenizer.

### 5.7 Making two runs identical

* [`determinism.py`](harness/determinism.py): Python, NumPy and PyTorch seeds set together;
  `torch.use_deterministic_algorithms(True)` (PyTorch raises an error instead of silently using an operation
  that can differ between runs); `CUBLAS_WORKSPACE_CONFIG`; cuDNN autotuning off; a fixed number of CPU
  threads; on a GPU, the math attention kernel.
* The data order comes from a seeded generator; dropout is 0.
* The MoE layer writes each expert result into its own slot and counts tokens with a one-hot sum.
* [`train.py`](harness/train.py) writes `losses.txt` with the losses and gradient norm as **exact hexadecimal
  floats**, for example `0x1.76c1806087cc6p+3` (= 11.7111…). "The same" therefore means bit-for-bit
  identical, and [`seed_test.py`](harness/seed_test.py) checks it with a plain line-by-line comparison of two
  runs started as fresh processes.

### 5.8 Metrics written at every step (proposal for G8)

`metrics.jsonl` has one JSON line per step. The first six names are the team's list from the V4 dashboard
(call 25 Oct); the rest are proposed additions.

| Name | Meaning | Unit |
|---|---|---|
| `step` | optimizer step, from 0 | — |
| `loss` | the full training loss (5.6) | nats per token |
| `loss_ntp` | next-token loss | nats per token |
| `loss_mtp` | multi-token-prediction loss | nats per token |
| `lr` | learning rate of the step | — |
| `tokens_per_second` | training tokens per second of the step | tokens/s |
| `gpu_alloc_gb` | GPU memory allocated (0 on CPU) | GB |
| `bits_per_byte` | next-token loss per byte of the training batch | bits per byte |
| `grad_norm` | gradient norm before clipping | — |
| `moe_aux_loss` | sequence-level balance loss | — |
| `expert_load_max`, `expert_load_min` | busiest and quietest expert's tokens as a multiple of an even share, over all MoE layers (1.0 = balanced; 8 = one expert takes everything with 16 experts and 2 per token) | × |
| `val_bits_per_byte` | bits per byte on the fixed held-out windows, every 20 steps | bits per byte |

**Reading a printed line:**

```
  step  199  loss 6.4600  ntp 4.7106  mtp 5.7973  bpb 1.366 val_bpb 1.559  lr 3.68e-04 load 0.52-1.39  3,444 tok/s
```

The next-token loss at step 0 should be near ln(7,911) = 8.98, the loss of an even guess (it is 9.0033).
`load` is `expert_load_min`–`expert_load_max`. `tok/s` varies from run to run, which is why the seed test
does not compare it.

---

## 6. How one training step flows through the code

What happens when you run `python harness/train.py --config configs/tiny/tiny_moe.yaml`:

| # | What happens | Where |
|---|---|---|
| 1 | Read config v0 + `tiny_dense` + `tiny_moe` overrides + the two side files | `configlib.load_config` |
| 2 | Fix every source of randomness | `determinism.make_deterministic` (called in `train.run`) |
| 3 | Load the sample, build the model and AdamW with its parameter groups | `train.run`, `train.param_groups` |
| 4 | Set the step's learning rate | `train.wsd_lr` |
| 5 | For each of the 2 micro-batches: draw 8 windows in the seeded order | `data.MixedSample.batch` → `_rows` |
| 6 | Kronecker embedding, then the 4 blocks | `model.TinyModel.forward` |
| 7 | In an MoE block: score, choose top 2 with the bias, compute every pair into its own slot, add the shared expert | `model.MoE.forward` |
| 8 | Next-token and MTP losses in chunks; add z-loss and balance loss; divide by the step's token count; backward | `model.chunked_cross_entropy`, `train.run` |
| 9 | Clip the gradient norm to 1.0; AdamW step | `train.run` |
| 10 | Move each layer's expert biases using the whole step's token counts | `model.MoE.update_expert_bias` |
| 11 | Stop on NaN; write the `metrics.jsonl` line and the exact `losses.txt` line; every 20 steps, held-out bits per byte | `train.run`, `train.evaluate` |

---

## 7. Evidence

[`evidence/`](evidence/) holds the outputs behind section 2, copied from `work/` with local folder names
removed. Each run folder has three files:

* `losses.txt`: one line per step: step, total loss, next-token loss, MTP loss, gradient norm, as exact hex floats.
* `metrics.jsonl`: every metric of section 5.8, one JSON line per step.
* `run_info.json`: the fully resolved config (after inheritance), parameter counts, device, library versions
  and the sample manifest, so any run can be traced back.

| Path | What it shows |
|---|---|
| `evidence/seed_test/tiny_moe/run_a/`, `run_b/` | the two seed-1234 runs of the gate test (40 steps) |
| `evidence/seed_test/tiny_moe/seed_test_result.json` | the verdict: `identical`, `diff_lines` 0, first and last loss, `finite` |
| `evidence/seed_test/tiny_moe/control_seed1235/` | the same config with seed 1235: every line differs from `run_a` |
| `evidence/seed_test/tiny_dense/` | the same gate test for the dense model |
| `evidence/runs/tiny_moe_seed1234/`, `tiny_dense_seed1234/` | the full 200-step runs |
| `evidence/sample_manifest.json` | sizes and sha256 of the sample's inputs and of the tokenizer |

Check it yourself, without running anything:

```bash
git diff --no-index evidence/seed_test/tiny_moe/run_a/losses.txt evidence/seed_test/tiny_moe/run_b/losses.txt          # no output: identical
git diff --no-index --stat evidence/seed_test/tiny_moe/run_a/losses.txt evidence/seed_test/tiny_moe/control_seed1235/losses.txt   # 40 lines changed
```

Or rerun: `python harness/seed_test.py` writes the same files to `work/seed_test/tiny_moe/`. On the same
PyTorch version and CPU, its `losses.txt` should match the one here exactly. On another machine or version
the numbers may differ in the last bits, but run A and run B still match each other.

---

## 8. Findings

1. **The S14 bias update rate is too slow for short runs.** With `bias_update_rate: 0.001`, one expert took
   about 5–6× its share of tokens in a 100-step tiny run; at 0.01 the load settled at about 0.4×–1.9× with the
   same held-out bits per byte (1.68–1.69). 0.001 is meant for runs of many thousands of steps, so config v0
   keeps it and only the tiny configs use 0.01. In the 40-step seed test the load is still uneven (up to about
   7.9× at step 8): balancing needs more than 40 steps, and the imbalance is identical in both runs.
2. **The load must be counted over the whole step.** The first version moved the bias using the last
   micro-batch's counts only. It now accumulates over both micro-batches, as S14 says ("load counted across
   the whole batch").
3. **Cached Kronecker mode is needed for GPU determinism**, because the default mode uses `scatter_add`, which
   PyTorch refuses in deterministic mode on a GPU.

---

## 9. What this is not yet

* The output head is dense; the KE2 head waits for G4's output shape (27 Oct).
* Attention is plain GQA. The long-context parts in config v0 (linear state, the DDDGDDDG schedule, the
  Memory Stream) are not in the tiny model; they belong to experiment EXP-4.
* The MTP head is one extra Linear on the final hidden state, simpler than DeepSeek's MTP module with its own block.
* No reversible layers, no parallelism, no checkpoints, fp32: those matter at full width (weeks 3 and 4).
* The tokenizer is the session-2 test tokenizer, not BrahmicTokenizer-131K.
* Run on CPU only. The GPU path (`--device cuda`, with its deterministic settings) is written but untested.

---

## 10. Sources and provenance

* **Course sessions** S7 (embeddings), S8 (attention), S9 (loss and output heads), S10 (training loop),
  S11 (optimisers and schedules), S12–S13 (distributed training), S14 (mixture of experts), S16 (scaling,
  bits per byte).
* **Open model configs** in [`references/hf_configs/`](references/hf_configs/), from Hugging Face
  (`<org>/<model>/blob/main/config.json`): OLMoE-1B-7B-0125, LFM2.5-24B-A2B, MiMo-V2.6-Flash-RL,
  DeepSeek-V4.1-Flash, GLM-5.3-Flash, Kimi-K3.
* **Papers** cited for defaults: DeepSeek-V3 (arXiv 2412.19437: MTP weight 0.3, bias balancing, sequence loss);
  PaLM (2204.02311: z-loss 1e-4); MiniCPM (2404.06395: WSD's short decay); Llama 2 (2307.09288: AdamW
  betas, weight decay 0.1); Kronecker Embeddings (2605.29459).
* **Kronecker embedding code:** <https://github.com/theschoolofai/kronecker-embeddings>, commit
  `d07cb543f03d7a72a7c35bc68cfb795bcb02e1ec`, vendored unmodified under Apache 2.0
  ([`third_party/kronecker_embeddings/VENDORED.txt`](third_party/kronecker_embeddings/VENDORED.txt)). If
  `pip install kronecker-embeddings` works, the harness uses the installed package instead.
* **Test tokenizer:** [`assets/tokenizer_session2_enhibnte.json`](assets/tokenizer_session2_enhibnte.json),
  the BPE tokenizer from the session 2 assignment.
* **V4 production config** (`train_1b_prod.yaml`, shared in the G5 group): the source of the `V4 prod`
  values. Not included here because it contains internal paths.
