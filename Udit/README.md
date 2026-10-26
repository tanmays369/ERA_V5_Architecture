# G5 Week 1 audited research suite v3

Separate revision; v1/v2 source and evidence remain unchanged. This is a single-process FP32 research reference, not a certified frontier training stack. Source comments separate instructor/user requirements, provisional G5 hypotheses and pending G4/TSAI decisions. No V4 recipe, XAI package or unverified G4 implementation is imported.

## What is implemented

- Exact valid-target token budget per optimizer update; automatic microbatch accumulation and final-target trimming independent of padded slots. Sequence lengths may change at stage boundaries. Every scored content target has exact byte accounting; remaining corpus positions stay sampler-controlled.
- Serializable three-stage curriculum with cosine/linear ramps for mixture probabilities, auxiliary coefficient and router temperature. WSD warmup/stable/anneal controls both optimizers. Stage fractions are update fractions, not token fractions; token budget is invariant.
- Generic per-slice sampling, minimum mixture floor and an online excess-loss feedback hook using held-out per-slice CE minus configured reference CE. This is inspired by DoReMi, not its proxy/reference-model training procedure. Reference losses require calibration; zero defaults are only wiring values.
- Standard/latent routed MoE, shared experts, optional first-K dense FFNs, selection-only detached balancing bias, switch auxiliary or hybrid mode, padding exclusion, deterministic capacity masks and dropless mode. Bias updates once AFTER an optimizer update, from accumulated pre-capacity routing demand, never in validation.
- Six explicit attention controls: GQA, MHA, MQA, MLA, interleaved sliding-window hybrid, and differential attention. QK and V widths are decoupled; MLA uses latent KV/query ranks and separate no-RoPE/RoPE dimensions. QK RMSNorm, optional softcaps, partial RoPE, linear RoPE scaling, learnable sinks, and cross-layer KV reuse have reference implementations. Cached decode is exposed by `Model(..., use_cache=True)` in eval/no-grad; follow-up chunks require explicit position/document IDs. These are correctness reference paths, not fused kernels.
- Pre/post (sandwich) RMSNorm placement, aligned dense/expert FFN widths, routing score/group controls, MTP heads, precision/checkpoint/evaluation contracts and 4D parallelism schema are explicit config fields. Only single-process FP32 executes; multi-dimensional parallel sizes are schema contracts, not a distributed runtime.
- Empirical KE lookup variance normalization, explicit scale and width projection adapter; injectable external embedding provider with shape guard. This does not invent G4 factor equations. Untied dense, dynamically tied KE, or chunked factorized KE output head; tied/factorized modes are correctness-oriented and can be expensive at 131K vocabulary.
- Four semantic optimizer roles plus instructor algorithm allocation: internal matrices Muon, head/embedding/router/norm/bias AdamW. Router FP32/no decay; separate KE LR/decay. Legacy aliases for shared expert count, dropless, QK norm, tying and embedding decay must agree with their new fields.
- muP-inspired width hooks for initialization, attention 1/d_head, readout 1/width multiplier and internal matrix LR scaling; **not proof of maximal-update transfer**, especially with Muon and MoE. Coordinate tests and scale sweeps remain required.
- Isolated native XAI/SAE probes, train/held-out metrics, safe tensor checkpointing, exact same-environment resume including sampler/controller/RNG state.

## Handover commands (Bash)

From the directory containing `teams/`; installed suitable torch, PyYAML, numpy and tokenizers required. Do not replace a working CUDA install casually.

```bash
set -euo pipefail
G5=teams/05-architecture
TOKENIZER=/absolute/path/to/BrahmicTokenizer-131K/tokenizer.json
export BRAHMIC_TOKENIZER="$TOKENIZER"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=1234

# G7: schema/source handoff; pending sizes are printed, not fabricated.
python "$G5/harness/check_config.py" --allow-open-production-sizes

# G6: concrete executable tiny reference: base configuration plus tiny overrides.
python "$G5/harness/check_config.py" "$G5/configs/config_v0.yaml" --override "$G5/configs/tiny_moe.yaml"
python -m unittest discover -s "$G5/tests" -v

for RUN in run1 run2; do
  python "$G5/harness/train.py" --seed 1234 --device cuda \
    --tokenizer "$TOKENIZER" --out "$RUN"
done
cp run1/loss.txt run1.log
cp run2/loss.txt run2.log
diff -u run1.log run2.log
diff -u run1/total_loss.txt run2/total_loss.txt
python "$G5/harness/verify_runs.py" run1 run2

# Resume: preserve the full 30-step schedule; stop execution, not schedule length.
python "$G5/harness/train.py" --seed 1234 --device cuda \
  --tokenizer "$TOKENIZER" --out paused --stop-after 15
python "$G5/harness/train.py" --seed 1234 --device cuda \
  --tokenizer "$TOKENIZER" --out resumed --resume paused/checkpoint.pt
diff -u run1/loss.txt resumed/loss.txt

# G8: exact dictionary + real sample records.
cat "$G5/harness/metrics_contract_g8.md"
head -n 1 run1/metrics.jsonl
```

G7 open-size mode validates source/schema and non-size contracts, prints pending fields and explicitly defers unknown dimension invariants. It is NOT a runnable-production certificate. Saturday's actual-size check must pass WITHOUT that flag or tiny override after genuine TSAI/G4 resolution. Unknown fields, missing sources and nulls outside the declared pending-size list remain errors.

The unit suite also checks all six attention mechanisms, both router score functions, asymmetric QK/V widths, per-layer and grouped model KV-cache decode against prefix recomputation, and deterministic backward gradients. It runs five fresh processes for two-repeat, XAI on/off and uninterrupted-vs-15+15-resume gates. These take longer than pure unit tests; the tokenizer is required to run them rather than skip them. Use cpu if needed, but compare only identical hardware/software/device/config/data within a pair. Native sparse probes stay optional; random mode is an untrained test baseline.

## Data

`--data-path` overrides data.dataset_path; default fixture is a repeated two-document wiring check. JSONL text, TXT lines preserving newline bytes, pinned little-endian uint32 BIN and safe PT tensors are supported; JSONL.GZ also works. `--validation-data-path` gives an explicit held-out source; otherwise a deterministic exact-document-hash partition is used. No required semantic domains. For multiple slices, configure arbitrary domain_labels and provide matching JSONL slice_id (domain fallback) or binary-sidecar slice_id. Each configured slice needs training windows; online mixing additionally requires validation windows for every slice.

```bash
python "$G5/harness/train.py" --seed 1234 --device cuda \
  --tokenizer "$TOKENIZER" --data-path /data/train.jsonl \
  --validation-data-path /data/validation.jsonl --slice-documents 64 --out pilot
python "$G5/harness/prepare_tokens.py" --input /data/train.jsonl \
  --tokenizer "$TOKENIZER" --out /data/prepared/train.bin
```

BIN requires <file>.meta.json with dtype=uint32 and actual tokenizer_sha256. PT uses weights_only=True and integer input_ids plus tokenizer fingerprint. Arbitrary pickle and guessed tokenizers are rejected. Directory ingestion prefers binary over raw copies; do not mix train/validation files in one directory. Text is checked against exact reconstructed token bytes; pretokenized BPB refers to reconstructed text unless independently checked against original raw data. EOS/BOS delimit documents and are not prediction targets here; deciding EOS prediction is still a long-run objective decision.

Default deterministic_slice bounds documents; full mode uses the corpus. Fingerprinting/delimiter scans may inspect entire shards. The token cache is disk-backed. Evaluation is a fixed capped sample of 64 windows at maximum configured context/final theta and temperature; not a full-corpus capability evaluation. Arbitrary real-batch loss need not decrease every update. The tiny fixture gate tests that stronger condition empirically.

## Checkpoint guarantees and limits

Saved: model weights/buffers including router bias/pending counts, both optimizer states and group metadata, WSD state, curriculum weights/tokens, active-stage geometry, sampler permutation/cursors/generator, diagnostic histories, Python/NumPy/CPU/CUDA RNGs, loss/metric prefix and strict code/config/data/tokenizer/environment provenance. Resume refuses mismatched provenance. Files are written to a temporary file and atomically renamed. Stop/resume must keep training.steps unchanged. Prefix metrics are replayed to a new output directory; timing fields differ from uninterrupted execution, numerical records must match.

Checkpoints can be large and are not included in the ZIP. No cross-hardware bitwise claim, crash-atomic external database transaction, DDP/ZeRO engine or G7 cluster checkpoint system is claimed. Model parameters and diagnostics remain FP32. External custom providers/SAEs need explicit compatible artifacts and checkpoint policy.

G4 delivery, G8 logger/temporal alarms, production sizes, real distributed correctness/performance and TSAI plan approval remain pending. Nothing has been pushed or submitted automatically.
