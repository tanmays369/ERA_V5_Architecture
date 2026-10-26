# G8 metric contract v3

Scope: single-process FP32 research reference. Earlier v2 aliases remain, with the clarifications below. Exact names in this document are consumed by the experiment plan. No required semantic domain names. Source: current instructor/user requirements; DeepSeek-V3 report https://arxiv.org/abs/2412.19437; DoReMi https://arxiv.org/abs/2305.10429; muP https://github.com/microsoft/mup. Implemented variants are research controls, not reproductions of whole model recipes.

## Update-level records

metrics.jsonl: one sorted JSON dictionary per optimizer update. `step` starts at 1. Training losses are pre-update; held-out metrics are post-update at evaluation.eval_every_steps (tiny=1). Local files flush every 10 updates and on close. Numerical alarms flush immediately. loss.txt stores only update index and CE at 17 significant digits; total_loss.txt additionally records the optimized objective. Neither includes timestamps.

| Name | Definition |
|---|---|
| `loss/ce`, `train/ce_loss`, `train/ce` | Sum text-target NLL / exact valid target budget |
| `loss/moe_aux` | Valid-token-weighted mean microbatch switch auxiliary across routed layers; zero optimized term in aux_loss_free_bias mode |
| `loss/router_z` | Mean routed-layer logsumexp(router_logits)^2, weighted by valid targets |
| `loss/logit_z` | Mean full-vocabulary logsumexp(logits)^2 over text targets |
| `loss/total` | CE + ramped auxiliary coefficient * aux + router-z coefficient * router-z + output-z coefficient * output-z |
| `train/perplexity`, `train/token_perplexity` | exp(loss/ce), excludes all regularizers |
| `train/bpb` | Aggregate text NLL / (ln(2) * exact scored byte count) |
| `train/valid_tokens`, `train/raw_utf8_bytes` | Actual scored target and byte counts |
| `val/perplexity`, `eval/perplexity` | exp(aggregate held-out NLL / held-out valid tokens), model.eval + no_grad, no aux/z losses |
| `val/bpb`, `eval/bpb` | Aggregate held-out text NLL / ln(2) / held-out bytes |
| `eval/ce`, `validation/ce`, `validation/bpb` | Legacy held-out aliases |
| `probe/ce`, `probe/bpb` | Fixed training-window diagnostic, never advertised as held-out quality |
| `curriculum/tokens_per_step` | Actual valid target tokens this optimizer update; equals configured global_tokens_per_step |
| `curriculum/tokens_seen` | Cumulative valid target tokens, restored on resume |
| `curriculum/grad_accum_steps` | Number of microbatch forwards/backwards needed for the exact budget; not padded-slot division |
| `curriculum/stage`, `curriculum/seq_len`, `curriculum/rope_theta` | Active stage, training context and theta |
| `curriculum/moe_aux_loss_coeff`, `curriculum/router_temperature` | Smoothed active controls |
| `curriculum/transition_progress` | Linear time progress in boundary ramp, 0..1; interpolation may apply cosine to this progress |
| `curriculum/mixture_weights` | Actual slice-sampling probabilities after ramp/online feedback and floor; not a claim of exact observed token proportions |
| `curriculum/observed_tokens_per_slice` | Actual valid training targets by arbitrary configured slice ID |
| `curriculum/validation_slice_ce` | Online-mode held-out per-slice CE vector, in configured label order; absent in static mode |
| `optimizer/wsd_factor` | Shared warmup/stable/decay multiplier on each group's peak LR |
| `optimizer/adamw_lr`, `optimizer/muon_lr` | Base rate times WSD; per-role effective rates additionally apply documented multipliers in manifest optimizer groups |
| `stability/grad_norm_preclip`, `stability/clipped` | Global preclip norm over both optimizers and whether threshold exceeded |
| `perf/step_seconds`, `perf/tokens_per_second` | Synchronized local training time/rate including microbatches/diagnostics, excluding evaluation/checkpoint/file writing; not G6 component timers |
| `data/window_indices`, `data/batch_sha256` | Drawn window indices and hash of actual padded targets in microbatch order |

Cross-microbatch CE uses one global valid-token denominator. The balancing objective remains a weighted microbatch estimate; this is not falsely presented as a single global-routing auxiliary calculation. Expert bias-controller counts ARE accumulated over the whole optimizer update. Last microbatch may be target-trimmed; discarded target slots do not count toward loss or BPB.

## Per-layer records

`layers[layer_index]`; use layer/head/tap_point labels. Routed expert counts sum over microbatches; aggregate entropy/max violation/zero-use count are recomputed from those counts. Other numeric diagnostics are valid-token-weighted microbatch means. SAE observations are sampled tap diagnostics, not every-token full-dataset summaries.

| Name | Meaning |
|---|---|
| `moe/expert_counts` | Delivered routed slot counts, excludes pads/shared experts; sum=K*N only in dropless mode |
| `moe/utilization_entropy`, `explain/expert_utilization_entropy` | -sum(f ln f)/ln(E) from delivered counts; E=1 gives zero |
| `moe/max_load_violation`, `explain/max_load_violation` | max delivered count / (K*N/E) - 1 |
| `moe/dead_expert_count`, `explain/dead_expert_count` | Zero-assignment experts in this update; not proof of persistent dead experts |
| `moe/router_bias_max_abs` | Largest absolute selection-only bias BEFORE this update's controller commit |
| `moe/shared_expert_norm_ratio` | Valid-token-weighted microbatch mean ||shared output||/(||routed output||+1e-12) |
| `moe/dropped_token_ratio` | Fraction of valid tokens losing ALL routed assignments; shared path may still process them |
| `moe/dropped_slot_ratio` | Fraction of requested routed slots dropped by deterministic capacity masks |
| `moe/router_z_loss`, `explain/router_z` | Per-layer unweighted router-logit z-loss |
| `moe/dense_prefix` | True for early dense replacement blocks; routed stats absent, not invented |
| `attn/qk_norm_scale` | RMS magnitude of rotated Q after optional QK norm; not the learnable norm gain itself |
| `attn/kv_cache_bytes_per_token` | Per-layer theoretical persistent KV payload bytes per token for this FP32 reference (excludes metadata and learned sinks); MLA stores latent+rotary channels, differential attention stores two K maps, standard modes store K+V |
| `attn/effective_kv_heads` | Effective KV-head count for this layer; MLA reports the latent shared representation separately through its cache-byte metric |
| `attn/is_local` | Whether this hybrid layer applies the configured sliding-window mask |
| `attn/context_extension_ratio` | Maximum current position divided by configured original maximum position; absent when the original context is open |
| `attn/sink_mass` | Mean attention probability assigned to learned sink slots over valid queries; emitted only when sink tokens are enabled |
| `explain/residual_l2`, `explain/residual_cosine` | Mean valid-token block output norm / cosine(block input, block output) |
| `explain/attention_entropy_per_head` | Mean valid-query Shannon attention entropy per head, nats |
| `explain/logit_lens_ce` | Intermediate residual through final norm/readout, scored on same text targets |
| `explain/logit_lens_kl` | KL(final distribution || intermediate distribution); final block should be ~0 |

Router selection uses unbiased softmax(logits/temperature) plus detached bias; gates use unbiased probabilities, optionally renormalized over retained selections. Bias update is once/update from accumulated PRE-capacity demand, to avoid capacity masking hiding overload; target is mean expert demand. Validation does not accumulate or update bias. Switch mode uses auxiliary only; bias mode disables switch aux; hybrid does both. Dense-prefix blocks are excluded from the routed-layer auxiliary average.

## Attention and cache semantics

Mechanisms accepted by config/model: `gqa`, `mha`, `mqa`, `mla`, `sliding_window_hybrid`, and `diff_attn`. Query-key and value head widths are independent. MLA stores its compressed latent KV plus shared rotary key; the output expands latent KV per head for attention. Sliding-window mode interleaves local layers with global layers by `global_attn_every_n_layers`. Differential mode uses two attention maps and a learned per-layer subtraction scalar. Learned sinks are static key/value slots, not text tokens. `kv_sharing_group_size` reuses the owner's projected KV tensors across each layer group; each layer still has separate queries/output projection. The model exposes `use_cache=True` in `eval()` under `no_grad`; after the initial chunk, callers must pass explicit `position_ids` and `document_ids` so EOS/document resets remain unambiguous. Cache reuse in this Python reference is for correctness experiments, not a fused/optimized decode kernel.

## Native hooks, probes and alarms

Taps: resid_mid after attention residual; mlp_out before MoE residual addition; moe_router_logits before temperature/softmax. Detached copies remove padding. Snapshots are bounded to one forward. Built-in random probes use private generators; callback/probe execution also restores Python/NumPy/CPU/active-CUDA RNG state. External code must not mutate model parameters or optimizer state.

`explain/sae` is a list of tap-labelled dictionaries: `explain/sae_reconstruction_mse`, `explain/sae_feature_l0`, `explain/sae_dead_feature_count`, `explain/sae_dead_feature_fraction`, `explain/sae_observation_count`. Window counts tap observations, potentially multiple microbatches per update. Random Top-K is untrained, not interpretable. Trained native checkpoint: encoder [D,F], decoder [F,D], optional bias [F]. No TransformerLens/SAELens dependency.

Fatal every-update events: alarm/nonfinite_loss, alarm/nonfinite_activation, alarm/nonfinite_gradient, alarm/nonfinite_parameter. Optional warning alarm/sae_dead_features only after a full configured tap window. G8 temporal spike/EMA/router/throughput alarms require separate calibration and synthetic tests. Nonfinite checkpoint/config/IO errors must fail execution, not become a fake success record.

Checkpoint restores numeric prefix, controllers, sampler, optimizer/scheduler and RNG state. Timing fields can differ on resume. Original score metrics must remain identical. Archive excludes large checkpoints/data; local checkpoint tests verify the actual tensors.

BPB counts text-token bytes, no structural EOS/PAD losses. Raw text is byte-verified; token-only shards reconstruct UTF-8 under a pinned tokenizer but do not prove identity to external original text. Evaluation uses fixed maximum context/final theta/temperature on up to 64 windows. Comparing different raw texts or contexts is not a controlled BPB experiment. G6 owns independent FP64/5% timing/MFU gates; G7 owns actual distributed/checkpoint infrastructure; G8 owns logger integration and alarm calibration.
