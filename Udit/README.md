# G5 Configuration Parameter Reference

Snapshot date:27 October2026. Source files: `configs/config_v0.yaml` and `configs/tiny_moe.yaml`.

Base configuration SHA256: `026e516e095480ce542604a1837c8ea079f508d31ab9c8251297bdee20e2389a`.

## Purpose

This reference explains every one of the 235 leaf settings in the submitted config_v 0.yaml, why it was included, and the tiny override used for verification. It records rationale without changing any configuration or code.

## How to read the values

The effective tiny configuration is the base YAML recursively merged with tiny_moe.yaml. OPEN means an explicitly pending dimension, not zero. A dash in the tiny column means the base value is inherited. A configured alternative may be inactive, validator-only or pending; its explanation identifies that boundary.

## Decision status

G5 baseline values are provisional engineering controls, not claimed production optima. Instructor requirements, pinned artifacts and partner-team contracts have stronger authority. The exact inline source comments remain in config_v 0.yaml. The 235-setting count excludes 28 parent mapping headers.

## Verified scope

The archived final sign-off records 104 passing tests and same-environment CPU/CUDA repeat, resume and XAI-isolation agreement for the tiny fixture. The last source-comment cleanup changed no keys or values; old training manifests retain the pre-cleanup config hash. These results are not 100M Kaggle results or architecture-quality wins.

## 1 Configuration identity

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `meta.schema_version` | `1` | Inherited | Identifies version 1 of the field schema so consumers can detect incompatible future formats. |
| `meta.profile` | `"production_open"` | `"tiny_reference"` | Marks the base as an open production contract; the tiny override selects the executable reference profile. |
| `meta.target` | `"general_purpose"` | Inherited | Records general-purpose intent without imposing language or domain names. Descriptive metadata, not a model switch. |

## 2 Tokenizer identity and vocabulary

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `tokenizer.identifier` | `"theschoolofai/BrahmicTokenizer-131K"` | Inherited | Names the required Brahmic tokenizer for people; the file hash provides the authoritative runtime identity. |
| `tokenizer.sha256` | `"01d968a7970b029c071b29f060abf6104b66168dedf2ff3b1a0290a545739fff"` | Inherited | Pins the exact tokenizer artifact, preventing silent vocabulary changes. This fingerprint is public-safe and is not a credential. |
| `tokenizer.vocab_size` | `131072` | Inherited | Defines the 131072 real token IDs; affects head size, prediction cost, and byte-accounting compatibility. |
| `tokenizer.pad_multiple` | `128` | Inherited | Requires vocabulary allocation alignment to 128. This is a shape constraint, not the sequence padding token. |
| `tokenizer.padded_vocab_size` | `131072` | Inherited | Allocates the output vocabulary dimension. It equals the real vocabulary because 131072 already divides by 128. |
| `tokenizer.bos_id` | `130725` | Inherited | Identifies beginning-of-sequence. The loader uses this ID at a document start; it must match the pinned tokenizer. |
| `tokenizer.eos_id` | `36` | Inherited | Identifies the document end marker for boundary handling. Structural EOS is excluded from the current scored-text objective. |
| `tokenizer.pad_id` | `130726` | Inherited | Fills unused sequence slots. Padding and ignored targets must not contribute to loss or expert-load statistics. |

Tokenizer file fingerprint: `01d968a7970b029c071b29f060abf6104b66168dedf2ff3b1a0290a545739fff`.

## 3 Kronecker embedding and G4 interface

The reference maps token bytes to deterministic byte-position features and learns a projection into model width. With char_dim256 and pos_dim32, codec width is8192. Dynamic mode avoids the full expanded vocabulary codec table. G4 KE2 is not yet integrated.

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `embedding.type` | `"kronecker"` | Inherited | Keeps Kronecker embeddings as the agreed production direction; the checker rejects an unsupported embedding type. |
| `embedding.ke2_variance_norm` | `true` | Inherited | Centers and variance-normalizes lookup vectors before the optional projection. This controls output scale, not hypothetical G 4 factor variance. |
| `embedding.ke2_embed_scale` | `1.0` | Inherited | Scales the adapted lookup output. One preserves the reference magnitude and provides a controlled future scale ablation. |
| `embedding.ke2_projection_adapter` | `"linear_if_needed"` | Inherited | Allows a learned linear adapter when provider width differs from model width, avoiding changes to transformer block shapes. |
| `embedding.output_scale` | `1.0` | Inherited | Scales embeddings entering the residual stream and the tied-head reference. One preserves the tested control. |
| `embedding.implementation` | `"reference_v1"` | Inherited | Identifies the executable reference_v 1 backend; it must not be mistaken for a delivered G 4 KE 2 module. |
| `embedding.codec_mode` | `"dynamic"` | Inherited | Computes byte-position codec features on demand. Dynamic mode avoids storing the full expanded vocabulary codec table. |
| `embedding.char_dim` | `256` | Inherited | Represents the 256 possible byte values. This is the byte alphabet, not the 131072-token vocabulary. |
| `embedding.length_normalize` | `true` | Inherited | Applies inverse-square-root byte-length scaling in the codec, limiting direct dependence on token byte count. |
| `embedding.z_normalize` | `true` | Inherited | Centers and standardizes deterministic codec features before their learned projection. Distinct from normalization after lookup. |
| `embedding.projection_init` | `"normal"` | Inherited | Uses the existing normal initializer with standard deviation equal to inverse square root of codec width. |
| `embedding.pos_dim` | `32` | Inherited | Represents up to 32 byte positions within a token. It is not the training context length or RoPE dimension. |
| `embedding.output_dim` | `OPEN` | `96` | Declares the model-facing embedding width. It must equal hidden_size; production remains open and tiny uses 96. |

### G4 output interface

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `embedding.ke2.status` | `"pending_g4"` | `"reference_only"` | Separates pending G 4 integration from deliberate reference-only execution. Delivered status requires an actual supplied embedding provider. |
| `embedding.ke2.layout` | `"batch_sequence_hidden"` | Inherited | Defines the provider output contract as batch by sequence by hidden channels, enabling explicit shape validation. |
| `embedding.ke2.output_dim` | `OPEN` | `96` | Declares raw provider width before adaptation. G 4 supplies the real value; tiny uses the 96-wide reference. |
| `embedding.ke2.adapter` | `"none"` | Inherited | Legacy interface field fixed to none. Actual width adaptation is controlled by embedding.ke 2_projection_adapter; do not edit this independently. |

## 4 Transformer backbone

These controls determine residual width, depth, FFN widths, initialization and normalization. The tiny configuration has two width96 blocks; production dimensions remain intentionally open.

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `model.init_std` | `0.02` | Inherited | Exposes the existing 0.02 normal-initialization scale for model matrices, allowing reproducible activation-scale ablations; embedding initialization is separate. |
| `model.d_ff` | `OPEN` | `192` | Sets intermediate width for dense FFN layers, independently of expert widths. Open in production and 192 in tiny. |
| `model.ffn_multiple_of` | `32` | Inherited | Validates FFN-width alignment to 32. It constrains chosen widths but does not automatically round or resize layers. |
| `model.residual_init_scale_mode` | `"none"` | Inherited | Leaves residual-output initialization unscaled. The depth_scaled alternative applies inverse square root of twice the layer count. |
| `model.hidden_size` | `OPEN` | `96` | Sets residual-stream width and strongly affects parameters, compute and memory. Production is open; tiny is 96 channels. |
| `model.num_layers` | `OPEN` | `2` | Sets transformer depth. More layers increase compute and saved activations; tiny uses two for a fast engineering gate. |
| `model.norm_eps` | `1e-06` | Inherited | Stabilizes RMSNorm division near zero. The 1 e-6 value is a reference numerical control, not an optimized production finding. |
| `model.norm_type` | `"rmsnorm"` | Inherited | Selects the implemented RMSNorm family. Arbitrary normalization alternatives are rejected until implemented and verified. |
| `model.norm_placement` | `"pre_norm"` | Inherited | Normalizes before attention and FFN updates. The peri_norm alternative adds normalization after residual updates in this implementation. |
| `model.norm_gain` | `true` | Inherited | Retains trainable normalization gains as requested by the instructor. These gains belong to AdamW with no weight decay. |
| `model.residual` | `"standard"` | Inherited | Executes ordinary residual additions. Midpoint reconstruction is represented separately and is not implemented by this setting. |

## 5 Pending midpoint reversibility contract

This section is an interface and acceptance contract only. No inverse or reversible backward is implemented. Tiny runs standard residual layers; activation checkpointing is a different mechanism.

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `reversibility.enabled` | `false` | `false` | Keeps unfinished reversible execution disabled; runtime and executable validation reject enabling it prematurely. |
| `reversibility.method` | `"midpoint"` | `"none"` | Records midpoint as the intended future method. Tiny overrides this to none and executes standard residual layers. |
| `reversibility.contract_status` | `"pending_equations"` | Inherited | Makes missing equations explicit rather than claiming a reversible implementation exists. |
| `reversibility.equations_ref` | `"pending_g6_midpoint_contract"` | Inherited | Reserves a reference to approved equations and state-reconstruction semantics. It currently names an unresolved G 6 contract. |
| `reversibility.segment_layers` | `1` | Inherited | Proposes one layer per reversible segment for future memory/recompute experiments. It currently changes no execution. |
| `reversibility.preserve_rng_state` | `true` | Inherited | Requires future backward recomputation to replay stochastic state correctly so reconstruction and repeatability remain meaningful. |
| `reversibility.count_router_updates_once` | `true` | Inherited | Requires recomputation not to duplicate expert counts or router-bias updates, which otherwise alter the training trajectory. |
| `reversibility.reconstruction_dtype` | `"float32"` | Inherited | Specifies FP 32 for an initial reconstruction reference. Lower precision needs separately measured error bounds. |
| `reversibility.reconstruction_rtol` | `1e-05` | Inherited | Proposes a 1 e-5 relative reconstruction tolerance against an independent reference; it is not a measured result. |
| `reversibility.reconstruction_atol` | `1e-06` | Inherited | Proposes a 1 e-6 absolute reconstruction tolerance for values near zero; requires agreement with G 6. |
| `reversibility.gradient_rtol` | `1e-05` | Inherited | Proposes 1 e-5 relative agreement between future reversible-backward gradients and ordinary autograd. |
| `reversibility.gradient_atol` | `1e-06` | Inherited | Proposes 1 e-6 absolute gradient agreement near zero. All reversible tolerances remain provisional acceptance contracts. |

## 6 Attention geometry and positional encoding

Queries match keys to select values. The default tiny GQA uses three query heads sharing one KV head, with32 QK and32 value channels. MLA and differential settings are conditional; switching a mechanism does not make its reference an optimized production kernel.

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `attention.diff_lambda_init_base` | `0.8` | Inherited | Sets the base term in differential attention's depth-dependent initial subtraction coefficient. Inactive in the default GQA run. |
| `attention.diff_lambda_decay_scale` | `0.6` | Inherited | Sets the size of the exponential depth correction in differential attention. Exposed from the existing reference initializer. |
| `attention.diff_lambda_decay_rate` | `0.3` | Inherited | Sets how rapidly the differential initialization correction changes with layer depth. Inactive under GQA. |
| `attention.diff_lambda_param_std` | `0.1` | Inherited | Sets initialization spread for learned differential lambda vectors, making a previously hidden experimental control explicit. |
| `attention.v_head_dim` | `OPEN` | `32` | Sets value channels per head independently of query/key width. Tiny uses 32; asymmetric value widths have reference tests. |
| `attention.kv_lora_rank` | `OPEN` | `16` | Sets MLA compressed KV latent width. Active only in MLA; a smaller latent reduces cache payload while constraining capacity. |
| `attention.q_lora_rank` | `OPEN` | `16` | Sets the MLA query-projection bottleneck width. This is internal attention compression, not instruction-tuning LoRA. |
| `attention.qk_nope_head_dim` | `OPEN` | `24` | Sets MLA content channels without rotary positions. Tiny allocates 24 of its 32 query/key channels here. |
| `attention.qk_rope_head_dim` | `OPEN` | `8` | Sets MLA rotary channels. Tiny uses eight, with content 24 plus rotary 8 matching QK width 32. |
| `attention.sliding_window_size` | `OPEN` | `16` | Sets the local attention lookback window in hybrid attention. Tiny uses 16; inactive for ordinary GQA. |
| `attention.global_attn_every_n_layers` | `2` | Inherited | Makes every second hybrid-attention layer global. Intervening layers use the local window; this does not affect GQA. |
| `attention.num_sink_tokens` | `0` | Inherited | Disables learned attention sink slots. Positive values add static learned per-head vectors, not tokenizer vocabulary tokens. |
| `attention.kv_sharing_group_size` | `1` | Inherited | One gives every layer its own KV owner. Larger supported groups reuse owner KV; hybrid local/global sharing is currently rejected. |
| `attention.partial_rotary_factor` | `1.0` | Inherited | Uses all ordinary QK channels for RoPE. MLA instead uses its explicit rotary-channel geometry. |
| `attention.mechanism` | `"gqa"` | Inherited | Selects GQA, where query heads share KV heads. Reference alternatives include MHA, MQA, MLA, hybrid-window and differential attention. |
| `attention.query_heads` | `OPEN` | `3` | Sets query-head count. Tiny uses three; head relationships and the current residual-width divisibility contract are validated. |
| `attention.kv_heads` | `OPEN` | `1` | Sets GQA key/value-head count. Tiny uses one KV head shared by three query heads, reducing KV representation size. |
| `attention.qk_head_dim` | `OPEN` | `32` | Sets query/key matching width per head. Tiny uses 32; rotary subdimensions must satisfy even-dimension constraints. |
| `attention.rope_theta` | `1000000.0` | Inherited | Sets the RoPE frequency base, not maximum context length. Curriculum stage theta overrides it during scheduled training. |
| `attention.use_qk_norm` | `true` | Inherited | Normalizes queries and keys before RoPE to control score magnitude. This is the runtime normalization switch. |
| `attention.attn_logit_softcapping` | `0.0` | Inherited | Zero disables capping. A positive C applies C times tanh(logits divided by C) before softmax. |
| `attention.qk_norm` | `true` | Inherited | Validated alias of use_qk_norm. Both values must agree; it is not an independent second normalization. |
| `attention.dropout` | `0.0` | Inherited | Zero means no attention dropout. The reference rejects nonzero values instead of pretending stochastic attention is implemented. |
| `attention.bias` | `false` | Inherited | Disables additive attention-projection biases. This is an enforced capability boundary in the current backbone. |

### RoPE scaling

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `attention.rope_scaling.type` | `"none"` | Inherited | Disables extra positional interpolation. The implemented alternative is linear; YaRN is not silently accepted. |
| `attention.rope_scaling.factor` | `1.0` | Inherited | One leaves positions unchanged. In linear mode, positions are divided by the factor before rotation. |
| `attention.rope_scaling.original_max_position_embeddings` | `OPEN` | `64` | Records original context size for extension reporting. Tiny uses 64; the value alone does not enforce a context limit. |

## 7 Experts routing and balancing

Tiny logically combines one shared expert with two of eight routed experts. Dense reference execution still computes every expert, so logical top-k capacity is not an optimized sparse-compute benchmark.

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `moe.dispatch_impl` | `"dense_reference"` | Inherited | Uses fixed-order dense expert execution followed by routing weights. Deterministic, but it computes all experts rather than optimized sparse compute. |
| `moe.aux_loss_scope` | `"batch"` | Inherited | Computes balancing statistics over the batch. Sequence scope is available as a controlled alternative with padding excluded. |
| `moe.moe_layer_freq` | `1` | Inherited | One places MoE at every eligible layer after the dense prefix. Larger values insert dense layers between routed layers. |
| `moe.n_group` | `1` | Inherited | One keeps routed experts in a single selection group. Multiple groups enable restricted group-first expert selection. |
| `moe.topk_group` | `1` | Inherited | Allows selection from the single default group. Must be compatible with group count and requested expert top-k. |
| `moe.activation` | `"swiglu"` | Inherited | Uses SwiGLU expert FFNs: a SiLU-gated branch multiplied by an up-projection, followed by a down-projection. |
| `moe.scoring_func` | `"softmax"` | Inherited | Uses softmax router scores. Sigmoid is a tested alternative; score entropy and top-k normalization need careful interpretation. |
| `moe.type` | `"standard"` | Inherited | Keeps experts in the standard hidden space. Latent mode adds compression/expansion around routed expert execution. |
| `moe.num_shared_experts` | `OPEN` | `1` | Alias of shared_experts. Both remain open for production and both must equal one in the tiny benchmark. |
| `moe.first_k_dense_replace` | `0` | Inherited | Zero keeps no forced dense prefix. Positive values replace early routed layers with dense FFNs; full depth expresses the dense control. |
| `moe.router_balance_mode` | `"switch_aux_loss"` | Inherited | Uses Switch-style balancing loss. Other modes use detached selection bias or a hybrid of bias and auxiliary loss. |
| `moe.bias_update_rate` | `0.001` | Inherited | Controls detached expert-selection bias updates in aux_loss_free_bias or hybrid mode. Inactive in the default pure auxiliary-loss mode. |
| `moe.train_capacity_factor` | `1.25` | Inherited | Sets finite training slot capacity relative to ideal expert load. Inactive while dropless_moe is true. |
| `moe.eval_capacity_factor` | `2.0` | Inherited | Sets finite evaluation capacity. Its larger default allows more assignments when dropping is enabled; inactive in dropless mode. |
| `moe.dropless_moe` | `true` | Inherited | Keeps all requested routed assignments instead of discarding overflow slots. This is the actual capacity-policy switch. |
| `moe.num_experts` | `OPEN` | `8` | Sets total routed expert count. Production is open; tiny uses eight, increasing total capacity beyond the two selected experts. |
| `moe.num_experts_per_tok` | `OPEN` | `2` | Sets routed experts selected per valid token. Tiny uses two; shared experts are additional contributions. |
| `moe.shared_experts` | `OPEN` | `1` | Sets experts always contributing to valid tokens. Tiny uses one, independent of top-two routed selection. |
| `moe.moe_intermediate_size` | `OPEN` | `192` | Sets each routed expert's FFN width. Tiny uses 192; it is independent of the dense FFN width field. |
| `moe.shared_intermediate_size` | `OPEN` | `192` | Sets each shared expert's intermediate width. Tiny uses 192; shared compute must be included in performance comparisons. |
| `moe.latent_size` | `OPEN` | `48` | Sets compressed routed-expert width only in latent MoE. Tiny defines 48 but standard mode does not use it. |
| `moe.norm_topk_prob` | `true` | Inherited | Renormalizes kept expert weights to sum to one per token, stabilizing routed contribution scale as selected mass changes. |
| `moe.routed_scaling_factor` | `1.0` | Inherited | One preserves the routed contribution scale before adding shared-expert outputs; it supports isolated scale ablations. |
| `moe.aux_coefficient` | `0.01` | Inherited | Sets default balancing-loss weight when curriculum is disabled. Enabled curriculum supplies its scheduled coefficient instead. |
| `moe.router_z_coefficient` | `0.001` | Inherited | Weights the router log-partition penalty independently of balancing loss, helping constrain excessively large router logits. |
| `moe.dropless` | `true` | Inherited | Validated alias of dropless_moe. Keep them synchronized; it does not add a second capacity mechanism. |

## 8 Vocabulary projection and output loss

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `output.factorized_chunk_vocab` | `2048` | Inherited | Sets vocabulary construction chunks for the lookup-based factorized head. Inactive under the default untied dense head. |
| `output.final_logit_softcapping` | `0.0` | Inherited | Zero leaves output logits uncapped. Positive values apply a smooth bound before cross-entropy and z-loss calculation. |
| `output.loss_chunk_tokens` | `16` | Inherited | Projects and scores 16 valid token positions at a time to bound temporary vocabulary-logit memory; sequence length is unchanged. |
| `output.logit_z_coefficient` | `0.0001` | Inherited | Weights output-logit z-loss. This stabilizer is separate from the MoE router z-loss and excluded from validation PPL/BPB. |

## 9 Parameter updates and width scaling

The instructor-directed hybrid assigns eligible internal2D matrices to Muon and embeddings, head, router, norms and remaining parameters to AdamW. Every trainable parameter belongs to one group; norm/router decay is excluded. Width hooks remain experimental.

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `optimizer.router_lr_mult` | `1.0` | Inherited | One keeps router LR at its AdamW base. Separate tuning can address routing changes without moving other groups' rates. |
| `optimizer.ke2_lr_mult` | `1.0` | Inherited | One keeps embedding-group LR unchanged. Allows separate embedding-projection or adapter adaptation experiments. |
| `optimizer.ke2_wd` | `0.0` | Inherited | Disables decay for embedding-group parameters. This group includes reference embedding projections and applicable adapters, not guaranteed G 4 factors. |
| `optimizer.mup_enabled` | `false` | Inherited | Disables experimental width-transfer hooks until coordinate and multi-scale transfer experiments justify them. |
| `optimizer.mup_base_width` | `96` | Inherited | Uses the tiny hidden width 96 as the transfer anchor. This is not a production-width claim or a complete muP base-shape specification. |
| `optimizer.mup_attention_scaling` | `"inverse_head_dim"` | Inherited | Declares inverse-head-dimension attention scaling when muP hooks are enabled, replacing ordinary inverse-square-root scaling. |
| `optimizer.mup_output_mult` | `1.0` | Inherited | Provides a multiplier within the width-scaled readout rule. Inactive while muP hooks are disabled. |
| `optimizer.optimizer_type` | `"muon_adamw_hybrid"` | Inherited | Uses Muon on eligible internal matrices and AdamW on embeddings, head, router, norms and remaining parameters; retains instructor-required gains. |
| `optimizer.embedding_weight_decay` | `0.0` | Inherited | Validated alias of ke 2_wd. Both values must agree so embedding decay cannot silently differ between naming conventions. |

### AdamW

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `optimizer.adamw.learning_rate` | `0.0003` | Inherited | Peak/base AdamW rate before scheduler and role multipliers. The 0.0003 value is a tested engineering control, not a production optimum. |
| `optimizer.adamw.beta1` | `0.9` | Inherited | Smooths the gradient first moment with coefficient 0.9, balancing responsiveness and update history. |
| `optimizer.adamw.beta2` | `0.999` | Inherited | Smooths squared-gradient estimates with coefficient 0.999. Changing it alters how quickly adaptive scaling responds. |
| `optimizer.adamw.eps` | `1e-08` | Inherited | Adds a 1 e-8 numerical stabilizer in AdamW's adaptive denominator. Distinct from normalization epsilon. |
| `optimizer.adamw.weight_decay` | `0.01` | Inherited | Applies 0.01 decay only to eligible AdamW groups; norm, router and embedding exclusions retain their separate policies. |

### Muon

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `optimizer.muon.learning_rate` | `0.002` | Inherited | Peak/base Muon rate before schedule and width multipliers. Its scale should not be compared directly with AdamW's rate. |
| `optimizer.muon.momentum` | `0.95` | Inherited | Uses 0.95 momentum to smooth matrix update directions across optimizer steps. |
| `optimizer.muon.nesterov` | `true` | Inherited | Enables the reference Nesterov-style momentum update before the matrix transformation. |
| `optimizer.muon.ns_steps` | `5` | Inherited | Uses five Newton-Schulz iterations in the FP 32 matrix update transformation, trading optimizer cost against approximation effort. |
| `optimizer.muon.eps` | `1e-07` | Inherited | Stabilizes normalization of the Muon update with 1 e-7. The reference is not a fused or sharded optimizer. |
| `optimizer.muon.weight_decay` | `0.0` | Inherited | Disables decay for Muon-assigned matrices in this baseline; changes should be evaluated separately from AdamW decay. |

## 10 Training execution

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `training.seed` | `1234` | Inherited | Sets the required 1234 initialization/sampling seed. Exact repeats also require matching inputs, code, hardware/software and deterministic operations. |
| `training.steps` | `30` | Inherited | Runs 30 optimizer updates for the engineering gate. This is not an adequate budget for complete model pretraining. |
| `training.sequence_length` | `OPEN` | `64` | Sets the maximum/reference evaluation window length; active training lengths come from curriculum. Tiny resolves this to 64. |
| `training.batch_size` | `2` | Inherited | Requests two document windows per microbatch. Accumulation and final-target trimming determine the actual valid-token update budget. |
| `training.gradient_clip` | `1.0` | Inherited | Clips the global gradient norm to 1 when needed, limiting unusually large optimizer updates without adding a loss term. |
| `training.warmup_steps` | `3` | `3` | Warms learning rates over the first three updates. It must fit inside the first WSD curriculum stage. |
| `training.flush_every` | `10` | Inherited | Flushes buffered metric/loss files every ten updates. Records are still generated at every update. |

## 11 Dataset ingestion and accounting

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `data.dataset_path` | `"fixture"` | Inherited | Uses a built-in deterministic fixture without external downloads. Real training selects a local dataset path or --data-path. |
| `data.validation_dataset_path` | `"auto_split"` | Inherited | Uses deterministic document-hash splitting for real data unless an explicit holdout path is provided. Fixture assignment is explicit. |
| `data.slice_documents` | `64` | Inherited | Limits ingestion to 64 documents in deterministic-slice mode, enabling quick repeatable checks; full mode removes that document cap. |
| `data.mode` | `"deterministic_slice"` | Inherited | Selects bounded deterministic-slice ingestion. Full mode loads all supported documents and is a separate real-data execution choice. |
| `data.bpb_denominator` | `"exact_scored_token_bytes"` | Inherited | Uses exact bytes represented by scored text tokens, keeping BPB grounded in text rather than token count or display-string guesses. |
| `data.document_attention` | `"isolated"` | Inherited | Requires isolated document windows, avoiding unintended attention across unrelated documents. |
| `data.loss_normalization` | `"valid_tokens"` | Inherited | Normalizes summed CE by valid scored targets, not padded array size, making variable document lengths comparable. |

## 12 Parallelism and memory contracts

The runtime is single-process FP32, with every parallel degree equal to one and ZeRO stage0. Parallelism and stage3 fields describe future ownership/precision contracts; changing numbers alone cannot create DDP or DeepSpeed execution.

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `distributed.dp_size` | `OPEN` | `1` | Declares data-parallel degree; tiny uses one. The current trainer rejects nontrivial parallel sizes rather than spawning workers. |
| `distributed.ep_size` | `OPEN` | `1` | Declares expert-parallel degree; tiny uses one. The expert count must divide by EP when both values are known. |
| `distributed.tp_size` | `OPEN` | `1` | Declares tensor-parallel degree for future split-matrix execution. The current single-process runtime requires one. |
| `distributed.cp_size` | `OPEN` | `1` | Declares context-parallel degree for future sequence partitioning. The current runtime requires one. |
| `distributed.activation_checkpointing` | `false` | Inherited | Disables selective expert-FFN recomputation by default. Supported checkpointing trades backward compute for fewer saved activations; not midpoint reversibility. |
| `distributed.zero_stage` | `0` | `0` | Selects no ZeRO sharding. Stage 3 is only a future validated contract and is rejected by this trainer. |
| `distributed.moe_sharding_strategy` | `"expert_data_parallel"` | Inherited | Records the intended expert/data-parallel ownership strategy. A string does not implement collectives or process groups. |
| `distributed.zero3_shard_dense_params` | `true` | Inherited | Requires future stage 3 integration to shard dense parameters. Inactive in the stage 0 tiny run. |
| `distributed.keep_router_in_fp32_under_zero` | `true` | Inherited | Requires FP 32 router handling under a future ZeRO 3 backend; enforced for the declared contract. |
| `distributed.stage3_param_persistence_threshold` | `OPEN` | `0` | Sets a future persistence threshold in parameter elements. Tiny uses zero and no stage 3 runtime is active. |
| `distributed.reduce_bucket_size_elements` | `OPEN` | `100000` | Sizes future reduction buckets in elements, not bytes. Tiny defines 100000 but stage 0 does not use it. |
| `distributed.stage3_prefetch_bucket_size_elements` | `OPEN` | `100000` | Sizes future parameter-prefetch buckets in elements. Tiny defines 100000; inactive without stage 3 integration. |
| `distributed.overlap_comm` | `false` | `false` | Disables communication/compute overlap for the reference. No overlapping collectives are implemented in this single-process trainer. |

## 13 Curriculum and token budgets

The budget counts valid scored targets, not padded slots. Tiny scores18 targets per update although two64-position windows allocate128 slots. Accumulation and final-target masking meet the exact target budget. Ramps interpolate mixture weights, temperature and auxiliary coefficient, not sequence length.

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `data_curriculum.packing_mode` | `"document_isolated"` | Inherited | Uses isolated document windows with exact byte accounting. General multi-document bin packing is not implemented. |
| `data_curriculum.fim_rate` | `0.0` | Inherited | Disables fill-in-the-middle transforms. Positive rates are rejected until marker IDs and transformation rules are implemented. |
| `data_curriculum.stage_rng_seed_stride` | `1000` | Inherited | Offsets the private sampler seed by 1000 per stage, allowing deterministic stage-specific permutations without consuming model RNG. |
| `data_curriculum.global_tokens_per_step` | `OPEN` | `18` | Sets exact valid targets per optimizer update. Production is open; tiny uses 18, irrespective of padded batch slots. |
| `data_curriculum.enforce_token_batch_invariance` | `true` | Inherited | Requires the valid-target budget to stay constant when context/window lengths change; accumulation and final masking enforce it. |
| `data_curriculum.transition_ramp_steps` | `2` | Inherited | Interpolates selected data/router controls over two boundary steps, rather than applying an immediate coefficient/temperature shock. |
| `data_curriculum.transition_curve` | `"cosine"` | Inherited | Uses cosine interpolation for mixture weights, router temperature and auxiliary coefficient. Sequence lengths are not continuously interpolated. |
| `data_curriculum.reset_position_ids_on_eos` | `true` | Inherited | Requires document-boundary position resets so independent texts do not inherit an arbitrary continuous RoPE position offset. |
| `data_curriculum.intra_document_causal_mask` | `true` | Inherited | Requires causal attention within the same document; prevents earlier unrelated documents from becoming unintended context. |
| `data_curriculum.weighting_mode` | `"static"` | Inherited | Uses static configured sampling weights. Online excess-loss mode is an optional feedback hook, not a full DoReMi reproduction. |
| `data_curriculum.min_domain_weight` | `0.05` | Inherited | Applies a 0.05 sampling-probability floor per configured slice. Labels are arbitrary; probabilities are not exact token fractions. |
| `data_curriculum.online_weight_lr` | `0.1` | Inherited | Controls exponentiated excess-loss feedback strength. Inactive in static mode and requires calibrated validation references when enabled. |
| `data_curriculum.domain_labels` | `["slice_0"]` | Inherited | Names the single arbitrary slice_0 source. Replaceable labels avoid hardcoding code, Indic or any required semantic domain. |
| `data_curriculum.reference_losses` | `[0.0]` | Inherited | Defines per-slice feedback baselines. Zero is only a static-mode placeholder; online mixing requires calibrated positive reference losses. |
| `data_curriculum.enabled` | `true` | Inherited | Activates the three-stage curriculum. Disabling it uses static training sequence length, attention theta and MoE auxiliary defaults. |

### Stage 1 warmup

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `data_curriculum.stage_1_warmup.mixture_weights` | `[1.0]` | Inherited | Samples the sole configured slice with probability one before floor adjustment. Weight order must match domain_labels; multi-slice weights must sum to one. |
| `data_curriculum.stage_1_warmup.router_temperature` | `1.0` | Inherited | Uses unchanged router score temperature 1. Values above one soften score differences; stage-boundary ramps interpolate this control. |
| `data_curriculum.stage_1_warmup.step_fraction` | `0.1` | Inherited | Allocates 10 percent of optimizer updates to stage 1. All three fractions must sum to one. |
| `data_curriculum.stage_1_warmup.seq_len` | `OPEN` | `64` | Leaves production context length open. Tiny keeps 64 in every stage to isolate reproducibility instead of claiming context-extension quality gains. |
| `data_curriculum.stage_1_warmup.rope_theta` | `10000.0` | `1000000.0` | Uses this stage-specific RoPE frequency base during training. Tiny holds all stages at 1000000; a large theta does not establish long-context capability. |
| `data_curriculum.stage_1_warmup.moe_aux_loss_coeff` | `0.03` | `0.01` | Sets the target balancing-loss weight for this stage, overriding moe.aux_coefficient while curriculum is enabled. Boundary interpolation smooths changes. |

### Stage 2 main

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `data_curriculum.stage_2_main.mixture_weights` | `[1.0]` | Inherited | Samples the sole configured slice with probability one before floor adjustment. Weight order must match domain_labels; multi-slice weights must sum to one. |
| `data_curriculum.stage_2_main.router_temperature` | `1.0` | Inherited | Uses unchanged router score temperature 1. Values above one soften score differences; stage-boundary ramps interpolate this control. |
| `data_curriculum.stage_2_main.step_fraction` | `0.7` | Inherited | Allocates 70 percent of optimizer updates to stage 2. All three fractions must sum to one. |
| `data_curriculum.stage_2_main.seq_len` | `OPEN` | `64` | Leaves production context length open. Tiny keeps 64 in every stage to isolate reproducibility instead of claiming context-extension quality gains. |
| `data_curriculum.stage_2_main.rope_theta` | `100000.0` | `1000000.0` | Uses this stage-specific RoPE frequency base during training. Tiny holds all stages at 1000000; a large theta does not establish long-context capability. |
| `data_curriculum.stage_2_main.moe_aux_loss_coeff` | `0.01` | `0.005` | Sets the target balancing-loss weight for this stage, overriding moe.aux_coefficient while curriculum is enabled. Boundary interpolation smooths changes. |

### Stage 3 annealing

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `data_curriculum.stage_3_anneal.mixture_weights` | `[1.0]` | Inherited | Samples the sole configured slice with probability one before floor adjustment. Weight order must match domain_labels; multi-slice weights must sum to one. |
| `data_curriculum.stage_3_anneal.router_temperature` | `1.0` | Inherited | Uses unchanged router score temperature 1. Values above one soften score differences; stage-boundary ramps interpolate this control. |
| `data_curriculum.stage_3_anneal.step_fraction` | `0.2` | Inherited | Allocates 20 percent of optimizer updates to stage 3. All three fractions must sum to one. |
| `data_curriculum.stage_3_anneal.seq_len` | `OPEN` | `64` | Leaves production context length open. Tiny keeps 64 in every stage to isolate reproducibility instead of claiming context-extension quality gains. |
| `data_curriculum.stage_3_anneal.rope_theta` | `1000000.0` | `1000000.0` | Uses this stage-specific RoPE frequency base during training. Tiny holds all stages at 1000000; a large theta does not establish long-context capability. |
| `data_curriculum.stage_3_anneal.moe_aux_loss_coeff` | `0.001` | `0.001` | Sets the target balancing-loss weight for this stage, overriding moe.aux_coefficient while curriculum is enabled. Boundary interpolation smooths changes. |

## 14 Native diagnostics and SAE hooks

Detached activation hooks isolate Python, NumPy and PyTorch RNG effects. Measurements help diagnose training but do not prove causal explanations of reasoning. The random Top-K probe is untrained and disabled by default.

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `explainability.enabled` | `true` | Inherited | Enables native detached telemetry with isolated RNGs. Diagnostics are tested not to perturb the main training loss trajectory. |
| `explainability.track_perplexity_and_bpb` | `true` | Inherited | Requires first-class PPL/BPB reporting. These are mandatory metric policies, not optional decorative visualization switches. |
| `explainability.track_residual_cosine_sim` | `true` | Inherited | Measures how much each block changes residual direction using input/output cosine similarity; a diagnostic rather than causal proof. |
| `explainability.track_attention_entropy` | `true` | Inherited | Reports per-head concentration of attention probabilities. Low entropy alone does not establish collapse or an error. |
| `explainability.track_moe_router_metrics` | `true` | Inherited | Keeps expert load, entropy, dropped-assignment, dead-expert and router stability diagnostics needed for routing analysis and monitoring. |
| `explainability.enable_logit_lens` | `true` | Inherited | Reads intermediate states through final norm/head to measure CE and drift toward final predictions without adding a training objective. |
| `explainability.sae_tap_points` | `["resid_mid", "mlp_out", "moe_router_logits"]` | Inherited | Captures residual-after-attention, FFN/MoE output and raw router logits through detached native hooks, without external XAI packages. |
| `explainability.logit_lens_every_steps` | `10` | Inherited | Runs the lens at step 1 and every tenth update, bounding expensive intermediate vocabulary readout overhead. |
| `explainability.tap_every_steps` | `10` | Inherited | Runs configured telemetry/taps at step 1 and every tenth update; observations are cadence-limited rather than continuous monitoring. |

### SAE probe

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `explainability.sae_probe.enabled` | `false` | Inherited | Disables sparse-autoencoder probing by default. Other XAI measurements and activation taps remain available. |
| `explainability.sae_probe.kind` | `"random_topk_reference"` | Inherited | Selects an untrained random Top-K plumbing baseline if enabled. Trained mode needs a compatible checkpoint; random features are not interpreted concepts. |
| `explainability.sae_probe.weights_path` | `"none"` | Inherited | Supplies no trained SAE artifact. Trained mode requires an explicit valid file and compatible tap dimensions. |
| `explainability.sae_probe.seed` | `1729` | Inherited | Uses a private 1729 seed plus tap width for random probe initialization, isolated from the main training seed. |
| `explainability.sae_probe.feature_count` | `64` | Inherited | Creates 64 optional probe features, keeping the untrained diagnostic inexpensive relative to full trained SAE work. |
| `explainability.sae_probe.top_k` | `8` | Inherited | Retains up to eight nonzero sparse features per token; ReLU can leave fewer than eight active. |
| `explainability.sae_probe.dead_feature_window` | `100` | Inherited | Tracks inactivity across 100 per-tap observations. This is not necessarily 100 optimizer updates, especially with sparse cadence or microbatches. |
| `explainability.sae_probe.dead_feature_fraction_threshold` | `0.95` | Inherited | Sets a provisional warning when at least 95 percent of features are inactive after a populated observation window; requires probe-specific calibration. |

## 15 Language model head and future token loss

Despite this parent name, provider dimensions live in embedding.ke2. The default untied head alone has131072 times96 weights, or12582912 parameters. Lookup-based head alternatives are correctness references, not fused G4 decoding kernels.

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `embeddings_ke2.lm_head_mode` | `"untied_dense"` | Inherited | Uses a separate dense vocabulary matrix. Tied and chunked lookup-based alternatives exist; the latter is not a fused G 4 factorized decoder. |
| `embeddings_ke2.mtp_num_heads` | `0` | Inherited | Disables direct future-token prediction transforms. Extra heads are an auxiliary experiment rather than the default objective. |
| `embeddings_ke2.mtp_loss_weight` | `0.1` | Inherited | Weights optional MTP loss by 0.1. It contributes nothing with zero MTP heads and is excluded from validation perplexity. |

## 16 Learning rate schedule

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `scheduler.type` | `"wsd"` | Inherited | Uses warmup-stable-decay learning rates, coupled to curriculum stage fractions; a constant schedule is also supported. |
| `scheduler.min_lr_ratio` | `0.1` | Inherited | Sets final LR to ten percent of each group's peak after decay. Role multipliers still apply. |
| `scheduler.decay_style` | `"cosine"` | Inherited | Uses cosine annealing in the last stage. The sqrt_1_minus_t curve is a controlled reference alternative. |

## 17 Numerical precision

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `precision.param_dtype` | `"float32"` | Inherited | Keeps trainable parameters in FP 32 as the numerical reference. Arbitrary lower-precision settings are rejected. |
| `precision.compute_dtype` | `"float32"` | Inherited | Uses FP 32 model arithmetic. BF 16/FP 8 require implementation and independent numerical/repeatability validation. |
| `precision.grad_reduce_dtype` | `"float32"` | Inherited | Records FP 32 reduction precision for a future distributed backend. No gradient collective runs in the current trainer. |
| `precision.router_dtype` | `"float32"` | Inherited | Requires FP 32 router parameters/scores to retain a stable routing reference. |
| `precision.loss_dtype` | `"float32"` | Inherited | Requires FP 32 logits and loss calculation for accurate CE and stable training measurements. |
| `precision.allow_tf32` | `false` | Inherited | Disables TF 32 paths for the controlled same-environment reference. It is not a guarantee of cross-device floating-point identity. |

## 18 Checkpoint state and retention

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `checkpointing.save_every_steps` | `0` | Inherited | Zero disables periodic checkpoint saves. A stop-after run still saves; a default completed run is not guaranteed a final checkpoint. |
| `checkpointing.keep_last_n` | `1` | Inherited | Retains one numbered checkpoint version plus the latest alias, bounding checkpoint storage from periodic saves. |
| `checkpointing.save_optimizer_state` | `true` | Inherited | Saves moment/adaptive-update state needed to reproduce uninterrupted optimization, not merely restart from model weights. |
| `checkpointing.save_rng_and_dataloader_state` | `true` | Inherited | Saves RNGs and sampler/cursor state so resumed runs receive the same future batches and random sequence. |

## 19 Heldout evaluation

Heldout evaluation uses eval mode and no gradients, with fixed windows and a consistent final-stage theta/temperature definition. Train measurements, fixed training probes and heldout metrics must remain separately labeled.

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `evaluation.eval_every_steps` | `5` | `1` | Runs heldout evaluation every five updates; tiny overrides this to every update for detailed gate evidence. |
| `evaluation.eval_batches` | `32` | Inherited | Caps fixed evaluation/probe windows using a 32-batch budget. A tiny corpus can contain fewer windows than this cap. |
| `evaluation.exclude_aux_loss_from_val` | `true` | Inherited | Keeps heldout metrics CE-only; balancing, router/output z-loss and MTP must not contaminate PPL/BPB interpretation. |
| `evaluation.compute_token_ppl` | `true` | Inherited | Emits token perplexity as exp of heldout CE. Tokenizer differences affect its comparability. |
| `evaluation.compute_bpb` | `true` | Inherited | Emits heldout NLL per exact scored UTF-8 byte in bits, alongside token-normalized perplexity. |

## 20 Training health diagnostics

Thresholds are provisional warning rules requiring G8 calibration. Fatal NaN/Inf checks are separate. A30-step gate cannot naturally populate a100-update sustained-imbalance rule; synthetic tests cover it.

| Setting | Base value | Tiny override | Purpose and importance |
|---|---|---|---|
| `monitoring.enabled` | `true` | Inherited | Enables native deterministic warning diagnostics. External G 8 sink integration and operational threshold calibration remain separate work. |
| `monitoring.history_window` | `100` | Inherited | Retains 100 prior loss observations for rolling diagnostics. It is a provisional baseline, not a statistical calibration result. |
| `monitoring.warmup_steps` | `10` | Inherited | Waits for ten prior observations before interpreting deviation scores. Distinct from the three-step learning-rate warmup. |
| `monitoring.spike_sigma` | `7.0` | Inherited | Warns on large upward prior-history standardized loss deviations above 7. This does not imply a guaranteed seven-sigma false-alarm probability. |
| `monitoring.ema_alpha` | `0.97` | Inherited | Gives 97 percent weight to the prior EMA estimate and 3 percent to the new observation, smoothing monitoring history. |
| `monitoring.ema_z_threshold` | `2.5` | Inherited | Sets a provisional 2.5 EMA-deviation warning bar after warmup; requires G 8 calibration on realistic/synthetic streams. |
| `monitoring.max_attn_logit` | `100.0` | Inherited | Warns when sampled scaled attention logits exceed absolute magnitude 100 before softcapping. Unsampled transients may be missed. |
| `monitoring.max_load_violation` | `1.0` | Inherited | Sets the imbalance bar at 1, meaning maximum expert load is about twice ideal load; sustained patience is applied. |
| `monitoring.imbalance_patience` | `100` | Inherited | Requires 100 consecutive populated above-threshold updates for a sustained imbalance warning; a 30-step run cannot naturally fill it. |

## How the settings work together

Text is tokenized into pinned IDs, mapped through the Kronecker reference, processed by attention and dense/shared/routed FFNs, normalized and projected into vocabulary logits. CE trains next-token prediction; separate auxiliary objectives influence routing or stability.

With curriculum active, stage seq_len and rope_theta control training, and stage moe_aux_loss_coeff overrides the static MoE coefficient. Fixed heldout evaluation uses the final-stage theta and temperature definition. Tiny deliberately keeps theta and context length constant.

The three fractions 0.1,0.7,0.2 allocate a 30-update run into steps 1 to 3,4 to 24, and 25 to 30. WSD warms for 3 updates, stays at peak through 24, then decays toward 10 percent of peak. Stage-boundary interpolation changes selected controls over two steps.

Tiny uses hidden 96, two layers, three query heads sharing one KV head, QK/value width 32, eight routed experts with top-two selection, one shared expert, expert widths 192, FP 32, batch_size 2 and 18 valid targets per update. It sees 540 scored targets over 30 updates, not a full pretraining corpus.

CE is summed negative log likelihood divided by valid targets. Perplexity is exp of CE. BPB divides summed negative log likelihood by ln 2 times the exact scored UTF-8 byte count. Heldout versions exclude every auxiliary objective.

An ordinary tiny untied head contains 131072 times 96 weights, or 12582912 parameters, explaining why a two-layer width 96 network totals about 14.4 M parameters.

For reconstruction comparisons, acceptable error is at most atol plus rtol times the absolute reference. The reversible tolerances are proposed acceptance bars; no reversible reconstruction result has been measured.

Checkpoint resume needs model, optimizer, scheduler, curriculum, router buffers, sampler, RNG and monitoring history. Model weights alone cannot generally reproduce the uninterrupted trajectory.

## Pending work and decision tracking

Await actual G 4 KE 2 shape/provider and TSAI production dimensions. Midpoint equations/inverse/backward, ZeRO 3/parallel execution, mixed precision and external G 8 logger integration remain pending. Config contracts do not implement these capabilities.

Each future change should record the exact key, old/new value, reason or hypothesis, source or owner, experiment identifier, token budget, metric/pass bar, actual result and approval status. Keep provisional defaults separate from measured winners.

For example, changing optimizer.adamw.beta 2 from 0.999 to 0.95 is an ablation proposal, not a recommendation established by the fixture. Compare equal token budgets and fixed real holdout windows on paired seeds before adoption.

The provenance for each key is its inline source comment in the YAML. Instructor requirements, G 4/G 7/G 8 contracts, artifact identities and G 5 engineering hypotheses are different evidence classes. Do not relabel provisional control values as instructor-approved production choices.

## Change record template

| Date | Key | Old value | Proposed value | Why and source | Experiment and pass bar | Result | Decision |
|---|---|---|---|---|---|---|
| | | | | | | | |
