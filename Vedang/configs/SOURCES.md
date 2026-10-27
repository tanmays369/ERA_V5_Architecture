# Where config_v0.yaml values come from

Rule from the team lead: a source is the Kronecker embeddings code, its paper (arXiv 2605.29459), or a measurement from our
tiny run. `harness/check_config.py` enforces it: any other source word is an error, and a measurement source must point to a
real file in `measurements/` whose chosen value equals the default in the config.

| Source kind | Settings | How to verify |
|---|---|---|
| `kronecker-code` | tokenizer.name_or_path, embedding_type, kron_mode, kron_char_dim, kron_pos_dim | read `kronecker_embeddings/embedding.py` in the package or repo (defaults 256 and 32, modes dynamic/cached) |
| `kronecker-paper` | tie_embeddings = false | arXiv 2605.29459: the byte codec's width is not the model width, so it cannot share the output matrix |
| `measurement:tokenizer_vocab` | model.vocab_size | `python harness/prepare_data.py` writes it |
| `measurement:lr_sweep` | train.lr | `python harness/sweep.py --only lr_sweep` |
| `measurement:decay_sweep` | train.min_lr_ratio | `--only decay_sweep` |
| `measurement:wd_sweep` | train.weight_decay | `--only wd_sweep` |
| `measurement:clip_sweep` | train.grad_clip | `--only clip_sweep` |
| `measurement:init_sweep` | model.init_std | `--only init_sweep` |
| `measurement:topk_sweep` | moe.top_k | `--only topk_sweep` |
| `measurement:shared_sweep` | moe.num_shared_experts | `--only shared_sweep` |
| `measurement:layout_sweep` | moe.first_k_dense, moe.moe_layer_freq | `--only layout_sweep` |
| `measurement:router_sweep` | moe.router_type | `--only router_sweep` |
| `measurement:normalize_sweep` | moe.normalize_topk_weights | `--only normalize_sweep` |
| `measurement:balance_sweep` | moe.balance_method, moe.load_balance_coef | `--only balance_sweep` |
| `measurement:zloss_sweep` | moe.router_z_loss_coef | `--only zloss_sweep` |
| `measurement:mtp_sweep` | mtp.mtp_depth | `--only mtp_sweep` |
| `harness-code` | norm and activation choices, RoPE base, dropout, bias update rate, MTP loss weight and width, data and run-length settings, Adam betas, seed | read `harness/train.py`. These are fixed implementation choices or test-rig settings, not performance claims. The checker lists them so the team can confirm at the call that this counts as a source |
| `TBD_by_TSAI_31Oct` | the 8 sizes | TSAI gives them on 31 Oct |
