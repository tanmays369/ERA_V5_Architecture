Name: Revant Gupta · Pack: ind-hi-fw2

# MoE sweep: Kaggle 2× T4, 26 Oct 2026 (notebook versions 4 and 5)

Every arm ran 1000 steps, dp2, seed 1234, on the `t4_base` sizes: 4 layers, d_model 256, seq 256, global batch 16, about 9.0M-token 12-source mix.
In every MoE arm the active FFN width per token is 768, the same as dense d_ff: `top_k × d_ff + shared × d_ff`.
Commands are in `make_moe_nb.py`; each arm is `t4x2_dp2[_moe].yaml` plus `--set tokens.train_steps=1000` and the overrides below. Per-arm numbers and held-out curves are in `summary.json`, and each arm's `losses.tsv` is in its own folder.

| Arm | Overrides | Params | Held-out CE (final) | Train CE (last 100 steps) | ms/step | Steps with a dead expert | Top expert share (last 100 steps) |
|---|---|---|---|---|---|---|---|
| D0 dense | `t4x2_dp2` | 38.8M | 6.3144 | 5.612 | 282 | - | - |
| M1 E4 k2 | `t4x2_dp2_moe` | 41.2M | 6.3009 | 5.597 | 388 | 0 | 0.302 |
| M2 E8 k2 | `moe.num_experts=8` | 45.9M | 6.2849 | 5.573 | 505 | 0 | 0.177 |
| M3 E16 k4, d_ff 192 | `moe.num_experts=16 moe.top_k=4 model.d_ff=192` | 45.9M | 6.2991 | 5.596 | 755 | 0 | 0.092 |
| M4 E8 k1 + 1 shared | `moe.num_experts=8 moe.top_k=1 moe.shared_experts=1` | 47.1M | **6.2814** | **5.572** | 542 | 0 | 0.163 |
| M5 E8 k2, no balancing | `moe.num_experts=8 moe.bias_update_speed=0 moe.seq_aux_loss_coef=0` | 45.9M | 6.2871 | 5.581 | 522 | **192 of 1000** (up to 7 in one step, first at step 34) | 0.250 |

**What it shows**
- Every MoE arm trains stably next to dense: no NaN, and every arm passes the gate.
- With balancing on (aux-free bias + sequence aux), no expert ever receives zero tokens.
- With balancing off (M5), routing collapses early: dead experts appear from step 34, then recover by the end. Load stays more skewed than with balancing (top share 0.25 vs 0.18).
- **Determinism:** M2 rerun in the same session is byte-identical over 1000 steps. M2 is also byte-identical across two separate Kaggle sessions (versions 4 and 5).

**What it does not show**
- This is one seed with a 48-window held-out set, so gaps under about 0.03 nats are within noise [INFERENCE].
- MoE beats dense by 0.015–0.033 nats, M4 ≥ M2 > M3 ≈ M1. These are directions only. `experiments/plan_v0.md` re-tests them with 2 seeds and a measured noise floor.
- Step time mostly reflects the Python per-expert loop on T4 in fp32, not the cost of the architecture on H100.
- `bpb_token_weighted` in `summary.json` uses the old token-weighted bytes per token, which the harness has since fixed to equal-weight. Compare arms by CE.
