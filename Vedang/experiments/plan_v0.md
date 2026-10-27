# Team 05 - Experiment plan v0.3

Status: draft for TSAI approval on Sat 31 Oct. Config freeze is 14 Nov.

## How to read this plan

- Every arm runs on the harness (`harness/train.py`, `configs/small_colab.yaml`) on a GPU: a tiny MoE written from scratch,
  Kronecker input embeddings, BrahmicTokenizer-131K, a Hindi + English token file from `harness/prepare_data.py`.
  Standard embeddings are not tested (team lead: Kronecker is the KE path for this course).
- Results are **directional**: they choose what goes into dry run 1, they do not prove final-model quality.
- Metric names are those printed by `train.py --list-metrics`. Primary metric is `val_loss` at the final step.
- **Two layers of evidence.** (1) `harness/sweep.py` runs short CPU sweeps and writes the `measurements/` files that
  `config_v0.yaml` cites as sources; these cost 0 GPU hours. (2) The arms below confirm the important choices on a GPU with
  3 to 5 seeds each. An arm that contradicts a measurement means the config default changes before the 14 Nov freeze.
- **Noise first.** Arm A0 gives sigma = standard deviation of final `val_loss` over 5 seeds. Every "better than" bar is in
  units of sigma, so we do not chase noise.
- **GPU hours are provisional.** Assumption: one `small_colab` run = 0.10 GPU-h. Replace it with the measured `wall_s` from
  the RESULT line after the first GPU run, then recompute. Cost of an arm = runs x 0.10 GPU-h.
- Each arm changes one setting from the baseline (E=8, top_k=2, no shared expert, MoE in every layer, softmax router,
  aux_loss 0.01, z-loss 0.001, mtp_depth 1) and uses 3 seeds (1234, 1235, 1236) unless stated.

## Gate 0 (must pass before any arm; ~0 GPU-h)

| Check | Pass bar |
|---|---|
| Two runs, seed 1234, `diff` of the CSV logs (not the `.system.csv`) | empty diff on CPU, and on the GPU |
| Loss falls | `loss_ntp` last-10-step mean at least 10% below first-10-step mean; no NaN |
| Step-1 `loss_ntp` | within 0.3 nats of ln(vocab) = 11.78 (confirms labels and vocabulary are wired correctly) |
| `check_config.py` on config_v0 and both small configs | 0 errors (`--strict` on the small configs once measurements exist) |

## Arms

| Arm | What changes (`--set ...`) | Metric | Pass bar | Runs | GPU-h |
|---|---|---|---|---|---|
| A0 | Baseline MoE, seeds 1234-1238 | `val_loss` spread | Defines sigma. Valid if sigma < 0.05 nats; if larger, raise `train.steps` and rerun | 5 | 0.50 |
| A1 | `moe.dense_baseline=true` (dense FFN as wide as the active experts) | `val_loss` | MoE (A0) lower than dense by at least 2 sigma, else MoE is not paying for itself at this scale | 3 | 0.30 |
| A2 | `moe.top_k=1` | `val_loss`, `expert_load_cv` | top-2 beats top-1 by at least 2 sigma to justify the extra compute; otherwise adopt top-1 | 3 | 0.30 |
| A3 | `moe.num_experts=4` and `=16` (expert width fixed) | `val_loss`, `router_entropy` | Choose the smallest count within 1 sigma of the best; reject any count whose `router_entropy` falls below 50% of ln(num_experts) | 6 | 0.60 |
| A4 | Balancing: `moe.balance_method=bias`; `=none`; `moe.load_balance_coef=0.001` and `=0.1` | `expert_load_max_frac`, `val_loss` | Keep a method only if `expert_load_max_frac` <= 2/num_experts and `val_loss` is within 1 sigma of the best; `none` is expected to fail the load test | 12 | 1.20 |
| A5 | `moe.num_shared_experts=1` | `val_loss` | Adopt only if better than A0 by at least 2 sigma | 3 | 0.30 |
| A6 | `moe.router_z_loss_coef=0` | `val_loss`, `grad_norm` | Keep z-loss if removing it raises `val_loss` by at least 1 sigma or max `grad_norm` by at least 2x; otherwise drop it | 3 | 0.30 |
| A7 | Layout: `moe.first_k_dense=1`; and `moe.moe_layer_freq=2` | `val_loss`, active parameters | Adopt only if within 1 sigma of A0 at fewer active parameters, or better by 2 sigma | 6 | 0.60 |
| A8 | `moe.router_type=sigmoid` | `val_loss`, `router_entropy` | Adopt only if better than A0 by at least 2 sigma and `expert_load_max_frac` stays <= 2/num_experts | 3 | 0.30 |
| A9 | `mtp.mtp_depth=0` and `=2` | `val_loss` (next-token only), `loss_mtp` | Keep MTP only if depth 1 or 2 improves `val_loss` over depth 0 by at least 1 sigma; otherwise turn it off to save the extra output-layer pass | 6 | 0.60 |
| | **Subtotal** | | | 50 | **5.00** |
| | Reserve for reruns (20%) | | | | 1.00 |
| | **Total** | | | | **6.00** |

A3 and A5-A9 use sizes tied to `small_colab.yaml` (expert width 512); recompute them if TSAI's sizes differ.

## Decision rule

1. Run Gate 0, then A0 (it sets sigma), then A1.
2. If A1 fails (MoE does not beat dense), stop and report to TSAI before spending the rest.
3. Otherwise run A2-A9 in any order. Winners go into the config candidate for the 14 Nov freeze, and the matching
   `measurements/` file is rerun on the GPU config so the config source stays true.
4. If the budget shrinks, cut in this order: A8, A7, A6, A5, A3 (16 experts), A2.

## Known limits

- With expert weights initialised at 0.02 and a tiny hidden size, the experts add very little to the output early in
  training. In my 40-step test with a toy tokenizer, top-k, router type and z-loss changed the loss by under 0.01, while
  learning rate and init changed it a lot. So short CPU sweeps are reliable for learning rate, init, decay, weight decay
  and clipping, and weak for MoE settings: run those sweeps on `small_colab.yaml` (600 steps) or with `--seeds`.
- The model has a 131,072-token output layer, so with tiny hidden sizes most parameters sit outside the MoE layers.
  Active-parameter counts are printed at the start of each run; compare them across arms.
- Experts are computed densely in the harness, so tokens/second here says little about real MoE speed.
- Wikipedia text is cleaner than the final data mix. The frozen config is retested in dry run 1.
- Sizes are placeholders until TSAI gives them on 31 Oct.
