# Gate-1 sweep, 27 Oct 2026

Run: `python verify/gate1.py --all ..` against the working repo at `a5172b3`.
The tool does not execute any code from the folders it checks.

```

=== Dharshan ====================================================
  [PASS] config_small.yaml loads and every setting has a value and a source  74 settings, 0 errors, 0 declared open
  [PASS] config_v0.yaml loads and every setting has a value and a source     74 settings, 0 errors, 0 declared open
  [FAIL] a pair of same-seed runs is present                                 0 loss file(s), no a/b pair found
  [note] config_small.yaml: sources are in README.md, not per-line comments - this tool checks values only for that file
  [note] config_v0.yaml: sources are in README.md, not per-line comments - this tool checks values only for that file
  [note] config_v0.yaml: no setting is left open. config_v0 is supposed to keep the sizes empty until TSAI fills them on 31 Oct - worth a look
  -> GATE 1 CHECKS INCOMPLETE

=== Udit ========================================================
  runs: evidence/operational_gates/full_a/loss.txt  vs  evidence/operational_gates/full_b/loss.txt
  runs: evidence/operational_gates/full_a/total_loss.txt  vs  evidence/operational_gates/full_b/total_loss.txt
  [PASS] configs/config_v0.yaml loads and every setting has a value and a source  201 settings, 0 errors, 35 declared open
  [PASS] two runs have the same number of steps                                   30 vs 30
  [PASS] the two loss series are numerically identical                            identical
  [PASS] the files are byte-identical                                             exact
  [PASS] no NaN or inf in the loss values                                         0 found
  [PASS] the loss goes down                                                       11.69676802 -> 9.53836229
  [PASS] two runs have the same number of steps                                   30 vs 30
  [PASS] the two loss series are numerically identical                            identical
  [PASS] the files are byte-identical                                             exact
  [PASS] no NaN or inf in the loss values                                         0 found
  [PASS] the loss goes down                                                       11.72560883 -> 9.55579853
  [note] G8 metric names - 0/6 exact over 30 steps; renamed: tokens_per_second -> perf/tokens_per_second, loss_ntp -> train/ce_loss, loss_mtp -> loss/mtp, loss -> train/total_loss, lr -> optimizer/adamw_lr; NOT LOGGED: gpu_alloc_gb
  [note] G8 metric names - 0/6 exact over 30 steps; renamed: tokens_per_second -> perf/tokens_per_second, loss_ntp -> train/ce_loss, loss_mtp -> loss/mtp, loss -> train/total_loss, lr -> optimizer/adamw_lr; NOT LOGGED: gpu_alloc_gb
  -> GATE 1 CHECKS PASS

=== Vedang ======================================================
  runs: results/determinism_cpu/run1.csv  vs  results/determinism_cpu/run2.csv
  [PASS] configs/config_v0.yaml loads and every setting has a value and a source  52 settings, 0 errors, 8 declared open
  [PASS] two runs have the same number of steps                                   100 vs 100
  [PASS] the two loss series are numerically identical                            identical
  [PASS] the files are byte-identical                                             exact
  [PASS] no NaN or inf in the loss values                                         0 found
  [PASS] the loss goes down                                                       15.35006140 -> 11.58852670
  [note] configs/config_v0.yaml: sources are in SOURCES.md, not per-line comments - this tool checks values only for that file
  -> GATE 1 CHECKS PASS

=== deepjyotisaha ===============================================
  runs: evidence/seed_test/tiny_dense/run_a/losses.txt  vs  evidence/seed_test/tiny_dense/run_b/losses.txt
  runs: evidence/seed_test/tiny_moe/run_a/losses.txt  vs  evidence/seed_test/tiny_moe/run_b/losses.txt
  [PASS] configs/config_v0.yaml loads and every setting has a value and a source  85 settings, 0 errors, 0 declared open
  [PASS] two runs have the same number of steps                                   40 vs 40
  [PASS] the two loss series are numerically identical                            identical
  [PASS] the files are byte-identical                                             exact
  [PASS] no NaN or inf in the loss values                                         0 found
  [PASS] the loss goes down                                                       11.70810338 -> 8.53114815
  [PASS] two runs have the same number of steps                                   40 vs 40
  [PASS] the two loss series are numerically identical                            identical
  [PASS] the files are byte-identical                                             exact
  [PASS] no NaN or inf in the loss values                                         0 found
  [PASS] the loss goes down                                                       11.71112079 -> 8.53810687
  [note] configs/config_v0.yaml: no setting is left open. config_v0 is supposed to keep the sizes empty until TSAI fills them on 31 Oct - worth a look
  [note] G8 metric names - 6/6 exact over 40 steps
  [note] G8 metric names - 6/6 exact over 40 steps
  -> GATE 1 CHECKS PASS

=== revantg =====================================================
  runs: results/kaggle_matrix_300/R2a/losses.tsv  vs  results/kaggle_matrix_300/R2b/losses.tsv
  runs: results/kaggle_matrix_300/R3a/losses.tsv  vs  results/kaggle_matrix_300/R3b/losses.tsv
  runs: results/kaggle_matrix_300/R7a/losses.tsv  vs  results/kaggle_matrix_300/R7b/losses.tsv
  [PASS] configs/config_small.yaml loads and every setting has a value and a source  107 settings, 0 errors, 2 declared open
  [PASS] configs/config_v0.yaml loads and every setting has a value and a source     107 settings, 0 errors, 16 declared open
  [PASS] two runs have the same number of steps                                      300 vs 300
  [PASS] the two loss series are numerically identical                               identical
  [PASS] the files are byte-identical                                                exact
  [PASS] no NaN or inf in the loss values                                            0 found
  [PASS] the loss goes down                                                          11.84370041 -> 7.33716631
  [PASS] two runs have the same number of steps                                      300 vs 300
  [PASS] the two loss series are numerically identical                               identical
  [PASS] the files are byte-identical                                                exact
  [PASS] no NaN or inf in the loss values                                            0 found
  [PASS] the loss goes down                                                          11.84370041 -> 7.33716631
  [PASS] two runs have the same number of steps                                      300 vs 300
  [PASS] the two loss series are numerically identical                               identical
  [PASS] the files are byte-identical                                                exact
  [PASS] no NaN or inf in the loss values                                            0 found
  [PASS] the loss goes down                                                          11.82132816 -> 7.34148073
  [note] configs/config_small.yaml: sources are in README.md, not per-line comments - this tool checks values only for that file
  [note] configs/config_v0.yaml: sources are in README.md, not per-line comments - this tool checks values only for that file
  -> GATE 1 CHECKS PASS

=== shivman =====================================================
  runs: evidence/run_a/losses.txt  vs  evidence/run_b/losses.txt
  [PASS] configs/config_v0.yaml loads and every setting has a value and a source  19 settings, 0 errors, 11 declared open
  [PASS] two runs have the same number of steps                                   40 vs 40
  [PASS] the two loss series are numerically identical                            identical
  [PASS] the files are byte-identical                                             exact
  [PASS] no NaN or inf in the loss values                                         0 found
  [PASS] the loss goes down                                                       4.31079626 -> 0.44367132
  [note] G8 metric names - 6/6 exact over 40 steps
  -> GATE 1 CHECKS PASS

=== vardhan =====================================================
  [FAIL] a config file is present             none found
  [FAIL] a pair of same-seed runs is present  0 loss file(s), no a/b pair found
  -> GATE 1 CHECKS INCOMPLETE

====================================================================
  INCOMPLETE  Dharshan
  PASS  Udit
  PASS  Vedang
  PASS  deepjyotisaha
  PASS  revantg
  PASS  shivman
  INCOMPLETE  vardhan
```

## What to act on

### 1. Udit's harness does not emit the six names we sent G8

On Tuesday the lead sent G8 exactly: `tokens_per_second`, `loss_ntp`,
`loss_mtp`, `loss`, `lr`, `gpu_alloc_gb`. Udit's `metrics.jsonl` logs 48 keys
per step — richer than the six — but under a namespaced scheme:

| G8 name | In Udit's runs |
|---|---|
| `tokens_per_second` | `perf/tokens_per_second` |
| `loss_ntp` | `train/ce_loss` (also `loss/ce`) |
| `loss_mtp` | `loss/mtp` |
| `loss` | `train/total_loss` (also `loss/total`) |
| `lr` | `optimizer/adamw_lr` **and** `optimizer/muon_lr` — no single `lr` |
| `gpu_alloc_gb` | **not logged at all** |

Two separate problems. The renames are a mapping we can agree in an hour. But
`gpu_alloc_gb` has no counterpart, and `lr` is genuinely ambiguous under the
Muon/AdamW hybrid — G8 asked for one learning rate and the optimizer has two.
G8's logger lands Thursday 29 Oct, so this wants settling before then: either
Udit emits the six flat aliases alongside his namespaced keys, or we tell G8
the mapping and the fact that `lr` is now two fields.

Shivanshu's and Deepjyoti's runs both log 6/6 exactly.

### 2. Two `config_v0.yaml` files leave nothing open

The rule is "size values stay empty, TSAI fills them on 31 Oct". Dharshan's and
Deepjyoti's `config_v0.yaml` have **no** setting marked open — every value is
filled. Udit leaves 35 open, Shivanshu 11, Revant 14.

This is not a gate failure and the tool does not treat it as one, but a
config_v0 with zero open sizes either pre-empts TSAI's 31 Oct injection or is
really a small-run config under the wrong name. Worth one line from each
author saying which.

### 3. Dharshan has no run evidence

No loss file anywhere in the folder, so "two seed-1234 runs match", "the loss
goes down" and "no NaN" cannot be checked at all. The config and the harness
code are there; the evidence is not. If the runs were done, the outputs just
need committing.

## Caveats on this sweep

- It checks the **evidence**, not the harness. A folder that passes has
  internally consistent artifacts; it does not prove the training code is
  correct.
- "Sources are in a separate file" means values were checked but sourcing was
  not graded for that config. That applies to Dharshan, Revant and Vedang.
- Pass/fail here is about the Saturday gate wording only. TSAI approving the
  experiment plan is the remaining gate item and is not something a script
  can check.
