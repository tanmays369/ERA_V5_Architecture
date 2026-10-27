# Harness quickstart (Team 05)

Run everything from `teams/05-architecture`. Works on Colab, Kaggle or a laptop (CPU is enough for the checks).

```bash
pip -q install pyyaml datasets kronecker-embeddings        # transformers and torch are already on Colab/Kaggle
# fallback if the PyPI package fails: pip -q install git+https://github.com/theschoolofai/kronecker-embeddings.git

python harness/check_config.py configs/config_v0.yaml      # 0 errors; 8 OPEN sizes; PENDING until the sweeps are run
python harness/train.py --list-metrics                     # metric names for G8

python harness/prepare_data.py --tokens_per_source 300000  # Hindi + English; each download has a hard time limit
# slow or blocked network:  python harness/prepare_data.py --offline   (sample sentences: smoke tests only)

# determinism: the loss at every step must match
python harness/train.py --config configs/small_cpu.yaml --device cpu --seed 1234 --log /tmp/run1.csv
python harness/train.py --config configs/small_cpu.yaml --device cpu --seed 1234 --log /tmp/run2.csv
diff /tmp/run1.csv /tmp/run2.csv && echo IDENTICAL

# measurements that back the config sources (see configs/SOURCES.md)
python harness/sweep.py --list
python harness/sweep.py --config configs/small_cpu.yaml --only lr_sweep --steps 100   # prints a time estimate after the first run
python harness/sweep.py --config configs/small_cpu.yaml --all --steps 100             # a stopped sweep resumes; finished sweeps are skipped (--redo repeats)
python harness/sweep.py --apply configs/config_v0.yaml configs/small_cpu.yaml configs/small_colab.yaml
python harness/check_config.py configs/config_v0.yaml --strict
```

Files: `train.py` (harness and tiny MoE), `prepare_data.py` (Hindi + English tokens), `check_config.py` (config checker),
`sweep.py` (measurements). Speed and memory go to `<log>.system.csv`; that file differs between runs, so never diff it.
`data/mixed_stats.json` says whether any part of the data is synthetic: do not commit measurements made from synthetic data.

## Metric names (G8)

`step, loss, loss_ntp, loss_mtp, load_balance_loss, router_z_loss, router_entropy, expert_load_cv, expert_load_max_frac,
grad_norm, lr, val_loss` in the reproducible CSV; `tokens_per_second, gpu_alloc_gb` in the `.system.csv`.
