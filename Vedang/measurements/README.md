# measurements/

Result files from our own tiny runs. `config_v0.yaml` cites them as `# src: measurement:<id>`, and
`harness/check_config.py` checks that each file exists and that the config default equals the value it chose.

- `tokenizer_vocab.json` is written by `harness/prepare_data.py`.
- every other file (`lr_sweep.json`, `balance_sweep.json`, ...) is written by `harness/sweep.py`.
- do not edit these by hand; rerun the sweep instead. Commit the real ones (from the real tokenizer and Hindi + English data),
  not ones made with `--offline` sample data or the 100-step test runs.
