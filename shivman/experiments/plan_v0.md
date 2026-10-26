# Experiment plan v0

Draft for the Saturday gate. Each arm has a metric, a pass bar, and GPU hours.
The hours below are the CPU smoke test, not the capstone run.

| Arm | What changes | Metric | Pass bar | GPU hours |
|---|---|---|---|---|
| seed-match | Two launches, seed 1234, tiny MoE | `loss` | The two `losses.txt` files are identical | 0.05 |
| loss-falls | Same tiny run | `loss_ntp` | Last step is below the first step, and no value is NaN | 0.05 |
| log-names | Same tiny run | the six names below | Every step writes all six | 0.05 |

Names the harness writes, from the Sunday G5 list. G8 still has to confirm units and cadence:

- `tokens_per_second`
- `loss_ntp`
- `loss_mtp`
- `loss`
- `lr`
- `gpu_alloc_gb` is 0 on this CPU run

Not in this draft: a production learning rate, a model size, or the KE2 output shape. Those stay empty until TSAI and G4 fill them.
