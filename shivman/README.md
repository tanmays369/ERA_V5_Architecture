# shivman

Shivanshu Thapliyal. Week-1 folder in the working repo. This is not the lab-repo skeleton.

`configs/config_v0.yaml` keeps model sizes empty. `configs/small/tiny_moe.yaml` is only the CPU smoke test. The tiny run uses 4 heads and top-2 experts, a second linear head for `loss_mtp`, and a new batch on every step. `gpu_alloc_gb` is read from CUDA and is 0 on CPU.

```bash
python harness/check_config.py configs/config_v0.yaml
python harness/check_config.py configs/config_v0.yaml --tiny configs/small/tiny_moe.yaml
python harness/train.py --config configs/config_v0.yaml --tiny configs/small/tiny_moe.yaml --out evidence/run_a
python harness/train.py --config configs/config_v0.yaml --tiny configs/small/tiny_moe.yaml --out evidence/run_b
diff evidence/run_a/losses.txt evidence/run_b/losses.txt
```

The diagnosis rows are not in this folder. The English FineWeb-Edu pack is not on this machine, and the labels have to be read from that pack.
