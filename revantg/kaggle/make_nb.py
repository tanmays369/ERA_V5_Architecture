"""Build the Kaggle T4x2 run-matrix notebook (R0-R5) as JSON for kaggle save_notebook.

Usage: python make_nb.py [BRANCH] > nb.json
"""
import json, sys

branch = sys.argv[1] if len(sys.argv) > 1 else "t5-harness-v0"
T = "teams/05-architecture"
H = f"{T}/harness"
C = f"{T}/configs/small"

run_sh = f"""set -u
cd /kaggle/working/w
export HF_HOME=/kaggle/working/hf PYTHONHASHSEED=0
R=/kaggle/working/results; mkdir -p $R
tr() {{ n=$1; shift; torchrun --nproc_per_node=$n --master_port=$((29500 + RANDOM % 1000)) {H}/train.py --device cuda "$@"; }}
st() {{ echo "=== $1 exit=$2" | tee -a $R/status.txt; }}
for c in t4x1_ref t4x2_dp2 t4x2_tp2 t4x2_pp2 t4x2_dp2_zero1 t4x1_moe t4x2_dp2_moe t4x2_pp2_moe; do
  ws=2; case $c in t4x1_*) ws=1;; esac
  python {H}/check_config.py {C}/$c.yaml --world-size $ws 2>&1 | tee $R/check_$c.txt
done
python {H}/check_config.py {T}/configs/config_v0.yaml 2>&1 | tee $R/check_config_v0.txt
tr 1 --config {C}/t4x1_ref.yaml --out $R/R0 > $R/R0.log 2>&1; st R0 $?
for r in a b; do tr 2 --config {C}/t4x2_dp2.yaml --out $R/R1$r > $R/R1$r.log 2>&1; st R1$r $?; done
for r in a b; do tr 2 --config {C}/t4x2_tp2.yaml --out $R/R2$r > $R/R2$r.log 2>&1; st R2$r $?; done
for r in a b; do tr 2 --config {C}/t4x2_pp2.yaml --out $R/R3$r > $R/R3$r.log 2>&1; st R3$r $?; done
tr 2 --config {C}/t4x2_dp2_zero1.yaml --out $R/R4 > $R/R4.log 2>&1; st R4 $?
tr 2 --config {C}/t4x2_dp2.yaml --out $R/R5 --inject-nan 150 > $R/R5.log 2>&1; st R5 $?
ck=$(ls -d $R/R5/checkpoints/step_* 2>/dev/null | tail -1); echo "resume from $ck" | tee -a $R/status.txt
tr 2 --config {C}/t4x2_dp2.yaml --out $R/R5 --resume $ck > $R/R5r.log 2>&1; st R5r $?
for r in a b; do tr 1 --config {C}/t4x1_moe.yaml --out $R/R6$r > $R/R6$r.log 2>&1; st R6$r $?; done
for r in a b; do tr 2 --config {C}/t4x2_dp2_moe.yaml --out $R/R7$r > $R/R7$r.log 2>&1; st R7$r $?; done
tr 2 --config {C}/t4x2_pp2_moe.yaml --out $R/R8 > $R/R8.log 2>&1; st R8 $?
{{
python {H}/compare_runs.py --gate $R/R0; echo "gate R0 exit=$?"
for g in R1a R6a R7a; do python {H}/compare_runs.py --gate $R/$g; echo "gate $g exit=$?"; done
for p in R1 R2 R3 R6 R7; do python {H}/compare_runs.py $R/${{p}}a $R/${{p}}b --exact; echo "exact $p exit=$?"; done
python {H}/compare_runs.py $R/R1a $R/R5 --exact; echo "exact R1a-R5resumed exit=$?"
for x in R1a R2a R3a R4; do python {H}/compare_runs.py $R/R0 $R/$x --rtol 1e-3; echo "rtol R0-$x exit=$?"; done
python {H}/compare_runs.py $R/R1a $R/R4 --rtol 1e-3; echo "rtol R1a-R4 exit=$?"
for x in R7a R8; do python {H}/compare_runs.py $R/R6a $R/$x --rtol 1e-3; echo "rtol R6a-$x exit=$?"; done
}} 2>&1 | tee $R/compare.txt
find $R -name checkpoints -type d -prune -exec rm -rf {{}} +
cp -r work/data/t4/manifest.json $R/ 2>/dev/null
tar czf /kaggle/working/results.tgz -C /kaggle/working results
"""

cells = [
    f"!rm -rf /kaggle/working/w && git clone -q --depth 1 -b {branch} https://github.com/revantg/era-v5-lab.git /kaggle/working/w && git -C /kaggle/working/w log --oneline -1\n"
    "!pip install -q kronecker-embeddings 2>&1 | tail -1\n"
    "import torch, platform; print(platform.python_version(), torch.__version__, torch.cuda.device_count())",
    "%%bash\ncd /kaggle/working/w\nmkdir -p /kaggle/working/results\n"
    f"(time HF_HOME=/kaggle/working/hf python {H}/prepare_data.py --config {C}/t4_base.yaml) 2>&1 | tee /kaggle/working/results/prepare_data.log | tail -30",
    "%%bash\n" + run_sh,
    "!cat /kaggle/working/results/status.txt /kaggle/working/results/compare.txt; ls -la /kaggle/working/results",
]
nb = {
    "metadata": {"kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 4,
    "cells": [{"cell_type": "code", "source": c, "metadata": {}, "outputs": [], "execution_count": None} for c in cells],
}
print(json.dumps(nb))
