"""Build revantg/index.html (designdotmd "arctic" style) from results/ and index.template.html.

Usage: python kaggle/build_index.py index.html   (run from revantg/)
"""
import json, sys
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
SW = RES / "moe_sweep_1000"
ARMS = [  # key, label, colour
    ("D0_dense", "Dense (d_ff 768)", "#8A9DAB"),
    ("M1_e4k2", "MoE 4 experts, top-2", "#7DD3FC"),
    ("M2_e8k2", "MoE 8 experts, top-2", "#2563EB"),
    ("M3_e16k4_fine", "MoE 16 fine experts, top-4", "#14B8A6"),
    ("M4_e8k1_shared1", "MoE 8 experts, top-1 + 1 shared", "#0E1A24"),
]
LAYOUTS = [  # results/kaggle_matrix_300/README.md table
    ("Dense", "1 GPU", 8573), ("Dense", "DP=2", 13659), ("Dense", "TP=2", 5547),
    ("Dense", "PP=2", 9276), ("Dense", "DP=2 + ZeRO-1", 13797),
    ("MoE", "1 GPU", 6724), ("MoE", "DP=2", 10228), ("MoE", "PP=2", 8129),
]


def losses(k):
    return [float(l.split("\t")[1]) for l in (SW / k / "losses.tsv").read_text().split("\n") if l.strip()]


S = json.loads((SW / "summary.json").read_text())
dense = losses("D0_dense")
series = []
for k, label, col in ARMS:
    L = losses(k)
    d = [a - b for a, b in zip(L, dense)]
    gap = [[i, round(sum(d[i - 49:i + 1]) / 50, 5)] for i in range(49, len(d), 5)] + [[len(d) - 1, round(sum(d[-50:]) / 50, 5)]]
    s = S[k]
    series.append(dict(key=k, label=label, color=col, gap=gap, ho=[[a, round(b, 4)] for a, b in s["ho"]],
                       params=s["params"], ce=s["ce_last100"], held=s["held_out"], bpb=s["bpb"], tps=s["tps"],
                       dead=s["dead_max"], top=s["top_max"]))
DATA = json.dumps(dict(arms=series, layouts=LAYOUTS), separators=(",", ":"))

HTML = Path(__file__).with_name("index.template.html").read_text().replace("/*DATA*/null", DATA)
Path(sys.argv[1]).write_text(HTML)
print("wrote", sys.argv[1], len(HTML), "bytes")
