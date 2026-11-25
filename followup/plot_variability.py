"""Plot per-batch success rates of the logged 500k evaluations (reads evaluation_variability_500k.csv)."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "followup/results/evaluation_variability_500k.csv"
OUT = ROOT / "followup/results/evaluation_variability_500k.png"
LABELS = {
    "source_to_source": "source→source",
    "source_to_target_lower_bound": "source→target\n(lower bound)",
    "target_to_target_upper_bound": "target→target\n(upper bound)",
    "udr_source_to_source": "UDR\nsource→source",
    "udr_source_to_target": "UDR\nsource→target",
    "adr_source_to_source": "ADR\nsource→source",
    "adr_source_to_target": "ADR\nsource→target",
}

rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
fig, ax = plt.subplots(figsize=(8.5, 3.6))
for i, r in enumerate(rows):
    vals = [float(v) for v in r["success_per_batch"].split(" / ")]
    lo, hi = (float(x) for x in r["pooled_wilson95"].strip("[]").split(", "))
    p = float(r["success_pooled"])
    ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt="o", color="#1f4e79", capsize=4, zorder=3)
    ax.scatter([i + 0.12] * len(vals), vals, s=14, color="#9a9a9a", zorder=2)
ax.set_xticks(range(len(rows)), [LABELS[r["config"]] for r in rows], fontsize=8)
ax.set_ylabel("success rate")
ax.set_ylim(0.3, 1.04)
ax.grid(axis="y", alpha=0.3)
ax.set_title("SAC 500k, PandaPush: pooled success (Wilson 95% CI) and the three logged 50-episode batches (grey)",
             fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("saved", OUT)
