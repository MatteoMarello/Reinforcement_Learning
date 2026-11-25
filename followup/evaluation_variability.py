"""Quantify the spread of the logged 500k evaluations (no new runs).

`results/part2/summaries/sac_phase3_phase4_summary_seed0.csv` contains three complete evaluation
batches (2026-05-20, 2026-05-25, 2026-05-26) of the 500k SAC configuration matrix, 50 episodes each.
The final Phase 4 table reports the last batch only. This script pools the batches, reports the
per-batch range and a Wilson 95% interval for each batch, and writes a summary CSV.

Caveat: the log records the checkpoint path, not a checkpoint hash, so it cannot show whether a
checkpoint was retrained between batches. The spread therefore mixes evaluation noise
(`eval_sb3.py` does not seed `env.reset()`) and possibly training-run variance.

Usage: python followup/evaluation_variability.py
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results/part2/summaries/sac_phase3_phase4_summary_seed0.csv"
OUT = ROOT / "followup/results/evaluation_variability_500k.csv"
ORDER = [
    "source_to_source", "source_to_target_lower_bound", "target_to_target_upper_bound",
    "udr_source_to_source", "udr_source_to_target", "adr_source_to_source", "adr_source_to_target",
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def main() -> None:
    batches: dict[str, list[tuple[str, int, float, float]]] = defaultdict(list)
    with open(LOG, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if "500k" not in r["model_path"]:
                continue
            n = int(r["episodes"])
            k = round(float(r["success_rate"]) * n)
            batches[r["label"]].append((r["timestamp"][:10], n, k / n, float(r["mean_return"])))

    rows = []
    for label in ORDER:
        b = batches[label]
        n_tot, k_tot = sum(x[1] for x in b), sum(round(x[1] * x[2]) for x in b)
        lo, hi = wilson(k_tot, n_tot)
        last_lo, last_hi = wilson(round(b[-1][1] * b[-1][2]), b[-1][1])
        rows.append({
            "config": label,
            "batches": len(b),
            "success_per_batch": " / ".join(f"{x[2]:.2f}" for x in b),
            "success_pooled": round(k_tot / n_tot, 3),
            "pooled_wilson95": f"[{lo:.2f}, {hi:.2f}]",
            "final_table_value": round(b[-1][2], 2),
            "final_table_wilson95": f"[{last_lo:.2f}, {last_hi:.2f}]",
            "mean_return_per_batch": " / ".join(f"{x[3]:.3f}" for x in b),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['config']:30s} per-batch {r['success_per_batch']:18s} pooled {r['success_pooled']:.3f} "
              f"{r['pooled_wilson95']}  final {r['final_table_value']:.2f} {r['final_table_wilson95']}")


if __name__ == "__main__":
    main()
