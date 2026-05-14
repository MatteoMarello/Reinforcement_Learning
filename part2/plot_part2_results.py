"""Plot Phase 3/4 evaluation summaries saved by eval_sb3.py."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: str) -> List[dict]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mean returns from a Phase 3/4 summary CSV.")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--title", type=str, default="PandaPush evaluation summary")
    args = parser.parse_args()

    rows = read_rows(args.csv)
    if not rows:
        raise ValueError(f"No rows found in {args.csv}")

    labels = [row.get("label") or f"row_{i}" for i, row in enumerate(rows)]
    means = np.asarray([float(row["mean_return"]) for row in rows], dtype=float)
    stds = np.asarray([float(row["std_return"]) for row in rows], dtype=float)

    fig_width = max(8, 0.8 * len(labels))
    plt.figure(figsize=(fig_width, 5))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Average return")
    plt.title(args.title)
    plt.tight_layout()

    output = args.output or str(Path(args.csv).with_suffix(".png"))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    print(f"Saved plot to: {output}")


if __name__ == "__main__":
    main()
