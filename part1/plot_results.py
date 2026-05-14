"""Plot Phase 2 training curves from a training_log.csv file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Hopper training returns.")
    parser.add_argument("--log", type=str, required=True, help="Path to training_log.csv.")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path. Defaults next to the CSV.")
    return parser.parse_args()


def load_csv(path: Path) -> Dict[str, List[float]]:
    columns: Dict[str, List[float]] = {}
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key, value in row.items():
                columns.setdefault(key, [])
                if value == "":
                    columns[key].append(float("nan"))
                else:
                    columns[key].append(float(value))
    return columns


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)
    output_path = Path(args.output) if args.output else log_path.with_name("learning_curve.png")

    data = load_csv(log_path)
    episodes = data["episode"]

    plt.figure(figsize=(9, 5))
    plt.plot(episodes, data["train_return"], label="train return", alpha=0.35)
    plt.plot(episodes, data["moving_avg_return_20"], label="moving avg return (20)")

    if "eval_mean_return" in data and any(value == value for value in data["eval_mean_return"]):
        eval_points_x = [ep for ep, y in zip(episodes, data["eval_mean_return"]) if y == y]
        eval_points_y = [y for y in data["eval_mean_return"] if y == y]
        plt.plot(eval_points_x, eval_points_y, marker="o", linestyle="--", label="deterministic eval mean")

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Hopper-v4 Phase 2 training curve")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
