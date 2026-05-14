"""Run the standard Phase 3/4 experiment matrix.

This script is a convenience launcher.  It intentionally uses the public
`train_sb3.py` and `eval_sb3.py` entry points so every command is reproducible
and can be copied into a report or terminal.

Default matrix:
1. Train source fixed-mass policy.
2. Train target fixed-mass policy.
3. Evaluate source->source, source->target, target->target.
4. Optionally train UDR and ADR source policies and evaluate each on source and target.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


THIS_DIR = Path(__file__).resolve().parent


def run(cmd: List[str], dry_run: bool = False) -> None:
    print("\n$ " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def latest_run_dir(root: Path, prefix: str) -> Path:
    candidates = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)], key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No run directory found in {root} with prefix {prefix}")
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3/4 PandaPush experiment matrix.")
    parser.add_argument("--algorithm", choices=["ppo", "sac"], default="sac")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="results/part2")
    parser.add_argument("--mass-range", type=float, nargs=2, default=(0.5, 8.0), metavar=("LOW", "HIGH"))
    parser.add_argument("--include-udr", action="store_true")
    parser.add_argument("--include-adr", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()


def train_cmd(args: argparse.Namespace, env_type: str, sampling: str) -> List[str]:
    run_name = f"{args.algorithm}_{sampling}_{env_type}_{args.timesteps // 1000}k_seed{args.seed}"
    cmd = [
        sys.executable,
        str(THIS_DIR / "train_sb3.py"),
        "--algorithm",
        args.algorithm,
        "--env-type",
        env_type,
        "--sampling-strategy",
        sampling,
        "--timesteps",
        str(args.timesteps),
        "--seed",
        str(args.seed),
        "--save-dir",
        args.save_dir,
        "--run-name",
        run_name,
        "--mass-range",
        str(args.mass_range[0]),
        str(args.mass_range[1]),
    ]
    if args.progress_bar:
        cmd.append("--progress-bar")
    return cmd


def eval_cmd(
    args: argparse.Namespace,
    model_path: Path,
    env_type: str,
    label: str,
    csv_path: Path,
    sampling: str = "none",
) -> List[str]:
    json_path = csv_path.parent / f"{label}.json"
    return [
        sys.executable,
        str(THIS_DIR / "eval_sb3.py"),
        "--model-path",
        str(model_path),
        "--algorithm",
        args.algorithm,
        "--env-type",
        env_type,
        "--sampling-strategy",
        sampling,
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed + 1234),
        "--mass-range",
        str(args.mass_range[0]),
        str(args.mass_range[1]),
        "--label",
        label,
        "--output-json",
        str(json_path),
        "--output-csv",
        str(csv_path),
    ]


def main() -> None:
    args = parse_args()
    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    summary_dir = save_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / f"{args.algorithm}_phase3_phase4_summary_seed{args.seed}.csv"

    train_specs = [("source", "none"), ("target", "none")]
    if args.include_udr:
        train_specs.append(("source", "udr"))
    if args.include_adr:
        train_specs.append(("source", "adr"))

    for env_type, sampling in train_specs:
        run(train_cmd(args, env_type=env_type, sampling=sampling), dry_run=args.dry_run)

    if args.dry_run:
        return

    source_model = save_root / f"{args.algorithm}_none_source_{args.timesteps // 1000}k_seed{args.seed}" / "final_model.zip"
    target_model = save_root / f"{args.algorithm}_none_target_{args.timesteps // 1000}k_seed{args.seed}" / "final_model.zip"

    eval_specs = [
        (source_model, "source", "source_to_source", "none"),
        (source_model, "target", "source_to_target_lower_bound", "none"),
        (target_model, "target", "target_to_target_upper_bound", "none"),
    ]

    if args.include_udr:
        udr_model = save_root / f"{args.algorithm}_udr_source_{args.timesteps // 1000}k_seed{args.seed}" / "final_model.zip"
        eval_specs.extend(
            [
                (udr_model, "source", "udr_source_to_source", "none"),
                (udr_model, "target", "udr_source_to_target", "none"),
            ]
        )

    if args.include_adr:
        adr_model = save_root / f"{args.algorithm}_adr_source_{args.timesteps // 1000}k_seed{args.seed}" / "final_model.zip"
        eval_specs.extend(
            [
                (adr_model, "source", "adr_source_to_source", "none"),
                (adr_model, "target", "adr_source_to_target", "none"),
            ]
        )

    for model_path, env_type, label, sampling in eval_specs:
        run(eval_cmd(args, model_path=model_path, env_type=env_type, label=label, csv_path=csv_path, sampling=sampling))

    metadata = {
        "algorithm": args.algorithm,
        "timesteps": args.timesteps,
        "episodes": args.episodes,
        "seed": args.seed,
        "mass_range": list(args.mass_range),
        "summary_csv": str(csv_path),
    }
    with open(summary_dir / f"{args.algorithm}_phase3_phase4_metadata_seed{args.seed}.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    print(f"\nSummary CSV: {csv_path}")


if __name__ == "__main__":
    main()
