"""Evaluate a trained PPO/SAC model on PandaPush-v3."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from env_utils import load_model_auto, make_push_env, parse_mass_range


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO/SAC model on PandaPush-v3.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--algorithm", type=str, default="auto", choices=["auto", "ppo", "sac"])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy sampling.")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--env-type", type=str, default="target", choices=["source", "target"])
    parser.add_argument("--reward-type", type=str, default="dense", choices=["dense", "sparse"])
    parser.add_argument("--control-type", type=str, default="ee", choices=["ee", "joints"])
    parser.add_argument("--sampling-strategy", type=str, default="none", choices=["none", "udr", "adr"])
    parser.add_argument("--mass-range", type=float, nargs=2, default=(0.5, 8.0), metavar=("LOW", "HIGH"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--use-sb3-helper", action="store_true", help="Also compute the mean/std via SB3 evaluate_policy.")
    return parser.parse_args()


def evaluate_manual(
    model,
    env,
    n_episodes: int,
    deterministic: bool,
    render: bool,
) -> Dict[str, Any]:
    episode_returns: List[float] = []
    episode_lengths: List[int] = []
    successes: List[float] = []

    for episode in range(1, n_episodes + 1):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_return = 0.0
        episode_len = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            episode_len += 1
            if render:
                env.render()

        episode_returns.append(episode_return)
        episode_lengths.append(episode_len)
        if isinstance(info, dict) and "is_success" in info:
            successes.append(float(info["is_success"]))

        print(
            f"Episode {episode:03d} | return={episode_return:9.3f} | "
            f"length={episode_len:03d} | success={successes[-1] if successes else float('nan'):.0f}"
        )

    returns = np.asarray(episode_returns, dtype=np.float64)
    lengths = np.asarray(episode_lengths, dtype=np.float64)
    result: Dict[str, Any] = {
        "episodes": int(n_episodes),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "mean_length": float(lengths.mean()),
        "std_length": float(lengths.std()),
        "success_rate": float(np.mean(successes)) if successes else None,
        "returns": [float(x) for x in episode_returns],
        "lengths": [int(x) for x in episode_lengths],
        "successes": [float(x) for x in successes],
    }
    return result


def append_csv(path: str, row: Dict[str, Any]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "label",
        "model_path",
        "algorithm",
        "env_type",
        "sampling_strategy",
        "episodes",
        "mean_return",
        "std_return",
        "min_return",
        "max_return",
        "mean_length",
        "std_length",
        "success_rate",
    ]
    exists = path_obj.exists()
    with open(path_obj, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    mass_range = parse_mass_range(args.mass_range)
    render_mode = "human" if args.render else "rgb_array"
    env = make_push_env(
        env_type=args.env_type,
        reward_type=args.reward_type,
        render_mode=render_mode,
        control_type=args.control_type,
        sampling_strategy=args.sampling_strategy,
        mass_range=mass_range,
        seed=args.seed,
    )
    model = load_model_auto(args.model_path, args.algorithm)

    deterministic = not args.stochastic
    result = evaluate_manual(model, env, args.episodes, deterministic=deterministic, render=args.render)

    if args.use_sb3_helper:
        helper_mean, helper_std = evaluate_policy(
            model,
            env,
            n_eval_episodes=args.episodes,
            deterministic=deterministic,
            render=False,
        )
        result["sb3_helper_mean_return"] = float(helper_mean)
        result["sb3_helper_std_return"] = float(helper_std)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "label": args.label,
        "model_path": args.model_path,
        "algorithm": args.algorithm,
        "env_type": args.env_type,
        "reward_type": args.reward_type,
        "control_type": args.control_type,
        "sampling_strategy": args.sampling_strategy,
        "deterministic": deterministic,
        "mass_range": list(mass_range),
        **result,
    }

    print("\n=== Evaluation summary ===")
    compact = {key: value for key, value in summary.items() if key not in {"returns", "lengths", "successes"}}
    print(json.dumps(compact, indent=2, sort_keys=True))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"Saved JSON results to: {output_path}")

    if args.output_csv:
        append_csv(args.output_csv, summary)
        print(f"Appended CSV summary to: {args.output_csv}")

    env.close()


if __name__ == "__main__":
    main()
