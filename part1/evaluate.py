"""Evaluate a saved Phase 2 Hopper policy checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch

from agent import Agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved REINFORCE/Actor-Critic Hopper checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt or final_model.pt.")
    parser.add_argument("--env-id", type=str, default="Hopper-v4")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save-json", type=str, default=None, help="Optional path where evaluation metrics are saved.")
    return parser.parse_args()


def evaluate_checkpoint(
    checkpoint: str,
    env_id: str,
    episodes: int,
    seed: int,
    render: bool,
    max_steps: Optional[int],
    device: str,
) -> dict:
    render_mode = "human" if render else "rgb_array"
    env = gym.make(env_id, render_mode=render_mode)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    agent = Agent.load(checkpoint, device=device)
    agent.policy.eval()

    returns: List[float] = []
    lengths: List[int] = []

    for episode in range(1, episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        done = False
        episode_return = 0.0
        episode_length = 0

        while not done:
            action_tensor, _ = agent.act(state, evaluation=True)
            action = action_tensor.detach().cpu().numpy()
            state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            episode_return += float(reward)
            episode_length += 1
            if max_steps is not None and episode_length >= max_steps:
                done = True

        returns.append(episode_return)
        lengths.append(episode_length)
        print(f"Episode {episode:03d} | return={episode_return:.3f} | length={episode_length}")

    env.close()
    returns_np = np.asarray(returns, dtype=np.float64)
    lengths_np = np.asarray(lengths, dtype=np.float64)

    return {
        "checkpoint": str(checkpoint),
        "env_id": env_id,
        "episodes": episodes,
        "mean_return": float(returns_np.mean()),
        "std_return": float(returns_np.std()),
        "min_return": float(returns_np.min()),
        "max_return": float(returns_np.max()),
        "mean_length": float(lengths_np.mean()),
        "std_length": float(lengths_np.std()),
    }


def main() -> None:
    args = parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    metrics = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        env_id=args.env_id,
        episodes=args.episodes,
        seed=args.seed,
        render=args.render,
        max_steps=args.max_steps,
        device=device,
    )

    print("\n=== Evaluation summary ===")
    print(json.dumps(metrics, indent=2))

    if args.save_json is not None:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as handle:
            json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()
