"""Train REINFORCE and Actor-Critic agents on Gymnasium Hopper-v4.

Examples
--------
REINFORCE without baseline::

    python train.py --algorithm reinforce --episodes 1000 --run-name reinforce

REINFORCE with a scalar baseline::

    python train.py --algorithm reinforce_baseline --episodes 1000 --run-name reinforce_bl

Actor-Critic::

    python train.py --algorithm actor_critic --episodes 1000 --run-name actor_critic \
        --normalize-advantages

The script writes all outputs under ``part1/results/<run-name>/``:
configuration, environment metadata, training CSV, final model, and best model.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch

from agent import Agent, AgentConfig, Policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train from-scratch policy-gradient agents on Hopper-v4.")
    parser.add_argument("--env-id", type=str, default="Hopper-v4", help="Gymnasium environment id.")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="reinforce",
        choices=["reinforce", "reinforce_baseline", "actor_critic"],
        help="From-scratch algorithm to train.",
    )
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional per-episode step cap.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--actor-lr", type=float, default=3e-4, help="Actor learning rate.")
    parser.add_argument("--critic-lr", type=float, default=1e-3, help="Critic learning rate for Actor-Critic.")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden units per layer.")
    parser.add_argument("--value-loss-coef", type=float, default=0.5, help="Weight of critic MSE in Actor-Critic.")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping max norm; <=0 disables it.")
    parser.add_argument(
        "--normalize-advantages",
        action="store_true",
        help="Normalize advantages/returns before the actor update.",
    )
    parser.add_argument(
        "--baseline-value",
        type=float,
        default=None,
        help=(
            "Fixed scalar baseline for reinforce_baseline. If omitted, the script uses "
            "an exponential moving average scalar baseline from past returns."
        ),
    )
    parser.add_argument(
        "--baseline-momentum",
        type=float,
        default=0.9,
        help="Momentum used for the moving scalar baseline when --baseline-value is omitted.",
    )
    parser.add_argument("--eval-every", type=int, default=25, help="Evaluate every N episodes; 0 disables evaluation.")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of deterministic eval episodes.")
    parser.add_argument("--render", action="store_true", help="Render training episodes in a window.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Torch device.")
    parser.add_argument("--save-dir", type=str, default="results", help="Directory for logs and checkpoints.")
    parser.add_argument("--run-name", type=str, default=None, help="Run name. Defaults to algorithm_timestamp.")
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(env_id: str, seed: int, render: bool = False) -> gym.Env:
    render_mode = "human" if render else "rgb_array"
    env = gym.make(env_id, render_mode=render_mode)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def get_body_names_and_masses(env: gym.Env) -> List[Dict[str, Any]]:
    """Return MuJoCo body names and masses when available."""

    metadata: List[Dict[str, Any]] = []
    unwrapped = env.unwrapped
    model = getattr(unwrapped, "model", None)
    if model is None:
        return metadata

    masses = np.asarray(getattr(model, "body_mass", []), dtype=np.float64)
    nbody = int(getattr(model, "nbody", len(masses)))

    for idx in range(nbody):
        name = str(idx)
        try:
            name = model.body(idx).name
        except Exception:
            pass
        mass = float(masses[idx]) if idx < len(masses) else None
        metadata.append({"body_id": idx, "body_name": name, "mass": mass})
    return metadata


def get_env_metadata(env: gym.Env) -> Dict[str, Any]:
    unwrapped = env.unwrapped
    model = getattr(unwrapped, "model", None)
    metadata: Dict[str, Any] = {
        "env_id": getattr(getattr(env, "spec", None), "id", None),
        "observation_space": str(env.observation_space),
        "action_space": str(env.action_space),
        "action_low": json_ready(env.action_space.low),
        "action_high": json_ready(env.action_space.high),
        "bodies": get_body_names_and_masses(env),
    }
    if model is not None:
        for attr in ["nq", "nv", "nu", "nbody"]:
            if hasattr(model, attr):
                metadata[attr] = int(getattr(model, attr))
        if hasattr(model, "actuator_ctrlrange"):
            metadata["actuator_ctrlrange"] = json_ready(np.asarray(model.actuator_ctrlrange))
    return metadata


def print_env_summary(metadata: Dict[str, Any]) -> None:
    print("\n=== Environment summary ===")
    print(f"Environment:       {metadata.get('env_id')}")
    print(f"Observation space: {metadata.get('observation_space')}")
    print(f"Action space:      {metadata.get('action_space')}")
    if "nq" in metadata and "nv" in metadata and "nu" in metadata:
        print(f"MuJoCo nq/nv/nu:   {metadata['nq']} / {metadata['nv']} / {metadata['nu']}")
    if metadata.get("bodies"):
        print("Body masses:")
        for body in metadata["bodies"]:
            print(f"  [{body['body_id']:02d}] {body['body_name']:<16} mass={body['mass']}")
    print("===========================\n")


def moving_average(values: List[float], window: int = 20) -> float:
    if not values:
        return float("nan")
    start = max(0, len(values) - window)
    return float(np.mean(values[start:]))


def evaluate(agent: Agent, env_id: str, seed: int, n_episodes: int, max_steps: Optional[int] = None) -> Dict[str, float]:
    eval_env = make_env(env_id, seed=seed, render=False)
    returns: List[float] = []
    lengths: List[int] = []

    for episode_idx in range(n_episodes):
        state, _ = eval_env.reset(seed=seed + episode_idx)
        done = False
        episode_return = 0.0
        episode_length = 0

        while not done:
            action_tensor, _ = agent.act(state, evaluation=True)
            action = action_tensor.detach().cpu().numpy()
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            done = bool(terminated or truncated)
            state = next_state
            episode_return += float(reward)
            episode_length += 1
            if max_steps is not None and episode_length >= max_steps:
                break

        returns.append(episode_return)
        lengths.append(episode_length)

    eval_env.close()
    return {
        "eval_mean_return": float(np.mean(returns)),
        "eval_std_return": float(np.std(returns)),
        "eval_mean_length": float(np.mean(lengths)),
    }


def write_csv_row(path: Path, row: Dict[str, Any], header: List[str]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in header})


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    run_name = args.run_name or f"{args.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.save_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.env_id, seed=args.seed, render=args.render)
    env_metadata = get_env_metadata(env)
    print_env_summary(env_metadata)

    with (output_dir / "env_info.json").open("w") as handle:
        json.dump(json_ready(env_metadata), handle, indent=2)
    with (output_dir / "config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    policy = Policy(
        state_space=obs_dim,
        action_space=action_dim,
        hidden_size=args.hidden_size,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
    )
    config = AgentConfig(
        algorithm=args.algorithm,
        gamma=args.gamma,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        value_loss_coef=args.value_loss_coef,
        grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm > 0 else None,
        normalize_advantages=args.normalize_advantages,
        baseline_value=args.baseline_value,
        baseline_momentum=args.baseline_momentum,
    )
    agent = Agent(policy=policy, config=config, device=device)

    log_path = output_dir / "training_log.csv"
    header = [
        "episode",
        "train_return",
        "train_length",
        "moving_avg_return_20",
        "loss",
        "actor_loss",
        "critic_loss",
        "mean_discounted_return",
        "baseline",
        "episode_time_sec",
        "elapsed_time_sec",
        "eval_mean_return",
        "eval_std_return",
        "eval_mean_length",
    ]

    train_returns: List[float] = []
    best_eval_return = -np.inf
    start_time = time.time()

    print(f"Training {args.algorithm} on {args.env_id} for {args.episodes} episodes using device={device}.")

    for episode in range(1, args.episodes + 1):
        episode_start = time.time()
        state, _ = env.reset(seed=args.seed + episode)
        done = False
        episode_return = 0.0
        episode_length = 0

        while not done:
            action_tensor, action_log_prob = agent.act(state, evaluation=False)
            action = action_tensor.detach().cpu().numpy()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            if action_log_prob is None:
                raise RuntimeError("Training action_log_prob should never be None.")

            agent.store_outcome(
                state=state,
                next_state=next_state,
                action_log_prob=action_log_prob,
                reward=float(reward),
                done=done,
            )

            state = next_state
            episode_return += float(reward)
            episode_length += 1

            if args.max_steps is not None and episode_length >= args.max_steps:
                # Treat a manual cap as terminal for the purpose of the stored trajectory.
                if agent.dones:
                    agent.dones[-1] = True
                done = True

        update_stats = agent.update_policy()
        train_returns.append(episode_return)

        eval_stats = {"eval_mean_return": "", "eval_std_return": "", "eval_mean_length": ""}
        should_eval = args.eval_every > 0 and (episode % args.eval_every == 0 or episode == args.episodes)
        if should_eval:
            eval_stats = evaluate(
                agent=agent,
                env_id=args.env_id,
                seed=args.seed + 10_000 + episode,
                n_episodes=args.eval_episodes,
                max_steps=args.max_steps,
            )
            if eval_stats["eval_mean_return"] > best_eval_return:
                best_eval_return = float(eval_stats["eval_mean_return"])
                agent.save(output_dir / "best_model.pt", extra={"episode": episode, **eval_stats})

        elapsed = time.time() - start_time
        row = {
            "episode": episode,
            "train_return": episode_return,
            "train_length": episode_length,
            "moving_avg_return_20": moving_average(train_returns, window=20),
            "episode_time_sec": time.time() - episode_start,
            "elapsed_time_sec": elapsed,
            **update_stats,
            **eval_stats,
        }
        write_csv_row(log_path, row, header=header)

        if episode == 1 or episode % 10 == 0 or should_eval:
            eval_msg = ""
            if should_eval:
                eval_msg = f" | eval={eval_stats['eval_mean_return']:.2f}±{eval_stats['eval_std_return']:.2f}"
            print(
                f"Episode {episode:04d}/{args.episodes} | "
                f"return={episode_return:.2f} | "
                f"ma20={row['moving_avg_return_20']:.2f} | "
                f"len={episode_length} | "
                f"loss={row['loss']:.4f}"
                f"{eval_msg}"
            )

    agent.save(output_dir / "final_model.pt", extra={"episodes": args.episodes, "best_eval_return": best_eval_return})
    env.close()

    summary = {
        "algorithm": args.algorithm,
        "episodes": args.episodes,
        "final_train_return": train_returns[-1] if train_returns else None,
        "best_eval_return": None if not np.isfinite(best_eval_return) else best_eval_return,
        "mean_last_20_train_return": moving_average(train_returns, window=20),
        "log_path": str(log_path),
        "final_model": str(output_dir / "final_model.pt"),
        "best_model": str(output_dir / "best_model.pt"),
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(json_ready(summary), handle, indent=2)

    print("\nTraining completed.")
    print(json.dumps(json_ready(summary), indent=2))


if __name__ == "__main__":
    main()
