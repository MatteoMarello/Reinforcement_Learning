"""Train PPO/SAC agents on PandaPush-v3 with optional UDR/ADR.

Examples
--------
Train the lower-bound source policy:
    python part2/train_sb3.py --algorithm sac --env-type source --sampling-strategy none

Train the upper-bound target policy:
    python part2/train_sb3.py --algorithm sac --env-type target --sampling-strategy none

Train a domain-randomized policy on the source domain:
    python part2/train_sb3.py --algorithm sac --env-type source --sampling-strategy udr --mass-range 0.5 8.0
    python part2/train_sb3.py --algorithm sac --env-type source --sampling-strategy adr --mass-range 0.5 8.0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from env_utils import get_algorithm_class, make_push_env, make_push_env_fn, parse_mass_range


class DomainRandomizationLoggingCallback(BaseCallback):
    """Log UDR/ADR state to SB3's logger every `log_freq` timesteps."""

    def __init__(self, log_freq: int = 1_000, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.log_freq = int(max(1, log_freq))

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True
        try:
            states = self.training_env.env_method("get_randomization_state")
        except Exception:
            return True
        if not states:
            return True

        numeric_keys = [
            "current_mass",
            "mass_min",
            "mass_max",
            "global_mass_min",
            "global_mass_max",
            "episodes_seen",
            "lower_boundary_success_rate",
            "upper_boundary_success_rate",
        ]
        for key in numeric_keys:
            values = [state.get(key) for state in states if state.get(key) is not None]
            if values:
                self.logger.record(f"domain_randomization/{key}", sum(values) / len(values))
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO/SAC on PandaPush-v3 with optional UDR/ADR.")

    parser.add_argument("--algorithm", type=str, default="sac", choices=["ppo", "sac"])
    parser.add_argument("--sampling-strategy", type=str, default="none", choices=["none", "udr", "adr"])
    parser.add_argument("--env-type", type=str, default="source", choices=["source", "target"])
    parser.add_argument("--reward-type", type=str, default="dense", choices=["dense", "sparse"])
    parser.add_argument("--control-type", type=str, default="ee", choices=["ee", "joints"])

    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--mass-range", type=float, nargs=2, default=(0.5, 8.0), metavar=("LOW", "HIGH"))
    parser.add_argument("--adr-initial-delta", type=float, default=0.1)
    parser.add_argument("--adr-step-size", type=float, default=0.25)
    parser.add_argument("--adr-boundary-prob", type=float, default=0.5)
    parser.add_argument("--adr-window-size", type=int, default=10)
    parser.add_argument("--adr-high-threshold", type=float, default=0.8)
    parser.add_argument("--adr-low-threshold", type=float, default=0.2)
    parser.add_argument("--adr-min-range-width", type=float, default=0.1)
    parser.add_argument("--verbose-randomization", action="store_true")

    parser.add_argument("--eval-env-type", type=str, default="target", choices=["source", "target"])
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--log-freq", type=int, default=1_000)

    parser.add_argument("--save-dir", type=str, default="results/part2")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="faiml-rl-project")

    # PPO-specific
    parser.add_argument("--ppo-n-steps", type=int, default=2048)
    parser.add_argument("--ppo-batch-size", type=int, default=64)
    parser.add_argument("--ppo-n-epochs", type=int, default=10)
    parser.add_argument("--ppo-clip-range", type=float, default=0.2)

    # SAC-specific
    parser.add_argument("--sac-buffer-size", type=int, default=1_000_000)
    parser.add_argument("--sac-learning-starts", type=int, default=1_000)
    parser.add_argument("--sac-batch-size", type=int, default=256)
    parser.add_argument("--sac-tau", type=float, default=0.05)
    parser.add_argument("--sac-train-freq", type=int, default=1)
    parser.add_argument("--sac-gradient-steps", type=int, default=1)

    return parser.parse_args()


def make_wrapper_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "adr_initial_delta": args.adr_initial_delta,
        "adr_step_size": args.adr_step_size,
        "adr_boundary_prob": args.adr_boundary_prob,
        "adr_window_size": args.adr_window_size,
        "adr_high_threshold": args.adr_high_threshold,
        "adr_low_threshold": args.adr_low_threshold,
        "adr_min_range_width": args.adr_min_range_width,
        "verbose": args.verbose_randomization,
    }


def make_run_dir(args: argparse.Namespace) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or (
        f"{args.algorithm}_{args.sampling_strategy}_{args.env_type}_"
        f"eval-{args.eval_env_type}_{args.timesteps // 1000}k_seed{args.seed}_{timestamp}"
    )
    run_dir = Path(args.save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "best_model").mkdir(exist_ok=True)
    (run_dir / "monitor").mkdir(exist_ok=True)
    return run_dir


def save_config(args: argparse.Namespace, run_dir: Path, mass_range: Tuple[float, float]) -> None:
    config = vars(args).copy()
    config["mass_range"] = list(mass_range)
    config["run_dir"] = str(run_dir)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def create_vec_env(args: argparse.Namespace, run_dir: Path, mass_range: Tuple[float, float]):
    wrapper_kwargs = make_wrapper_kwargs(args)
    env_fns = [
        make_push_env_fn(
            rank=i,
            env_type=args.env_type,
            reward_type=args.reward_type,
            render_mode="rgb_array",
            control_type=args.control_type,
            sampling_strategy=args.sampling_strategy,
            mass_range=mass_range,
            base_seed=args.seed,
            monitor_root=str(run_dir / "monitor"),
            wrapper_kwargs=wrapper_kwargs,
        )
        for i in range(args.n_envs)
    ]
    return DummyVecEnv(env_fns)


def create_model(args: argparse.Namespace, env, run_dir: Path):
    algorithm_cls = get_algorithm_class(args.algorithm)
    common_kwargs: Dict[str, Any] = dict(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        verbose=1,
        tensorboard_log=str(run_dir / "tensorboard"),
        seed=args.seed,
        device=args.device,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    if args.algorithm == "ppo":
        return algorithm_cls(
            **common_kwargs,
            n_steps=args.ppo_n_steps,
            batch_size=args.ppo_batch_size,
            n_epochs=args.ppo_n_epochs,
            clip_range=args.ppo_clip_range,
        )

    return algorithm_cls(
        **common_kwargs,
        buffer_size=args.sac_buffer_size,
        learning_starts=args.sac_learning_starts,
        batch_size=args.sac_batch_size,
        tau=args.sac_tau,
        train_freq=args.sac_train_freq,
        gradient_steps=args.sac_gradient_steps,
    )


def create_callbacks(args: argparse.Namespace, run_dir: Path, mass_range: Tuple[float, float]) -> CallbackList:
    eval_env = make_push_env(
        env_type=args.eval_env_type,
        reward_type=args.reward_type,
        render_mode="rgb_array",
        control_type=args.control_type,
        sampling_strategy="none",
        mass_range=mass_range,
        seed=args.seed + 10_000,
    )

    callbacks: List[BaseCallback] = [
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // max(args.n_envs, 1), 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix=f"{args.algorithm}_{args.sampling_strategy}_{args.env_type}",
            save_replay_buffer=False,
            save_vecnormalize=False,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "best_model"),
            log_path=str(run_dir / "evaluations"),
            eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            render=False,
        ),
        DomainRandomizationLoggingCallback(log_freq=args.log_freq),
    ]

    if args.use_wandb:
        try:
            import wandb
            from wandb.integration.sb3 import WandbCallback

            wandb.init(
                project=args.wandb_project,
                name=run_dir.name,
                config=vars(args),
                sync_tensorboard=True,
                monitor_gym=False,
                save_code=True,
            )
            callbacks.append(WandbCallback(model_save_path=str(run_dir / "wandb_models"), verbose=1))
        except Exception as exc:
            print(f"[warning] W&B requested but could not be initialized: {exc}")

    return CallbackList(callbacks)


def main() -> None:
    args = parse_args()
    mass_range = parse_mass_range(args.mass_range)
    run_dir = make_run_dir(args)
    save_config(args, run_dir, mass_range)

    print("=== Training configuration ===")
    print(json.dumps({**vars(args), "run_dir": str(run_dir), "mass_range": mass_range}, indent=2, sort_keys=True))

    env = create_vec_env(args, run_dir, mass_range)
    model = create_model(args, env, run_dir)
    callbacks = create_callbacks(args, run_dir, mass_range)

    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=args.progress_bar)

    final_model_path = run_dir / "final_model.zip"
    model.save(str(final_model_path))
    print(f"\nSaved final model to: {final_model_path}")
    print(f"Best evaluation model, if created, is in: {run_dir / 'best_model' / 'best_model.zip'}")

    env.close()


if __name__ == "__main__":
    main()
