"""Utility functions shared by the Phase 3/4 PandaPush scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Type

import gymnasium as gym

# Make the bundled editable panda-gym package importable even before the user
# runs `pip install -e part2/panda-gym`.  Installing it is still recommended,
# but this keeps the scripts convenient when run from the repository root.
_THIS_DIR = Path(__file__).resolve().parent
_PANDA_GYM_PATH = _THIS_DIR / "panda-gym"
if _PANDA_GYM_PATH.exists() and str(_PANDA_GYM_PATH) not in sys.path:
    sys.path.insert(0, str(_PANDA_GYM_PATH))

import panda_gym  # noqa: F401,E402 - registers Panda environments
from stable_baselines3 import PPO, SAC  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from rand_wrapper import RandomizationWrapper  # noqa: E402

ALGORITHMS: Dict[str, Type] = {"ppo": PPO, "sac": SAC}


def parse_mass_range(values: Optional[Tuple[float, float]]) -> Tuple[float, float]:
    if values is None:
        return (0.5, 8.0)
    low, high = float(values[0]), float(values[1])
    if low <= 0 or high <= 0 or low >= high:
        raise ValueError("Mass range must be two positive numbers: low high, with low < high.")
    return (low, high)


def make_push_env(
    env_type: str = "source",
    reward_type: str = "dense",
    render_mode: str = "rgb_array",
    control_type: str = "ee",
    sampling_strategy: str = "none",
    mass_range: Tuple[float, float] = (0.5, 8.0),
    seed: Optional[int] = None,
    monitor_dir: Optional[str] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
) -> gym.Env:
    """Create a PandaPush-v3 environment, optionally wrapped for DR.

    Returns a standard Gymnasium environment with dict observations.  SB3 will
    use `MultiInputPolicy` for PPO/SAC.
    """
    if env_type not in {"source", "target"}:
        raise ValueError("env_type must be 'source' or 'target'.")
    if sampling_strategy not in {"none", "udr", "adr"}:
        raise ValueError("sampling_strategy must be one of: none, udr, adr.")

    env = gym.make(
        "PandaPush-v3",
        render_mode=render_mode,
        type=env_type,
        reward_type=reward_type,
        control_type=control_type,
    )

    if sampling_strategy != "none":
        kwargs = dict(wrapper_kwargs or {})
        env = RandomizationWrapper(
            env,
            mass_range=mass_range,
            mode=sampling_strategy,
            seed=seed,
            **kwargs,
        )

    if seed is not None:
        env.action_space.seed(seed)

    if monitor_dir is not None:
        os.makedirs(monitor_dir, exist_ok=True)
        env = Monitor(env, filename=os.path.join(monitor_dir, "monitor.csv"))
    else:
        env = Monitor(env)

    return env


def make_push_env_fn(
    rank: int,
    env_type: str,
    reward_type: str,
    render_mode: str,
    control_type: str,
    sampling_strategy: str,
    mass_range: Tuple[float, float],
    base_seed: Optional[int],
    monitor_root: Optional[str],
    wrapper_kwargs: Optional[Dict[str, Any]],
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env_seed = None if base_seed is None else int(base_seed) + rank
        monitor_dir = None if monitor_root is None else os.path.join(monitor_root, f"env_{rank}")
        return make_push_env(
            env_type=env_type,
            reward_type=reward_type,
            render_mode=render_mode,
            control_type=control_type,
            sampling_strategy=sampling_strategy,
            mass_range=mass_range,
            seed=env_seed,
            monitor_dir=monitor_dir,
            wrapper_kwargs=wrapper_kwargs,
        )

    return _init


def get_algorithm_class(name: str):
    name = name.lower()
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{name}'. Valid choices: {sorted(ALGORITHMS)}")
    return ALGORITHMS[name]


def load_model_auto(model_path: str, algorithm: str = "auto"):
    """Load an SB3 model when the caller may not remember the algorithm."""
    if algorithm != "auto":
        return get_algorithm_class(algorithm).load(model_path)

    errors = []
    for name, cls in ALGORITHMS.items():
        try:
            return cls.load(model_path)
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"{name}: {exc}")
    joined = "\n".join(errors)
    raise ValueError(f"Could not infer algorithm for model '{model_path}'. Errors:\n{joined}")
