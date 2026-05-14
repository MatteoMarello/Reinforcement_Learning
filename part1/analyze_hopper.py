"""Inspect Gymnasium Hopper-v4 for Phase 2 Task 1.

This script prints the observation/action spaces and MuJoCo model quantities
suggested in the assignment: body names, body masses, number of DoFs, per-body
DoFs, number of actuators, and actuator control ranges.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Hopper-v4 spaces and MuJoCo physical parameters.")
    parser.add_argument("--env-id", type=str, default="Hopper-v4")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(v) for v in value]
    return value


def body_name(model: Any, idx: int) -> str:
    try:
        return str(model.body(idx).name)
    except Exception:
        return str(idx)


def inspect_env(env: gym.Env) -> Dict[str, Any]:
    model = env.unwrapped.model
    bodies: List[Dict[str, Any]] = []

    for idx in range(int(model.nbody)):
        bodies.append(
            {
                "body_id": idx,
                "body_name": body_name(model, idx),
                "mass": float(model.body_mass[idx]),
                "dof_count": int(model.body_dofnum[idx]) if hasattr(model, "body_dofnum") else None,
            }
        )

    return {
        "env_id": env.spec.id if env.spec is not None else None,
        "observation_space": str(env.observation_space),
        "observation_shape": env.observation_space.shape,
        "action_space": str(env.action_space),
        "action_shape": env.action_space.shape,
        "action_low": env.action_space.low,
        "action_high": env.action_space.high,
        "nq_position_coordinates": int(model.nq),
        "nv_degrees_of_freedom": int(model.nv),
        "nu_actuators": int(model.nu),
        "bodies": bodies,
        "actuator_ctrlrange": np.asarray(model.actuator_ctrlrange),
    }


def main() -> None:
    args = parse_args()
    env = gym.make(args.env_id, render_mode="human" if args.render else "rgb_array")
    info = inspect_env(env)
    env.close()

    print(json.dumps(as_jsonable(info), indent=2))
    print("\nInterpretation for the report/code comments:")
    print("- Hopper-v4 uses a continuous observation space and a continuous action space.")
    print("- With default settings, the observation vector has 11 components.")
    print("- The action vector has 3 components, one bounded torque command per actuator.")
    print("- Part 1 of the provided repository contains only the standard Hopper-v4 setup;")
    print("  the source/target domain split starts later in Part 2 with PandaPush.")


if __name__ == "__main__":
    main()
