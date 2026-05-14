"""Inspect PandaPush source/target domains and the object mass gap."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from env_utils import make_push_env


def json_ready(value: Any) -> Any:
    """Convert NumPy/Gymnasium values into JSON-serializable objects."""
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        return value.tolist()
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def get_object_mass(env) -> float:
    sim = env.unwrapped.task.sim
    body_id = sim._bodies_idx["object"]
    return float(sim.physics_client.getDynamicsInfo(body_id, -1)[0])


def inspect(env_type: str) -> dict:
    env = make_push_env(env_type=env_type, reward_type="dense", render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    result = {
        "env_type": env_type,
        "observation_space": str(env.observation_space),
        "action_space": str(env.action_space),
        "object_mass": get_object_mass(env),
        "initial_info": json_ready(info),
        "observation_keys": list(obs.keys()),
        "observation_shapes": {key: list(value.shape) for key, value in obs.items()},
    }
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PandaPush source and target domains.")
    parser.add_argument("--env-type", choices=["source", "target", "both"], default="both")
    args = parser.parse_args()

    env_types = ["source", "target"] if args.env_type == "both" else [args.env_type]
    results = [inspect(env_type) for env_type in env_types]
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
