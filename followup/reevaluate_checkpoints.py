"""Re-evaluate the released SAC checkpoints with seeded resets and confidence intervals.

Follow-up analysis added after the course submission. It does not train anything: it loads the
checkpoints committed by the group and evaluates them on the source (1 kg) and target (5 kg)
PandaPush domains with a fixed, per-episode reset seed, so every configuration is evaluated on
the same sequence of initial states.

Motivation: `part2/eval_sb3.py` seeds only the action space, not `env.reset()`, so two
evaluations of the same checkpoint see different initial states. This explains why the
source-to-target lower bound appears as 0.58 success in the Phase 3 table and 0.50 in the
Phase 4 table. Here we quantify that evaluation noise.

Usage (from the repository root):
    python followup/reevaluate_checkpoints.py --episodes 50 --eval-seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part2"))

from env_utils import load_model_auto, make_push_env  # noqa: E402

CHECKPOINTS = {
    "sac_source": "shaheer_phase3_reproducibility/checkpoints/sac_source_500k_final_model.zip",
    "sac_target": "shaheer_phase3_reproducibility/checkpoints/sac_target_500k_final_model.zip",
    "sac_udr": "results/part2/sac_udr_source_500k_seed0/final_model.zip",
    "sac_adr": "results/part2/sac_adr_source_500k_seed0/final_model.zip",
}

# (label, checkpoint, test domain) — mirrors the rows of the Phase 3/4 tables.
CONFIGS = [
    ("source_to_source", "sac_source", "source"),
    ("source_to_target", "sac_source", "target"),
    ("target_to_target", "sac_target", "target"),
    ("udr_source_to_source", "sac_udr", "source"),
    ("udr_source_to_target", "sac_udr", "target"),
    ("adr_source_to_source", "sac_adr", "source"),
    ("adr_source_to_target", "sac_adr", "target"),
]


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def run_episodes(model, env, reset_seeds: list[int]) -> tuple[list[float], list[float]]:
    returns, successes = [], []
    for s in reset_seeds:
        obs, info = env.reset(seed=s)
        done, ep_ret = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            done = terminated or truncated
        returns.append(ep_ret)
        successes.append(float(info.get("is_success", 0.0)))
    return returns, successes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=50, help="episodes per evaluation seed")
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out-dir", type=str, default="followup/results")
    args = ap.parse_args()

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    models = {k: load_model_auto(str(ROOT / p), "sac") for k, p in CHECKPOINTS.items()}

    rows, per_episode = [], []
    for label, ckpt, domain in CONFIGS:
        env = make_push_env(env_type=domain, reward_type="dense", control_type="ee")
        seed_success, all_ret, all_succ = [], [], []
        t0 = time.time()
        for es in args.eval_seeds:
            reset_seeds = [10_000 * es + i for i in range(args.episodes)]
            r, s = run_episodes(models[ckpt], env, reset_seeds)
            seed_success.append(float(np.mean(s)))
            all_ret += r
            all_succ += s
            per_episode += [
                {"config": label, "eval_seed": es, "reset_seed": rs, "return": rr, "success": ss}
                for rs, rr, ss in zip(reset_seeds, r, s)
            ]
        env.close()
        n, k = len(all_succ), int(sum(all_succ))
        lo, hi = wilson(k, n)
        rows.append({
            "config": label,
            "checkpoint": CHECKPOINTS[ckpt],
            "test_domain": domain,
            "episodes_total": n,
            "success_rate": k / n,
            "success_wilson95_low": lo,
            "success_wilson95_high": hi,
            "success_min_over_seeds": min(seed_success),
            "success_max_over_seeds": max(seed_success),
            "mean_return": float(np.mean(all_ret)),
            "std_return": float(np.std(all_ret)),
            "seconds": round(time.time() - t0, 1),
        })
        print(f"{label:24s} success={k/n:.3f} [{lo:.3f},{hi:.3f}] "
              f"per-seed {min(seed_success):.2f}-{max(seed_success):.2f} return={np.mean(all_ret):.3f}")

    with open(out / "reevaluation_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "reevaluation_per_episode.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_episode[0]))
        w.writeheader()
        w.writerows(per_episode)
    import gymnasium, stable_baselines3, torch  # noqa: E401
    meta = {
        "episodes_per_seed": args.episodes,
        "eval_seeds": args.eval_seeds,
        "deterministic_policy": True,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "gymnasium": gymnasium.__version__,
    }
    (out / "reevaluation_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
