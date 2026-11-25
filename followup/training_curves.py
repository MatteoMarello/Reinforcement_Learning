"""Target-domain success during training for the UDR and ADR runs (reads the EvalCallback logs, no new runs).

Each run evaluated the current policy on the target domain (5 kg) every 10k steps with 20 episodes
(`results/part2/sac_*_source_500k_seed0/evaluations/evaluations.npz`).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
fig, ax = plt.subplots(figsize=(6.5, 3.4))
for name, color in (("udr", "#1f4e79"), ("adr", "#c55a11")):
    d = np.load(ROOT / f"results/part2/sac_{name}_source_500k_seed0/evaluations/evaluations.npz")
    s = d["successes"].mean(1)
    ax.plot(d["timesteps"] / 1000, s, color=color, alpha=0.35, lw=1)
    ax.plot(d["timesteps"][4:] / 1000, np.convolve(s, np.ones(5) / 5, "valid"), color=color, lw=2,
            label=f"{name.upper()} (5-eval moving average)")
    last = d["successes"][-10:].ravel()
    print(f"{name}: target success over the last 10 evaluations = {last.mean():.3f} ({last.size} episodes)")
ax.set_xlabel("training steps (thousands)")
ax.set_ylabel("target success rate")
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
ax.set_title("Target-domain success during training (20 episodes per evaluation)", fontsize=9)
fig.tight_layout()
fig.savefig(ROOT / "followup/results/training_target_success.png", dpi=150)
