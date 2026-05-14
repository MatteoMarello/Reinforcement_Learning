# FAIML RL Project — Phase 1 and Phase 2 Completed

This repository has been completed up to **Phase 2 included** and deliberately stops before **Phase 3: Lower/upper bound baselines**.

Implemented scope:

1. **Preliminaries / environment inspection**
   - Hopper-v4 inspection utility.
   - Observation/action space reporting.
   - MuJoCo body mass and DoF reporting.

2. **Train your first RL agent on Hopper-v4**
   - REINFORCE without baseline.
   - REINFORCE with a scalar baseline.
   - One-step Actor-Critic policy gradient.
   - Training loop, deterministic evaluation, CSV logging, checkpoints, and optional learning-curve plotting.

Not implemented yet by design:

- PPO / SAC with Stable-Baselines3.
- PandaPush source/target lower and upper bounds.
- Uniform Domain Randomization.
- Automatic Domain Randomization.

Those items belong to Phase 3 and Phase 4.

---

## Files added or completed

```text
part1/
├── agent.py            # completed REINFORCE, baseline REINFORCE, Actor-Critic
├── train.py            # full training/evaluation/logging pipeline for Hopper-v4
├── evaluate.py         # evaluate a saved checkpoint over N episodes
├── analyze_hopper.py   # inspect Hopper spaces, masses, DoFs, actuators
└── plot_results.py     # plot learning curves from training_log.csv
```

The original `part2/` files are intentionally left as templates for the later phases.

---

## Installation

From the repository root:

```bash
pip install -r requirements.txt
```

If you use Conda, create an environment first, then install the requirements.
MuJoCo rendering is easier on a local Linux machine than on Colab.

---

## Phase 2 Task 1 — inspect Hopper-v4

From `part1/`:

```bash
python analyze_hopper.py
```

This prints:

- observation space,
- action space,
- action bounds,
- MuJoCo position coordinates `nq`, velocity DoFs `nv`, and actuators `nu`,
- body names,
- body masses,
- per-body DoF counts,
- actuator control ranges.

Important interpretation:

- Hopper-v4 has a **continuous observation space**.
- Hopper-v4 has a **continuous action space**.
- With default Gymnasium settings, observations have shape `(11,)`.
- Actions have shape `(3,)` and are bounded in `[-1, 1]`.
- The source/target mass split is not present in `part1`; it starts later with PandaPush in `part2`.

---

## Phase 2 Task 2 — REINFORCE

### REINFORCE without baseline

```bash
cd part1
python train.py \
  --algorithm reinforce \
  --episodes 1000 \
  --run-name reinforce_hopper
```

### REINFORCE with scalar baseline

Using the default moving scalar baseline:

```bash
python train.py \
  --algorithm reinforce_baseline \
  --episodes 1000 \
  --run-name reinforce_baseline_hopper
```

Using a fixed scalar baseline chosen manually:

```bash
python train.py \
  --algorithm reinforce_baseline \
  --baseline-value 100.0 \
  --episodes 1000 \
  --run-name reinforce_fixed_baseline_100
```

The scalar baseline is subtracted from the Monte-Carlo returns before the policy-gradient update. It is independent of the action, so it reduces variance without changing the expected policy-gradient direction.

---

## Phase 2 Task 3 — Actor-Critic

```bash
python train.py \
  --algorithm actor_critic \
  --episodes 1000 \
  --normalize-advantages \
  --run-name actor_critic_hopper
```

Actor-Critic uses:

- the same stochastic actor as REINFORCE,
- a critic network estimating `V(s)`,
- one-step TD targets:

```text
y_t = r_t + gamma * V(s_{t+1}) * (1 - done_t)
```

and the advantage:

```text
A_t = y_t - V(s_t)
```

The actor loss is:

```text
L_actor = - mean(log pi(a_t | s_t) * A_t)
```

The critic loss is:

```text
L_critic = MSE(V(s_t), y_t)
```

---

## Output structure

Every training run creates:

```text
part1/results/<run-name>/
├── config.json
├── env_info.json
├── training_log.csv
├── best_model.pt
├── final_model.pt
└── summary.json
```

`training_log.csv` includes:

- episode number,
- training return,
- episode length,
- moving average return over 20 episodes,
- total loss,
- actor loss,
- critic loss,
- baseline value,
- deterministic evaluation mean/std return.

---

## Evaluate a saved model

```bash
python evaluate.py \
  --checkpoint results/actor_critic_hopper/best_model.pt \
  --episodes 50
```

With rendering:

```bash
python evaluate.py \
  --checkpoint results/actor_critic_hopper/best_model.pt \
  --episodes 5 \
  --render
```

---

## Plot learning curves

```bash
python plot_results.py \
  --log results/actor_critic_hopper/training_log.csv
```

This creates:

```text
results/actor_critic_hopper/learning_curve.png
```

---

## Recommended experimental comparison for the report later

Run at least these three experiments with the same seed and comparable episode budget:

```bash
python train.py --algorithm reinforce --episodes 1000 --seed 42 --run-name reinforce_seed42
python train.py --algorithm reinforce_baseline --episodes 1000 --seed 42 --run-name reinforce_baseline_seed42
python train.py --algorithm actor_critic --episodes 1000 --seed 42 --normalize-advantages --run-name actor_critic_seed42
```

Then compare:

- final deterministic evaluation return,
- average return over the last 20 training episodes,
- training time,
- learning-curve stability,
- convergence speed.

Expected qualitative behavior:

- REINFORCE without baseline is simple but high variance.
- REINFORCE with baseline should usually have lower gradient variance and smoother learning.
- Actor-Critic usually updates with lower variance and can converge faster, but it may introduce bias because the critic is learned and imperfect.

---

## Notes for the later group work

Suggested division for the next phases:

1. One person completes the Stable-Baselines3 PPO/SAC training and evaluation pipeline.
2. One person runs the source→source, source→target, and target→target experiments.
3. One person implements and tests UDR.
4. One person implements ADR and consolidates plots/tables for the report.

The current code is intentionally modular: Phase 2 is self-contained in `part1/`, while `part2/` remains ready for the later PandaPush work.
