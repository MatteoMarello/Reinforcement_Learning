# FAIML RL Project — Phase 3 and Phase 4 completed

This repository now contains a complete implementation of the project pipeline up to Phase 4.  The paper/report is intentionally not included, because it will be prepared later.

## Implemented scope

### Phase 3 — Lower/upper bound baselines

Implemented in `part2/train_sb3.py`, `part2/eval_sb3.py`, `part2/inspect_push.py`, and `part2/run_part2_experiments.py`.

The code supports:

- PPO training on `PandaPush-v3` via Stable-Baselines3;
- SAC training on `PandaPush-v3` via Stable-Baselines3;
- source-domain training, where the cube mass is 1 kg;
- target-domain training, where the cube mass is 5 kg;
- deterministic evaluation over any number of episodes;
- source→source evaluation;
- source→target evaluation, the lower-bound transfer baseline;
- target→target evaluation, the upper-bound in-simulation baseline;
- checkpoint saving;
- periodic evaluation and best-model saving;
- TensorBoard logging;
- optional Weights & Biases integration.

### Phase 4 — Domain Randomization

Implemented in `part2/rand_wrapper.py` and integrated into `part2/train_sb3.py`.

The code supports:

- Uniform Domain Randomization (UDR) on the cube mass;
- Automatic Domain Randomization (ADR) on the cube mass;
- evaluation of UDR/ADR policies on both source and target fixed-mass domains;
- logging of current sampled mass and ADR interval boundaries.

UDR samples a new cube mass at each reset from a fixed uniform range.  ADR starts with a narrow interval around the nominal source mass and expands/contracts its interval according to recent success at the current boundaries.

## Installation

From the repository root:

```bash
pip install -r requirements.txt
cd part2/panda-gym
pip install -e .
cd ../..
```

The editable installation of `part2/panda-gym` is recommended because the project uses the bundled PandaPush variant with the explicit `type="source"` / `type="target"` mass setting.

## Sanity check

Inspect the source/target environments:

```bash
python part2/inspect_push.py --env-type both
```

Expected conceptual result:

- source cube mass: `1.0` kg;
- target cube mass: `5.0` kg;
- observation space: dictionary observation with `observation`, `achieved_goal`, `desired_goal`;
- action space: continuous Box.

## Phase 3 commands

Train PPO on source and target:

```bash
python part2/train_sb3.py --algorithm ppo --env-type source --sampling-strategy none --timesteps 500000 --run-name ppo_source
python part2/train_sb3.py --algorithm ppo --env-type target --sampling-strategy none --timesteps 500000 --run-name ppo_target
```

Train SAC on source and target:

```bash
python part2/train_sb3.py --algorithm sac --env-type source --sampling-strategy none --timesteps 500000 --run-name sac_source
python part2/train_sb3.py --algorithm sac --env-type target --sampling-strategy none --timesteps 500000 --run-name sac_target
```

Evaluate the baselines:

```bash
python part2/eval_sb3.py --algorithm sac --model-path results/part2/sac_source/final_model.zip --env-type source --episodes 50 --label source_to_source --output-csv results/part2/summaries/sac_baselines.csv
python part2/eval_sb3.py --algorithm sac --model-path results/part2/sac_source/final_model.zip --env-type target --episodes 50 --label source_to_target_lower_bound --output-csv results/part2/summaries/sac_baselines.csv
python part2/eval_sb3.py --algorithm sac --model-path results/part2/sac_target/final_model.zip --env-type target --episodes 50 --label target_to_target_upper_bound --output-csv results/part2/summaries/sac_baselines.csv
```

Repeat the same commands with `--algorithm ppo` for PPO.

## Phase 4 commands

Train UDR on the source domain:

```bash
python part2/train_sb3.py \
  --algorithm sac \
  --env-type source \
  --sampling-strategy udr \
  --mass-range 0.5 8.0 \
  --timesteps 500000 \
  --run-name sac_udr_source
```

Train ADR on the source domain:

```bash
python part2/train_sb3.py \
  --algorithm sac \
  --env-type source \
  --sampling-strategy adr \
  --mass-range 0.5 8.0 \
  --timesteps 500000 \
  --run-name sac_adr_source
```

Evaluate UDR/ADR policies on fixed source and target domains:

```bash
python part2/eval_sb3.py --algorithm sac --model-path results/part2/sac_udr_source/final_model.zip --env-type source --episodes 50 --label udr_source_to_source --output-csv results/part2/summaries/sac_dr.csv
python part2/eval_sb3.py --algorithm sac --model-path results/part2/sac_udr_source/final_model.zip --env-type target --episodes 50 --label udr_source_to_target --output-csv results/part2/summaries/sac_dr.csv
python part2/eval_sb3.py --algorithm sac --model-path results/part2/sac_adr_source/final_model.zip --env-type source --episodes 50 --label adr_source_to_source --output-csv results/part2/summaries/sac_dr.csv
python part2/eval_sb3.py --algorithm sac --model-path results/part2/sac_adr_source/final_model.zip --env-type target --episodes 50 --label adr_source_to_target --output-csv results/part2/summaries/sac_dr.csv
```

## Convenience launcher

To run the full standard matrix with SAC:

```bash
python part2/run_part2_experiments.py \
  --algorithm sac \
  --timesteps 500000 \
  --episodes 50 \
  --include-udr \
  --include-adr
```

To preview the commands without executing them:

```bash
python part2/run_part2_experiments.py --algorithm sac --include-udr --include-adr --dry-run
```

## Plot results

```bash
python part2/plot_part2_results.py \
  --csv results/part2/summaries/sac_phase3_phase4_summary_seed0.csv
```

## Suggested division of work for the group

1. One person runs PPO source/target and produces the Phase 3 PPO lower/upper baselines.
2. One person runs SAC source/target and chooses the better algorithm for Phase 4.
3. One person runs UDR sweeps with several mass ranges, for example `[0.5, 4.0]`, `[0.5, 6.0]`, `[0.5, 8.0]`.
4. One person runs ADR sweeps by tuning `adr_step_size`, `adr_window_size`, and the success thresholds.

For the final report, compare at least:

- source→source;
- source→target;
- target→target;
- UDR source→target;
- ADR source→target.

Use average return over 50 evaluation episodes, plus success rate when available.
