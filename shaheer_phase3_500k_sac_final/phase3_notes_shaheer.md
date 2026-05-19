# Phase 3 Final Improved Run Notes

## Owner
Shaheer Abdullah

## Task
Phase 3: PPO/SAC lower-upper bound baselines for PandaPush using Stable-Baselines3.

## Official baseline already pushed
The 100k PPO/SAC comparison was pushed as the main Phase 3 baseline deliverable.

## Final improved run
Based on the 100k comparison, SAC was selected as the best-performing algorithm. Therefore, the final improved 500k run was performed for SAC.

## Environment check
The PandaPush environments were verified as:

- Source cube mass: 1.0 kg
- Target cube mass: 5.0 kg

This creates the intended sim-to-real style gap.

## Algorithm
- SAC

## Timesteps
- 500,000 timesteps

## Seed
- Training seed: 0
- Evaluation seed: 1234

## Evaluation protocol
- 50 test episodes
- Dense reward
- Deterministic evaluation
- End-effector control

## Required configurations

| Configuration | Mean Return | Std Return | Success Rate |
|---|---:|---:|---:|
| source → source | -2.2759 | 2.0831 | 0.52 |
| source → target | -2.0733 | 1.9796 | 0.58 |
| target → target | -0.4746 | 0.2843 | 1.00 |

## Interpretation
The source → target configuration represents the lower-bound transfer case because the policy is trained on the source environment with a 1 kg cube and tested on the target environment with a 5 kg cube.

The target → target configuration represents the upper-bound case because the policy is trained and tested directly on the target environment.

The 500k SAC run improved the results significantly compared with the earlier 100k comparison. SAC reached a 0.58 success rate in the source → target transfer setting and a 1.00 success rate in the target → target upper-bound setting.

## Recommendation
SAC is recommended for the next phase, including domain randomization, because it showed the strongest transfer performance and the best target-domain performance.


