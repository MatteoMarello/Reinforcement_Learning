# Phase 3 PPO/SAC Baseline Notes

## Owner
Shaheer Abdullah

## Task
Phase 3: Lower/upper bound baselines using Stable-Baselines3 PPO and SAC for the PandaPush environment.

## Environment check
The PandaPush source and target environments were verified.

- Source cube mass: 1.0 kg
- Target cube mass: 5.0 kg

This confirms the intended sim-to-real style gap, where the policy is trained in a source domain with a lighter cube and tested in a target domain with a heavier cube.

## Algorithms tested
- SAC
- PPO

## Training setup
Both algorithms were trained with 100,000 timesteps.

Training domains:
- Source domain
- Target domain

Evaluation protocol:
- 50 test episodes
- Dense reward
- Deterministic evaluation
- End-effector control

## Required evaluation configurations
- source → source
- source → target
- target → target

## Results

| Algorithm | Configuration | Mean Return | Std Return | Success Rate |
|---|---|---:|---:|---:|
| SAC | source → source | -3.7824 | 1.9043 | 0.16 |
| SAC | source → target | -3.2626 | 2.0270 | 0.24 |
| SAC | target → target | -3.6955 | 1.9670 | 0.18 |
| PPO | source → source | -3.4399 | 2.1882 | 0.26 |
| PPO | source → target | -3.7662 | 1.6252 | 0.12 |
| PPO | target → target | -3.7593 | 2.1208 | 0.22 |

## Comparison
The most important configuration for Phase 3 is source → target, because it represents the lower-bound transfer case. In this setup, SAC achieved a better mean return and a higher success rate than PPO.

SAC source → target:
- Mean return: -3.2626
- Success rate: 0.24

PPO source → target:
- Mean return: -3.7662
- Success rate: 0.12

Since the return values are negative, a less negative value is better. Therefore, SAC performed better in the transfer setting.

## Best algorithm recommendation
Recommended algorithm for the next phase: SAC.

SAC is recommended because it showed better source → target transfer performance than PPO. This makes it the more suitable baseline algorithm for the following domain randomization phase.

## Interpretation
The source → target result is expected to be worse than target → target because the policy is trained with a 1 kg cube in the source environment and tested with a 5 kg cube in the target environment.

In a real sim-to-real setting, direct training on the target real robot is usually avoided because it is expensive, slow, risky, and may damage hardware. Therefore, simulation training is used first, and methods such as domain randomization are used to improve transfer robustness.

