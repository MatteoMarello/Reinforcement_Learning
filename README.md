# FAIMDL 2026 - Reinforcement Learning Project

Public repository for the group project of Fundamentals of Artificial Intelligence, Machine and Deep Learning at Politecnico di Torino, AY 2025/2026.

The project focuses on Reinforcement Learning for robotic control, sim-to-real transfer, and Domain Randomization.

## Project objectives

The work is organized into four main steps:

1. Familiarize with the core concepts of Reinforcement Learning.
2. Implement simple Reinforcement Learning algorithms.
3. Use state-of-the-art RL algorithms to solve a robotic task.
4. Implement Domain Randomization in a robotic environment.

Expected implementation:

- REINFORCE with baseline.
- Actor-Critic.
- PPO and SAC with Stable-Baselines3.
- MuJoCo or Gymnasium robotic environments.
- PandaPush goal-conditioned task.
- Uniform Domain Randomization and optional Automatic Domain Randomization.

## Repository structure

- src: source code.
- configs: experiment configurations.
- scripts: launch scripts.
- notebooks: exploratory notebooks.
- results: generated plots, logs, and evaluation outputs.
- models: saved trained models.
- report: report-related material.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Example commands

```bash
python src/reinforce.py --config configs/hopper.yaml
python src/actor_critic.py --config configs/hopper.yaml
python src/train_sb3.py --config configs/panda_push.yaml --algo ppo
python src/train_sb3.py --config configs/panda_push.yaml --algo sac --domain-randomization uniform
```

## Submission reminder

The final submission requires a public GitHub repository and a PDF report. The repository link must be included at the end of the abstract in the report. After the submission deadline, the code should not be modified.

## Authors

Group members to be added.
