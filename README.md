# Reinforcement Learning for Robotic Control under Sim-to-Sim Domain Shift

Public repository for the **Fundamentals of Artificial Intelligence, Machine and Deep Learning** Reinforcement Learning project, AY 2025/2026, Politecnico di Torino.

**Group 68 — Reinforcement Learning**

| Student | Student ID | Role |
|---|---:|---|
| Samuel Clemos | 362643 | Phase 1/2: Hopper, REINFORCE, baseline, Actor-Critic |
| Matteo Marello | 359719 | Paper, integration, final consistency checks |
| Shaheer Abdullah | 353868 | Phase 3: PPO/SAC baselines |
| Umesh Bansari | 351863 | Phase 4: UDR/ADR Domain Randomization |

GitHub repository: <https://github.com/MatteoMarello/Reinforcement_Learning>

---

## 1. Project overview

This project studies **Reinforcement Learning (RL) for robotic control** and the **sim-to-real transfer problem** in a simplified sim-to-sim setting.

The project is divided into four phases:

1. **Preliminaries**  
   Understand the core RL framework, the agent-environment loop, robotic control tasks, and the sim-to-real problem.

2. **Train basic RL agents from scratch**  
   Implement and compare basic policy-gradient algorithms on the `Hopper-v4` MuJoCo environment:
   - REINFORCE without baseline;
   - REINFORCE with a constant scalar baseline;
   - Actor-Critic.

3. **Lower/upper bound baselines on PandaPush**  
   Use Stable-Baselines3 to train advanced RL agents on a PandaPush robotic manipulation task:
   - PPO;
   - SAC;
   - source-to-source evaluation;
   - source-to-target lower-bound evaluation;
   - target-to-target upper-bound evaluation.

4. **Domain Randomization**  
   Implement and evaluate:
   - Uniform Domain Randomization (UDR);
   - Automatic Domain Randomization (ADR).

The source and target PandaPush environments differ in the cube mass:

| Environment | Cube mass |
|---|---:|
| Source | 1 kg |
| Target | 5 kg |

This manually introduced mismatch simulates a **reality gap**. The goal of Domain Randomization is to train policies that remain robust when transferred from the source domain to the target domain.

---

## 2. Repository structure

```text
Reinforcement_Learning-main/
├── README.md
├── requirements.txt
├── paper/
│   ├── 68_RL_s362643_s359719_s353868_s351863_Clemos_Marello_Abdullah_Bansari.pdf
│   └── overleaf_source/
├── PHASE_1_2_COMPLETED.md
├── PHASE_3_4_COMPLETED.md
│
├── part1/
│   ├── agent.py                 # REINFORCE, baseline REINFORCE, Actor-Critic
│   ├── train.py                 # Hopper training loop
│   ├── evaluate.py              # Hopper checkpoint evaluation
│   ├── analyze_hopper.py        # Hopper environment inspection
│   ├── plot_results.py          # Hopper learning-curve plotting
│   ├── test_random_policy.py
│   └── colab_template/
│
├── part2/
│   ├── env_utils.py             # PandaPush environment/model utilities
│   ├── train_sb3.py             # PPO/SAC training with Stable-Baselines3
│   ├── eval_sb3.py              # PPO/SAC evaluation over N episodes
│   ├── inspect_push.py          # Source/target mass inspection
│   ├── rand_wrapper.py          # UDR/ADR wrappers
│   ├── run_part2_experiments.py # Standard Phase 3/4 experiment launcher
│   ├── plot_part2_results.py
│   └── panda-gym/               # Bundled PandaGym implementation
│
├── samuel_part/
│   ├── logs/
│   ├── plots/
│   └── reports/
│       ├── comparison_metrics.csv
│       ├── hyperparameters_used.csv
│       └── SAMUEL_PHASE1_2_REPORT.md
│
├── shaheer_phase3_100k_deliverables/
├── shaheer_phase3_500k_sac_final/
├── shaheer_phase3_reproducibility/
│
└── results/
    └── part2/
        ├── sac_udr_source_500k_seed0/
        ├── sac_adr_source_500k_seed0/
        └── summaries/
            ├── umesh_phase4_final_results.csv
            ├── umesh_phase4_notes.txt
            ├── sac_phase3_phase4_summary_seed0.csv
            └── sac_phase3_phase4_summary_seed0.png
```

---

## 3. Installation

A clean Python environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The project uses a bundled version of PandaGym for the PandaPush task. Install it in editable mode:

```bash
cd part2/panda-gym
pip install -e .
cd ../..
```

If `pybullet` installation fails on a local machine, use Python 3.10/3.11 or run the PandaPush experiments on Google Colab.

---

## 4. Quick sanity checks

Run these commands from the repository root.

### Compile all project scripts

```bash
python -m py_compile part1/*.py part2/*.py
```

### Inspect Hopper

```bash
python part1/analyze_hopper.py
```

Expected conceptual result:

- continuous observation space;
- continuous action space;
- action dimension equal to 3;
- MuJoCo body masses, degrees of freedom, and actuator information printed.

### Inspect PandaPush source/target environments

```bash
python part2/inspect_push.py --env-type both
```

Expected conceptual result:

- source cube mass: 1 kg;
- target cube mass: 5 kg;
- continuous action space;
- goal-conditioned dictionary observation.

### Dry-run the Phase 3/4 launcher

```bash
python part2/run_part2_experiments.py --algorithm sac --include-udr --include-adr --dry-run
```

This prints the training and evaluation commands without running the full experiments.

---

## 5. Phase 1/2 — Hopper from-scratch algorithms

### Implemented algorithms

The following methods are implemented from scratch in `part1/agent.py`:

- `reinforce`;
- `reinforce_baseline`;
- `actor_critic`.

Training is handled by `part1/train.py`.

### Example training commands

REINFORCE without baseline:

```bash
python part1/train.py \
  --algorithm reinforce \
  --episodes 1000 \
  --seed 42 \
  --eval-every 100 \
  --run-name phase2_reinforce_bl0_1000
```

REINFORCE with constant baseline 20:

```bash
python part1/train.py \
  --algorithm reinforce_baseline \
  --baseline-value 20 \
  --episodes 1000 \
  --seed 42 \
  --eval-every 100 \
  --run-name phase2_reinforce_bl20_1000
```

Actor-Critic:

```bash
python part1/train.py \
  --algorithm actor_critic \
  --normalize-advantages \
  --episodes 1000 \
  --seed 42 \
  --eval-every 100 \
  --run-name phase2_actor_critic_1000
```

Long final runs:

```bash
python part1/train.py \
  --algorithm reinforce_baseline \
  --baseline-value 20 \
  --episodes 15000 \
  --seed 42 \
  --eval-every 100 \
  --run-name final_reinforce_bl20_15000

python part1/train.py \
  --algorithm actor_critic \
  --normalize-advantages \
  --episodes 15000 \
  --seed 42 \
  --eval-every 100 \
  --run-name final_actor_critic_15000
```

### Official Phase 1/2 results

Official aggregate results are stored in:

```text
samuel_part/reports/comparison_metrics.csv
samuel_part/reports/hyperparameters_used.csv
samuel_part/plots/
```

Higher return is better.

| Run | Algorithm | Episodes | Baseline | Best eval return | Final eval return |
|---|---|---:|---:|---:|---:|
| `phase2_reinforce_bl0_1000` | REINFORCE | 1000 | - | 346.675 | 243.164 |
| `phase2_reinforce_bl20_1000` | REINFORCE + baseline | 1000 | 20 | 223.212 | 219.723 |
| `phase2_reinforce_bl1000_1000` | REINFORCE + baseline | 1000 | 1000 | 181.041 | 97.763 |
| `phase2_actor_critic_1000` | Actor-Critic | 1000 | - | 466.191 | 466.191 |
| `final_reinforce_bl20_15000` | REINFORCE + baseline | 15000 | 20 | 444.293 | 321.348 |
| `final_actor_critic_15000` | Actor-Critic | 15000 | - | 1010.380 | 168.042 |

Main interpretation:

- Actor-Critic performs best in the 1000-episode comparison and reaches the highest peak evaluation.
- The long Actor-Critic run is unstable: it reaches a high best evaluation return, but its final evaluation return drops.
- REINFORCE with baseline 20 gives a more stable final long-run result than the long Actor-Critic run.
- A very large baseline value such as 1000 hurts the REINFORCE update because it distorts the advantage signal.

---

## 6. Phase 3 — PPO/SAC lower and upper bound baselines

### Implemented pipeline

Phase 3 is implemented with Stable-Baselines3 in:

```text
part2/train_sb3.py
part2/eval_sb3.py
part2/run_part2_experiments.py
```

The task is `PandaPush-v3`, using:

- dense reward;
- end-effector control;
- deterministic evaluation;
- 50 evaluation episodes;
- training seed 0;
- evaluation seed 1234.

### PPO/SAC 100k comparison

The 100k PPO/SAC comparison is stored in:

```text
shaheer_phase3_100k_deliverables/
```

Commands used:

```bash
python part2/run_part2_experiments.py --algorithm sac --timesteps 100000 --episodes 50
python part2/run_part2_experiments.py --algorithm ppo --timesteps 100000 --episodes 50
```

At 100k timesteps, SAC showed better transfer performance than PPO in the source-to-target setting, so SAC was selected for the final 500k baseline and for the Domain Randomization phase.

### Official Phase 3 SAC 500k results

Official Phase 3 500k SAC results are stored in:

```text
shaheer_phase3_500k_sac_final/sac_phase3_phase4_summary_seed0.csv
shaheer_phase3_500k_sac_final/sac_500k_phase3_baselines.png
shaheer_phase3_reproducibility/checkpoints/
```

Command used:

```bash
python part2/run_part2_experiments.py --algorithm sac --timesteps 500000 --episodes 50
```

Higher return is better, i.e. less negative values are better.

| Configuration | Mean return | Std return | Success rate |
|---|---:|---:|---:|
| source → source | -2.2759 | 2.0831 | 0.52 |
| source → target | -2.0733 | 1.9796 | 0.58 |
| target → target | -0.4746 | 0.2843 | 1.00 |

Interpretation:

- `source → target` is the lower-bound sim-to-sim transfer setting.
- `target → target` is the upper-bound setting because the agent is trained directly on the target dynamics.
- The target-trained policy reaches perfect success rate, showing that the target environment can be solved by SAC when training directly on it.
- The source-trained policy transfers partially, but remains far from the target-trained upper bound in success rate and return.

---

## 7. Phase 4 — Domain Randomization

### Implemented methods

Domain Randomization is implemented in:

```text
part2/rand_wrapper.py
part2/train_sb3.py
```

Two methods are supported:

- **UDR — Uniform Domain Randomization**  
  A new cube mass is sampled at reset time from a fixed uniform range.

- **ADR — Automatic Domain Randomization**  
  The mass interval adapts based on recent performance near the current domain boundaries.

The final SAC runs use:

| Parameter | Value |
|---|---:|
| Algorithm | SAC |
| Timesteps | 500,000 |
| Evaluation episodes | 50 |
| Seed | 0 |
| UDR mass range | [0.5, 8.0] kg |
| Source mass | 1 kg |
| Target mass | 5 kg |

The UDR range is broad and is not a narrow distribution centered only around the target mass.

### Example commands

Train SAC with UDR:

```bash
python part2/train_sb3.py \
  --algorithm sac \
  --env-type source \
  --sampling-strategy udr \
  --timesteps 500000 \
  --mass-range 0.5 8.0 \
  --seed 0 \
  --run-name sac_udr_source_500k_seed0
```

Train SAC with ADR:

```bash
python part2/train_sb3.py \
  --algorithm sac \
  --env-type source \
  --sampling-strategy adr \
  --timesteps 500000 \
  --mass-range 0.5 8.0 \
  --seed 0 \
  --run-name sac_adr_source_500k_seed0
```

The full matrix can also be launched with:

```bash
python part2/run_part2_experiments.py \
  --algorithm sac \
  --timesteps 500000 \
  --episodes 50 \
  --include-udr \
  --include-adr
```

### Official Phase 4 results

Official cleaned Phase 4 results are stored in:

```text
results/part2/summaries/umesh_phase4_final_results.csv
results/part2/summaries/umesh_phase4_notes.txt
results/part2/sac_udr_source_500k_seed0/
results/part2/sac_adr_source_500k_seed0/
```

Higher return is better, i.e. less negative values are better.

| Configuration | Mean return | Std return | Success rate |
|---|---:|---:|---:|
| source → target baseline | -2.7273 | 2.3281 | 0.50 |
| target → target upper bound | -0.4165 | 0.2358 | 1.00 |
| UDR source → source | -0.4911 | 0.5490 | 0.98 |
| UDR source → target | -0.5017 | 0.4201 | 1.00 |
| ADR source → source | -1.8827 | 1.7952 | 0.68 |
| ADR source → target | -1.9204 | 1.8932 | 0.66 |

Main interpretation:

- UDR gives the best transfer performance in this project.
- UDR improves source-to-target success rate from 0.50 to 1.00.
- UDR matches the target-trained upper bound in success rate and achieves a mean return close to the target-trained upper-bound return.
- ADR improves over the source-to-target baseline but is weaker than UDR in this run.
- A likely reason is that ADR did not expand its adaptive mass interval quickly or effectively enough, whereas UDR exposed the policy to a broad fixed range of masses throughout training.

---

## 8. Final paper

The final report is written using the CVPR LaTeX template. A copy of the compiled PDF and the Overleaf/LaTeX sources are included in this repository under `paper/`.

The PDF must also be submitted separately on the course portal.

Report filename:

```text
68_RL_s362643_s359719_s353868_s351863_Clemos_Marello_Abdullah_Bansari.pdf
```

The report contains:

- abstract with GitHub repository link;
- introduction;
- related work;
- methodology;
- experimental results;
- discussion;
- conclusion;
- references.

The paper uses the official result files listed above and does not introduce unsupported numerical claims.

Paper files:

```text
paper/68_RL_s362643_s359719_s353868_s351863_Clemos_Marello_Abdullah_Bansari.pdf
paper/overleaf_source/main.tex
paper/overleaf_source/main.bib
paper/overleaf_source/figures/
paper/overleaf_source/data/
```

---

## 9. Reproducibility notes

### Important result files

| Purpose | Path |
|---|---|
| Hopper official metrics | `samuel_part/reports/comparison_metrics.csv` |
| Hopper hyperparameters | `samuel_part/reports/hyperparameters_used.csv` |
| Hopper plots | `samuel_part/plots/` |
| PPO/SAC 100k comparison | `shaheer_phase3_100k_deliverables/` |
| SAC 500k Phase 3 results | `shaheer_phase3_500k_sac_final/` |
| SAC 500k checkpoints | `shaheer_phase3_reproducibility/checkpoints/` |
| UDR/ADR official final table | `results/part2/summaries/umesh_phase4_final_results.csv` |
| UDR/ADR final models | `results/part2/sac_udr_source_500k_seed0/`, `results/part2/sac_adr_source_500k_seed0/` |

### Evaluation protocol

| Phase | Metric protocol |
|---|---|
| Hopper | Deterministic evaluation during training; aggregate metrics in Samuel's CSV |
| PPO/SAC | 50 deterministic evaluation episodes |
| UDR/ADR | 50 deterministic evaluation episodes |

### Known limitations

- Most experiments use a single main seed due to compute limitations.
- Training RL agents is stochastic, so rerunning the same commands can produce slightly different values.
- The final paper uses cleaned official CSV tables rather than all intermediate smoke-test outputs.
- Some intermediate paths in older summary CSV files may refer to run directories not included in the final compact repository. The official paths are the ones listed in this README.

---

## 10. Authors

| Name | Student ID | Email |
|---|---:|---|
| Samuel Clemos | 362643 | s362643@studenti.polito.it |
| Matteo Marello | 359719 | s359719@studenti.polito.it |
| Shaheer Abdullah | 353868 | s353868@studenti.polito.it |
| Umesh Bansari | 351863 | s351863@studenti.polito.it |

---

## 11. Citation and references used in the report

The report discusses and cites the literature provided in the project assignment, including:

- Sutton and Barto, *Reinforcement Learning: An Introduction*;
- Kober, Bagnell, and Peters, *Reinforcement Learning in Robotics: A Survey*;
- Schulman et al., *Proximal Policy Optimization Algorithms*;
- Haarnoja et al., *Soft Actor-Critic*;
- Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*;
- Peng et al., *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization*;
- Rubik's Cube / Automatic Domain Randomization work by OpenAI et al.

For the complete bibliography, see the final CVPR-style PDF report.
