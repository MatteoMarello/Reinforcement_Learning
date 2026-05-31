# Final Submission Checklist — Group 68 RL Project

This checklist summarizes the final state of the repository against the project requirements.

## Mandatory deliverables

- [x] Public GitHub repository prepared: `https://github.com/MatteoMarello/Reinforcement_Learning`
- [x] Code implementation for Part 1/2: Hopper, REINFORCE, REINFORCE + constant baseline, Actor-Critic.
- [x] Code implementation for Part 3: PPO/SAC pipeline with Stable-Baselines3 and source/target evaluation.
- [x] Code implementation for Part 4: UDR and ADR on cube mass.
- [x] Final PDF paper included in `paper/` and ready for portal submission.
- [x] GitHub repository link included at the end of the paper abstract.
- [x] Final README with setup, commands, official result files, and final tables.

## Project steps from the assignment

| Step | Requirement | Repository evidence |
|---|---|---|
| 1 | RL preliminaries and sim-to-real/domain-randomization context | Paper Introduction/Related Work; README overview |
| 2 | From-scratch REINFORCE and Actor-Critic on Hopper | `part1/agent.py`, `part1/train.py`, `samuel_part/` |
| 3 | PPO/SAC baselines and lower/upper bound evaluations | `part2/train_sb3.py`, `part2/eval_sb3.py`, `shaheer_phase3_*` |
| 4 | UDR/ADR domain randomization on cube mass | `part2/rand_wrapper.py`, `results/part2/sac_udr_*`, `results/part2/sac_adr_*` |

## Official result files

- Hopper Phase 1/2: `samuel_part/reports/comparison_metrics.csv`
- PPO/SAC Phase 3: `shaheer_phase3_500k_sac_final/sac_phase3_phase4_summary_seed0.csv`
- UDR/ADR Phase 4: `results/part2/summaries/umesh_phase4_final_results.csv`

## Sanity checks

The Python syntax check was run on the final repository:

```bash
python -m py_compile part1/*.py part2/*.py
```

The paper PDF has 5 pages including references and is formatted with the CVPR-style LaTeX template.

## Before final submission

1. Push this final repository state to GitHub.
2. Do not modify the repository after the project submission deadline.
3. Submit the PDF named:

```text
68_RL_s362643_s359719_s353868_s351863_Clemos_Marello_Abdullah_Bansari.pdf
```

4. Confirm that the public GitHub link in the report abstract points to the final repository.
