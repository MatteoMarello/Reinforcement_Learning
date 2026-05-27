# Samuel - Phase 1/2 Hopper Deliverables

## Scope covered

- REINFORCE without baseline.
- REINFORCE with a scalar baseline.
- One-step Actor-Critic.
- Short 50-episode runs, 1000-episode runs, and long 15000-episode runs.

## Delivered artifacts

- Logs and CSV files: available under `samuel_part/results/<run>/training_log.csv`.
- Learning curves per run: available inside each result folder as `learning_curve.png`.
- Global comparison plots: available under `samuel_part/plots/`.
- Hyperparameters: available in `samuel_part/reports/hyperparameters_used.csv`.
- Comparison metrics and notes: available in `samuel_part/reports/comparison_metrics.csv` and this report.

## Final long-run presentation

The long Hopper results are presented as a single complete run in each long `training_log.csv` before generating the final plots.

The long comparison plot was also rebuilt so that:

- evaluation points are ordered by episode,
- the training trend is shown as a smoothed curve,
- evaluation points are displayed as markers instead of a misleading connected path.

## Main quantitative comparison

| Run | Algorithm | Episodes | Baseline | Best Eval | Final Eval | Final Train | Mean Last 20 Train |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| quick_reinforce_no_bl | reinforce | 50 | - | 363.282 | 363.282 | 29.870 | 27.960 |
| quick_reinforce_const_bl | reinforce_baseline | 50 | 1000.0 | 97.556 | 71.370 | 29.216 | 19.399 |
| quick_actor_critic | actor_critic | 50 | - | 27.066 | 8.565 | 9.005 | 9.704 |
| phase2_reinforce_bl0_1000 | reinforce | 1000 | - | 346.675 | 243.164 | 168.911 | 209.702 |
| phase2_reinforce_bl20_1000 | reinforce_baseline | 1000 | 20.0 | 223.212 | 219.723 | 199.341 | 220.171 |
| phase2_reinforce_bl1000_1000 | reinforce_baseline | 1000 | 1000.0 | 181.041 | 97.763 | 21.236 | 58.459 |
| phase2_actor_critic_1000 | actor_critic | 1000 | - | 466.191 | 466.191 | 401.295 | 369.647 |
| final_reinforce_bl20_15000 | reinforce_baseline | 15000 | 20.0 | 444.293 | 321.348 | 325.845 | 321.317 |
| final_actor_critic_15000 | actor_critic | 15000 | - | 1010.380 | 168.042 | 167.265 | 168.543 |

## Comparison notes

- The baseline value matters. A moderate baseline (`20`) is clearly better than an overly large baseline (`1000`) in the 1000-episode comparison.
- Actor-Critic achieves a very strong peak evaluation in the long run, but its final deterministic evaluation is much lower than the final REINFORCE + baseline=20 run.
- The long plot reflects a single complete run per file, which makes the comparison visually interpretable.

## Final run with the best setup

Selected run: `final_reinforce_bl20_15000`

Reason for selection:

- Algorithm: `reinforce_baseline`
- Baseline: `20`
- Final deterministic eval return: `321.348`
- Best eval return reached: `444.293`
- Final training return: `325.845`
- Mean of the last 20 training returns: `321.317`

This is the best final setup for the delivery because it is the strongest stable final long run and the one included in the final package.

## Reproducibility notes

- Hyperparameters and seeds are stored per run in `config.json` and aggregated in `hyperparameters_used.csv`.
- Environment metadata is stored per run in `env_info.json`.
- The raw outputs for the delivery are copied into `samuel_part/` so the package is self-contained.
