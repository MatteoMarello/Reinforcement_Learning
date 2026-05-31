# FAIML RL Project — CVPR-style Paper Package (v2 — improved)

This folder is the Overleaf-ready source for the project report.

## Main files
- `main.tex`: paper source (improved version).
- `preamble.tex`: extra packages and macros.
- `cvpr.sty`: local CVPR-style file (self-contained, no installation needed).
- `figures/`: all 6 figures used in the paper.
- `main.pdf`: compiled preview (5 pages + references).
- `68_RL_*.pdf`: submission-name copy.

## How to upload to Overleaf
1. Go to Overleaf → New Project → Upload Project.
2. Upload this ZIP file.
3. Set `main.tex` as the main document if Overleaf does not detect it automatically.
4. Compile with pdfLaTeX (two passes recommended for cross-references).

## Official result sources
- Hopper Phase 1/2: `data/comparison_metrics.csv`
- PandaPush Phase 3: `data/sac_phase3_phase4_summary_seed0.csv`
- PandaPush Phase 4: `data/umesh_phase4_final_results.csv`

## Improvements over v1
- All 6 available figures are now used (previously only 2 were referenced).
- Missing citations fixed: hofer2021perspectives, todorov2012mujoco, gallouedec2021pandagym.
- PPO vs SAC 100k comparison has its own table and figure.
- Lower bound / upper bound concept fully explained.
- Source and target environments defined explicitly.
- ADR curriculum mechanism described in detail.
- Phase 3 vs Phase 4 baseline discrepancy noted and explained.
- Discussion section expanded with analysis of UDR vs ADR failure mode.
- Limitations section expanded (single seed, sim-to-sim only, narrow DR scope).
- Phase 4 uses side-by-side success-rate + mean-return subfigures.
