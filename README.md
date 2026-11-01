# Implementation code — Physics-Structured PSO-Fuzzy Control and Residual TD3 for a RWIP


## Contents (to be copied from the project folder)
- Plant, PSO-FLC, SMC-ESO: `sim_rwip_v13.m`, `sim_rwip_r5.m`, PSO script and result, official FIS.
- Training: `Train_RL_v13_bc.m`, `Train_RL_v13_repro.m`, `rwip_env_v13.m`, `Train_RL_v13_bc_E2a.m` (bias-free actor).
- Evaluation: `Compare_v13_robust.m` and the Round 5 scripts `r5_E1.m`, `r5_E2a.m`, `r5_E3.m`, `r5_E4.m`, `r5_E5.m`.
- Checkpoints: 5 BC-only, 5 BC+TD3, 5+5 bias-free (E2a, `agents/E2a_final`), retuned E1 agents.
- Python post-hoc package: `rwipsim.py`, `actorx.py`, `phlib.py` and scripts (Round4_PostHoc_Package).
- Results CSVs (`results/`), logs, protocol `Round5_Protocol_E1-E8`.

## Environment
MATLAB R2026a (Simulink, Fuzzy Logic, Reinforcement Learning, Deep Learning toolboxes); Python 3 with NumPy, SciPy.

## Random seeds
Training seeds 1–5 (E2a uses 101-based smoke seed only for the smoke test). Deterministic simulation; E5 draws use fixed RandStream seeds in `r5_E5.m`.

## Reproducing
Tables 4–11 and Figures 5–7: run `Compare_v13_robust.m` on the stored agents, then the Python commands in the post-hoc package. Section 3.10: the matching `r5_*.m` script. Full retraining takes about 1–2.5 h per seed per stage on one PC.

## Folder layout
- `matlab_core/` the authors' plant, PSO, FLC/SMC-ESO, training and evaluation code (unchanged).
- `round5_scripts/` revision experiments E0–E5, E7/E8 and the E2a/E2b training variants.
- `results/` result CSVs and analysis text; `docs/` protocol and reports; `python_posthoc/` independent Python re-implementation and analysis scripts.
- Network checkpoints (BC-only, BC+TD3, E1/E2a/E2b agents; about 300 MB) are **not** in this folder yet; add them with Git LFS or a release asset before publishing.
