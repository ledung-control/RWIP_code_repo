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
- `round5_scripts/` revision experiments E0–E5, E7/E8 and the E2a/E2b training variants (E2a, E2b, E2c).
- `results/` result CSVs and analysis text; `docs/` protocol and reports; `python_posthoc/` independent Python re-implementation and analysis scripts.
- `checkpoints/` trained actors (MATLAB `.mat`, about 1.3 MB each): `stored_paper/` (BC-only and BC+TD3, seeds 1-5, the networks of Tables 4-11), `E1_retuned/` (retuned fine-tuning), `E2a_bias_free/` (actor without bias, pi(0)=0 by construction), `E2b_origin_penalty/` (penalty 100*pi(0)^2 in the BC loss), `E2c_gated/` (actor output multiplied by tanh(||o4||/0.05), pi(0)=0 by the gate). File names: `agent_bconly_seed<k>.mat` = BC-only, `agent_bc_seed<k>.mat` = BC+TD3.


## Supplementary report
`docs/Supplementary_Report_Tables_S1-S11.docx` holds Tables S1-S11 (clipping frequency, parameter change, residual near the origin, paired statistics, ...). Tables S12-S13 (ESO-bandwidth sweep, oracle gain) are `results/E7E8_MATLAB_2026a.csv`. Round-5 protocol and reports: `docs/Round5_Protocol_E1-E8.md`, `docs/Round5_E2a_Report.md`, `Round5_E2b_Report.md`, `Round5_E2c_Report.md`.

## Known gaps
- The original script of the bias-injection experiment was not kept; the injection results were regenerated with `python_posthoc/phlib.py` under the stated clipping order.
- Python analysis scripts read `python_posthoc/results/runs_600s.csv` (relative paths may need editing).
- Training and evaluation need MATLAB R2026a with the toolboxes listed above; no GPU is required.
