---
title: "Round 5 — E3c report: SMC-ESO re-optimized with the PSO-FLC budget at 100 Hz"
---

**Run.** MATLAB R2026a, authors' PC, 2 Nov 2026. Script `scripts/r5_E3c.m` (protocol v2.2, written before the runs). Same PSO as E3b (30 × 20, five runs, seeds 20260806–10), same cost as PSO-FLC without its gain regularizer, controller sampled at 100 Hz (`ctrlTs = smcTs = 1e-2`), search vector [c, k, η, φ_bl, ω0, Kp_φ, Kv_φ] (SMC-ESO gains and its own outer loop). Evaluation at 100 Hz in S1 (8 cases, 60 s) and S2 (kicks 1, 2 rad/s). Raw: `results/E3c_final/`, logs `logs/E3c_final/`, analysis `analysis/r5_E3c_analyze.py` → `results/E3c_final_analysis.txt`.

**Results.**

- The retuned sets still violate the specifications (gv 6.4–6.5; the authors' gains 8.5): OS_θ 0.038–0.039 (spec 0.006), ts_θ 5.5–6.4 s (spec 3 s).
- Recovered (10 cases): authors' SMC-ESO 5/10 (1 per case in nominal, mgl −30 %, Ip ±30 %, kick 1); retuned 29/50 (nominal, mgl ±30 %, adverse, kick 1: 5/5 each; Ip +30 % 2/5, Kt −30 % 2/5, Ip −30 %, Kt +30 %, kick 2 0/5); PSO-FLC 10/10; BC+TD3 (π(0) removed) 50/50. No retuned run diverges.
- Nominal IAE_ϕ retuned median 0.86 (0.75–1.04) is below PSO-FLC (3.07), but nominal IAE_θ 0.12 is about five times PSO-FLC (0.023); under Ip +30 %, Kt ±30 % and the 2 rad/s kick, IAE_θ is 1.1–3.4 versus 0.022–0.030 (PSO-FLC) except the 2 rad/s kick 2.2 vs 0.30.

**Reading.** Giving SMC-ESO the same optimization effort improves it in some cases (adverse, mgl +30 % no longer fail; recovery 5/10 → 29/50) but not to the level of PSO-FLC or the agents in robustness or IAE_θ. The order of the controllers is therefore not explained by tuning effort at this budget, with the exception that SMC-ESO has the lowest nominal IAE_ϕ.

**Limits.** One PSO budget (30 × 20) and one cost designed for PSO-FLC; SMC-ESO structure and bounds as listed; deterministic, one plant, one initial condition; LQR baseline (E4) not re-run with the PSO budget (it used 600 Latin-hypercube samples per outer-loop gain set).

**Where used.** Manuscript Section 3.10 (Sampling-rate paragraph), Limitations; Response R2-3.
