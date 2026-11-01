---
title: "Round 5 — E3b report: PSO-FLC retuned under 100 Hz sampling"
---

**Run.** MATLAB R2026a, authors' PC, 1–2 Nov 2026. Script `scripts/r5_E3b.m` (protocol v2.0, written before the runs). Five PSO runs (seeds 20260806–10, 30 particles × 20 iterations, same bounds/weights/initialisation as `pso_rwip_v13.m`), controller sampled at 100 Hz (`ctrlTs = 1e-2`), cost evaluated by the authors' evaluator. Then all variants evaluated at 100 Hz in S1 (8 plant cases, 60 s) and S2 (θ̇ kicks 1 and 2 rad/s). Raw: `results/E3b_final/`, logs `logs/E3b_final/`, analysis `analysis/r5_E3b_analyze.py` → `results/E3b_final_analysis.txt`.

**Results.**

- Retuned parameters: J 4.31–4.97 (evaluator scale; the stored J = 0.3148 is on another scale, the metrics agree: OS, ts_θ, ϕ_pk, ts_ϕ within 1 % of stored at 1 kHz). ts_ϕ 10.61–10.90 s.
- IAE_ϕ retuned (median) is 2–8 % **higher** than the original parameters at 100 Hz in all 10 cases (nominal 3.21 vs 3.07). Retuning did not improve on the original design.
- All 50 retuned evaluations recover (original 10/10; unbiased BC+TD3 50/50; raw BC+TD3 nominal 0/5).
- IAE_ϕ vs median of 5 unbiased BC+TD3 agents: retuned lower in 9 of 10 cases (45 of 50 run–case pairs); exception: 2 rad/s kick, 88.2 vs 75.7 (agents better).
- IAE_θ: retuned below the agent median in 8 of 10 cases (not Ip +30 %, kick 2.0).

**Limits.** SMC-ESO not retuned; same PSO budget as the paper, no equal budget for other baselines; deterministic, one plant, one initial condition; PSO runs are one seed each (n = 5 runs).

**Where used.** Manuscript Section 3.10 (Sampling rate paragraph); Response R2-3.
