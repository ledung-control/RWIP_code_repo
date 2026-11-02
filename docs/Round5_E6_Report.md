---
title: "Round 5 — E6 report: end-to-end TD3 baseline (R2-5)"
---

**Run.** MATLAB R2026a, authors' PC, 2 Nov 2026. Scripts `scripts/r5_E6_env.m`, `scripts/r5_E6.m` (protocol v2.1, written before the runs). Plant, integrator (RK4 1 ms), 100 Hz agent, observation scaling and reward are the authors' (plant check: PSO-FLC through the copied plant gives IAE_θ 0.0244629, IAE_ϕ 3.21104 = stored values). Actor outputs V = 24a from (θ, θ̇, ϕ, ϕ̇); no behaviour cloning, FLC, SMC-ESO or cascade. TD3: 256-256, actor LR 1e-4, critic LR 1e-3, γ 0.99, batch 256, exploration σ 0.2, seeds 1–3, 800 episodes × 2000 steps (1.6 M steps; the BC+TD3 pipeline uses about 0.4 M). Training 122–139 min per seed. Evaluation: S1 (8 plant cases, 60 s) and S2 (kicks 1, 2 rad/s), 100 Hz, deterministic actor, recovery criterion of E3. Raw: `results/E6_eval/E6_eval_MATLAB_2026a.csv`, logs `logs/E6_train`, `logs/E6_eval`, analysis `analysis/r5_E6_analyze.py` → `results/E6_final_analysis.txt`, agents `agents/E6/`.

**Results.**

- Recovered (10 cases): E2E TD3 0/10 in each of the three seeds; PSO-FLC 10/10; BC+TD3 (π(0) removed) 50/50.
- Seed 3: the pendulum is lost (|θ| > 1.2 rad) in all 10 cases.
- Seeds 1 and 2: |θ| ≈ 0 at the end but the flywheel keeps spinning, ϕ not bounded to 0.05 rad. IAE_ϕ seed 1: 340–403; seed 2: 430–608; PSO-FLC median 3.15; BC+TD3 median 5.58. IAE_θ seed 1: 0.14–0.34; seed 2: 0.057–0.25; PSO-FLC 0.0246; BC+TD3 0.0295.

**Reading.** In this configuration (one reward, one TD3 setting, 1.6 M steps, not tuned after evaluation) end-to-end TD3 did not stabilize the full state; the architectures of the paper (fuzzy base + BC residual) were needed to obtain recovery. This is not a general statement about end-to-end learning or about published methods.

**Limits.** One configuration, 3 seeds, no hyper-parameter search, one reward; own implementation, not a published method's code; deterministic evaluation, one plant, one initial condition.

**Where used.** Manuscript Section 3.10 (new paragraph "End-to-end TD3 baseline"), Limitations; Response R2-5.
