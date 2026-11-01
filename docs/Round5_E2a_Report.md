---
title: "Round 5 — E2a report: bias-free actor (π(0) = 0 by construction)"
---

**Run.** MATLAB R2026a, authors' PC, 31 Oct – 1 Nov 2026. Script `scripts/Train_RL_v13_bc_E2a.m` = authors' `Train_RL_v13_bc.m` with three edits: bias-free fully connected layers fc1–fc3, bias gradient zeroed in the BC loss, function name. Configuration of Table 3 (γ 0.995, S_PH 10), seeds 1–5, BC-only and BC+TD3; evaluation S0 at 60 s and 600 s. Raw: `results/E2a_final/*.csv`, logs `logs/E2a_final/*.txt`, analysis `analysis/r5_E2a_analyze.py` → `results/E2a_final/E2a_analysis.txt`.

**Results.**

- π(0) = 0 exactly in all 10 networks; ϕ(600 s) ≤ 1.1×10⁻⁶ rad.
- IAE_ϕ (60 s) median (range): BC-only 3.86 (3.32–4.66); BC+TD3 3.68 (2.67–4.40); PSO-FLC 3.211.
- Stored networks with π(0) subtracted: BC-only 4.22; BC+TD3 5.55; raw: 12.94 / 111.6.
- E2a below stored-unbias in IAE_ϕ(60 s): 5/5 seeds in both groups (exact two-sided sign-flip p = 0.0625, minimum attainable at n = 5; the one-sided value 0.0312 is not used). At 600 s: BC+TD3 5/5; BC-only 4/5 (p = 1.0 two-sided).
- IAE_θ: no distinguishable difference (BC-only 3/5 lower; BC+TD3 2/5 lower).
- BC+TD3 < BC-only in IAE_ϕ in 3 of 5 seeds: no advantage of fine-tuning shown.
- ρ: BC-only ≤ 0.27; BC+TD3 ≤ 0.20 except seed 5 = 0.75 (> 0.5 criterion).

**Limits.** Comparators are separate training runs matched by seed label only; n = 5; noise group (E5) not run on E2a networks; one configuration. Origin-penalized training and the simple equilibrium-preserving residual (E2b, E2c) not run.

**Where used.** Manuscript Section 3.10 (new paragraph), matched-control paragraph of Section 3.7, Limitations; Response R2-2.
