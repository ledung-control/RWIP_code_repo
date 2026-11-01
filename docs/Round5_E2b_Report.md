---
title: "Round 5 — E2b report: origin penalty in the behavior-cloning loss"
---

**Protocol:** v1.8 (written before the run; λ = 100 fixed, not tuned). **Run:** MATLAB R2026a, 1 Nov 2026, seeds 1–5, BC-only and BC+TD3. Script `scripts/Train_RL_v13_bc_E2b.m` (authors' file + penalty 100·π(0)² in the BC loss), driver `r5_E2b.m`. Raw `results/E2b_final/*.csv`; analysis `analysis/r5_E2b_analyze.py` → `results/E2b_final/E2b_analysis.txt`.

**Results.**

- BC-only: |π(0)| ≤ 7×10⁻⁴; IAE_ϕ (60 s) median 4.50 (4.19–7.96) vs 4.22 for stored networks with π(0) subtracted (lower in 2 of 5 seeds, two-sided exact sign-flip p ≈ 0.81); ϕ(600 s) between −0.012 and 0.005 rad.
- BC+TD3: π(0) = +0.079, +0.067, −0.064, −0.001, +0.033 (seeds 1–5); IAE_ϕ (60 s) 57.3, 48.2, 45.0, 3.88, 36.4 (median 45.0); at 600 s 637.8, 527.5, 445.6, 13.0, 390.1; |ϕ(600 s)| up to 1.07 rad. The seed with the smallest |π(0)| (seed 4) has the smallest error.
- ρ ≤ 0.28 in all ten networks.

**Reading.** A penalty applied in the BC stage keeps BC-only near zero at the origin but is not carried through the unchanged TD3 stage, which regenerates a non-zero π(0) in 4 of 5 seeds and the flywheel offset with it. Consistent with Section 3.7/4.3; n = 5, observational. E2a (π(0) = 0 by construction) removes the offset in both stages.

**Limits.** One λ, penalty at one point (origin) and in the BC stage only; TD3 actor loss not modified (would be a new implementation); stored comparators are separate runs matched by seed label; E2c (gated residual) not run.
