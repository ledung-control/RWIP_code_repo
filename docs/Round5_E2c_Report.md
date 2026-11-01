---
title: "Round 5 — E2c gated equilibrium-preserving residual: report"
---

**Protocol:** `Round5_Protocol_E1-E8` v1.9 (written before any run). **Code:** `scripts/Train_RL_v13_bc_E2c.m` (copy of the authors' script; only the actor definition changes), `r5_E2c_train.m`, `r5_E2c.m`; analysis `analysis/r5_E2c_analyze.py`; results `results/E2c_final/*.csv`, `results/E2c_final_analysis.txt`. MATLAB R2026a, authors' PC, 1 Nov 2026. Smoke test (seed 101, tiny schedule, not a result): π(0) = −9.6×10⁻¹⁷ (BC-only) and 1.7×10⁻¹⁵ (BC+TD3).

**Definition.** a(o) = g(‖o₄‖)·π(o), g(x) = tanh(x/0.05) with the 1×10⁻⁶ smoothing offset removed so that g(0) = 0; o₄ = (θ, θ̇, φ, φ̇) normalized; biases kept; paper configuration (γ 0.995, S_PH 10, 60 demonstration episodes, 3000 BC iterations, 20 warm-up and 18 fine-tuning episodes, SkipBCGate = true as in E2a); seeds 1–5; no seed excluded.

**Results (nominal IC, 60 s unless stated).**

| Seed | BC-only IAE_φ | BC-only φ(600 s) | BC+TD3 IAE_φ | BC+TD3 φ(600 s) |
|---|---|---|---|---|
| 1 | 27.68 | 7e-15 | 203.1 | −3.88 |
| 2 | 2.70 | 1e-14 | 3.07 | 2e-14 |
| 3 | 55.31 | 0.937 | 48.51 | 0.81 |
| 4 | 39.22 | 1e-14 | 2.94 | 1e-14 |
| 5 | 3.61 | 1e-14 | 127.2 | 2.2 |

Medians (range) of IAE_φ at 60 s: BC-only 27.7 (2.70–55.3), BC+TD3 48.5 (2.94–203); at 600 s: 27.7 and 485.7. IAE_θ median: 0.0286 and 0.0308. References: PSO-FLC 3.211; stored π(0)-removed networks 4.22 (BC-only) and 5.55 (BC+TD3). All ten |π(0)| < 2×10⁻¹⁴. Six of ten networks end at φ ≈ 0.

**Paired against the stored π(0)-removed networks (by seed label, separate training runs).** Gated lower in IAE_φ in 2 of 5 seeds in each group; exact two-sided sign-flip p = 0.91 (BC-only and BC+TD3, 60 and 600 s); IAE_θ lower in 3/5 (BC-only) and 2/5 (BC+TD3), p = 0.50–0.81.

**Reading.** One gate design, n = 5. Making π(0) = 0 by a gate did not give the regulation of the bias-free actor (E2a: all ten networks φ(600 s) < 1.1×10⁻⁶ rad) or of the π(0)-removed stored networks: four networks settle at a non-zero flywheel angle, which means that away from the origin the gated network still produces an action that holds φ ≠ 0. The result qualifies the statement that π(0) = 0 "by construction" suffices; it does not show why E2a and E2c differ (possible reasons, not tested: a ReLU network without bias is positively homogeneous, a(λo) = λa(o) for λ > 0 before the tanh, whereas the gate is saturated for ‖o₄‖ > ~0.15). No tuning of the gate width was done.
