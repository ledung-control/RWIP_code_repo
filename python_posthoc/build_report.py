import pandas as pd, json, numpy as np
R='results/'
S=json.load(open(R+'summary.json'))
def md(df,fmt=None):
    df=df.copy()
    return df.to_markdown(index=False,floatfmt=fmt or '.4g')
o=[]
w=o.append
w('''# Round 2 post-hoc analysis report

Manuscript: *Physics-Structured PSO-Fuzzy Control and Residual TD3 for a RWIP: Equilibrium Bias and Performance Limits* (IJIES Paper ID 20266940).
Status: **DRAFT prepared for the authors to verify.** Nothing here is a retraining experiment. Every number below is computed from the stored checkpoints by a Python re-implementation of the plant and loop, and nothing was re-run in MATLAB/Simulink.

## 0. Scope, inputs, validity

| Item | Content |
|---|---|
| Inputs (read only) | `rl_v13_repro/agent_bc_seed{1..5}.mat` (BC+TD3), `rl_v13_repro/agent_bconly_seed{1..5}.mat` (BC-only), `fis_rwip_v13_best.fis`, plant constants and gains as in `sim_rwip_v13.m` |
| Simulator | `rwipsim.py` + `actorx.py` (your audit scripts, unchanged); `phlib.py` (new, float re-implementation of the same loop that records clipping events and actor outputs) |
| Validation | `phlib.py` is bit-identical to `rwipsim.py` on the checked runs and reproduces the MATLAB Table 9 values and the 1.0 and 2.0 rad/s impulse rows printed in the manuscript (`t_val.py`, `t_dist.py`; logs `run.log`, `dist.log`) |
| Plant / solver | RK4, 1 ms; agent period 10 ms (ZOH); 201x201 LUT with bilinear interpolation; nominal IC theta0 = 0.10 rad, other states 0 |
| Not covered | The MATLAB/Simulink 600 s data were **not** re-opened for a cross-check; no SMC-ESO port exists in Python, so no SMC-ESO number in this report is new |
| Random numbers | neighbourhood sampling: `np.random.default_rng(20261025)`, 4000 observations per box; bootstrap: seed 1; all other analyses are deterministic |

Regeneration: `python3 posthoc_round2.py` (about 4 min), `python3 posthoc_disturbance_clip.py`, `python3 steady_state_check.py` (see `README.md`).

## Table S1. Clipping frequency (R1-5)

Order of operations, as implemented in `sim_rwip_v13.m`, `rl_policy_from_agent.m` and `rwipsim.py`: `a = pi(o) - aBias` -> clip to [-1, 1] -> `u_corr = 10 V * a` (held 10 ms) -> `V = clip(V_FLC + u_corr, -24, 24)` every 1 ms.
Script `posthoc_round2.py` -> `results/clipping_60s.csv`; `posthoc_disturbance_clip.py` -> `results/disturbance_clipping_and_metrics.csv`. Samples: 10 policies x {raw, unbias} x 60 s nominal (6000 agent steps, 60 000 ms steps each); 60 evaluated runs in the disturbance file (ten policies x raw/corrected x three scenarios).
''')
c=pd.read_csv(R+'clipping_60s.csv')
w(md(c[['policy','seed','variant','residual_clip_steps','agent_steps','total_clip_ms','ms_steps','max_abs_residual_pre_clip']],'.4g'))
d=pd.read_csv(R+'disturbance_clipping_and_metrics.csv')
g=d.groupby(['policy','variant','impulse_rad_s']).agg(res_clip_max=('residual_clip_steps','max'),total_clip_ms_max=('total_clip_ms','max'),max_pre_clip=('max_abs_residual_pre_clip','max')).reset_index()
w('\nImpulse scenarios (impulse on theta-dot at t = 5 s, 60 s), maximum over the 5 seeds:\n')
w(md(g))
w('''
**Result.** The residual clip at |a| = 1 was never active (largest pre-clip |a| = 0.864, BC-only, 2.0 rad/s). The total +-24 V clip was never active in the nominal runs and was active in at most 8 of 60 000 steps in any impulse run; PSO-FLC alone never reaches it. The bias-subtraction therefore does not rely on the clip.

## Table S2. Measured parameter change BC-only -> BC+TD3 (R1-6)

Script `posthoc_round2.py` -> `results/param_displacement.csv`. For each seed k, theta_BC = weights of `agent_bconly_seed{k}`, theta_TD3 = weights of `agent_bc_seed{k}`; L2 = ||theta_TD3 - theta_BC||_2; relative = L2 / ||theta_BC||_2; Linf = max |difference|; per tensor (W1,b1,W2,b2,W3,b3) and in total.
''')
p=pd.read_csv(R+'param_displacement.csv')
tot=pd.DataFrame([{'seed':k[1:],'L2':v['L2'],'relative':v['rel'],'Linf':v['Linf']} for k,v in S['param_change_ALL'].items()])
w(md(tot))
w('\nPer tensor:\n'); w(md(p))
w('''
**Result.** Total L2 change 3.83-4.06 (17.7-19.3 % of the BC-only norm), largest single parameter change 0.018-0.024. The earlier figure 0.36 (learning rate x episodes x updates) is a nominal budget, not a measurement, and was removed. The change is not small; the manuscript no longer says the fine-tuned actors stay close to the BC initialization.
Actor-update schedule (from `Train_RL_v13_bc.m` / `Train_RL_v13_repro.m`): 4000 agent steps per 40 s episode, policy-update frequency 2 -> 2000 actor updates per episode, 18 fine-tuning episodes after 20 warm-up episodes with actor learning rate 0.

## Table S3. Variation of pi(s) - pi(0) away from the origin (R1-10)

Script `posthoc_round2.py` -> `pi_minus_pi0_neighbourhood.csv`, `pi_jacobian_at_origin.csv`, `pi_minus_pi0_trajectory.csv`. Observations (normalised, 6-D) drawn uniformly in [-eps, eps]^6, eps in {0.01, 0.05, 0.1, 0.2, 0.5, 1}, 4000 samples each, rng 20261025; Jacobian by central finite differences at the origin; trajectory windows from the raw nominal 60 s runs. Statistic: std of pi(s) - pi(0).
''')
n=pd.read_csv(R+'pi_minus_pi0_neighbourhood.csv'); n=n[n.policy=='td3']
pv=n.pivot(index='seed',columns='box_halfwidth_normalized_obs',values='std_dev')
w('Std of pi(s) - pi(0), BC+TD3, by box half-width:\n'); w(pv.reset_index().to_markdown(index=False,floatfmt='.4f'))
j=pd.read_csv(R+'pi_jacobian_at_origin.csv'); j=j[j.policy=='td3']
w('\nJacobian at the origin (BC+TD3):\n'); w(md(j[['seed','pi0','d/dtheta','d/dthetadot','d/dphi','d/dphidot','norm']]))
t=pd.read_csv(R+'pi_minus_pi0_trajectory.csv'); t=t[t.policy=='td3']
w('\nAlong raw nominal trajectories (BC+TD3), deviation pi(s_t) - pi(0):\n'); w(md(t))
w('''
**Result.** pi(s) - pi(0) is state dependent: std 0.010-0.014 for half-width 0.01 and 0.084-0.110 for 0.1; Jacobian norm 1.7-2.4. Along trajectories it moves by +-0.01-0.05 in the first 5 s and then sits at a constant. At the raw steady state the residual action a_ss = -0.226, -0.013, +0.472, -0.151, -0.294 differs from pi(0) = -0.120, -0.004, +0.277, -0.065, -0.171, so the offset is set by a_ss, and pi(0) is the part that the diagnostic measures. The fact that phi -> 0 after subtracting pi(0) does not by itself prove that the deviation is constant; the manuscript says so.

## Table S4. Matched controls (R2-2): status

| Control requested | Status | Evidence / reason |
|---|---|---|
| BC-only + same bias subtraction | **Done (post-hoc, stored networks, not a retraining)** | `runs_600s.csv`, `paired_changes.csv`; Section 3.7 of the manuscript. BC-only unbias nominal IAE_phi 4.83, 3.99, 4.22, 4.05, 11.88 (median 4.22) |
| Actor constrained to 0 at the origin | **Not done** | needs retraining (MATLAB + Reinforcement Learning Toolbox + original training environment) |
| Origin-penalised training | **Not done** | same |
| Equilibrium-preserving residual | **Not done** | same |

## Table S5. Statistics and paired changes (R2-4, R1-8)

Script `posthoc_round2.py` -> `paired_changes.csv`, `summary.json`. Exact two-sided sign-flip test over the 2^5 = 32 sign patterns of the paired differences (minimum attainable p at n = 5 is 0.0625); exact permutation test of Pearson r over 5! = 120 permutations; Fisher z CI with n = 5; bootstrap 2.5-97.5 % (seed 1).
''')
rows=[]
for k,v in S['paired_signflip'].items():
    rows.append({'contrast:metric':k,'median diff':v['median'],'n negative / n':'%d/%d'%(v['n_negative'],v['n']),'exact sign-flip p':v['exact_signflip_two_sided_p']})
w(md(pd.DataFrame(rows)))
sc=S['stats_corr']
w('''
Correlation of |aBias| with raw nominal IAE_phi across the 5 BC+TD3 seeds: Pearson r = %.4f, exact permutation p = 1/120 = %.4f, Fisher 95%% CI [%.3f, %.4f], bootstrap [%.3f, %.4f], leave-one-out r = %.4f-%.4f, Spearman = 1; signed-aBias r = %.3f. With n = 5 this is an observational association, not a causal test, and the CI is wide.
''' % (sc['pearson_abs'],sc['perm_p'],*sc['fisher_ci95'],*sc['bootstrap_pct_2p5_97p5'],min(sc['loo']),max(sc['loo']),sc['signed_pearson']))
w('''
Horizon ratio, IAE_phi(600 s) / IAE_phi(60 s), per seed (`horizon_ratio_per_seed.csv`):
''')
h=pd.read_csv(R+'horizon_ratio_per_seed.csv')
w(md(h))
w('''
Reading: the raw BC+TD3 medians are 111.57 at 60 s and 1144.59 at 600 s (ratio 10.259, not exactly 10); per-seed ratios 7.01-10.70; the intercepts of the late linear fit (T = 200-600 s) are -3.2, +3.0, -23.1, -0.9, -6.0, i.e. negative for 4 of 5 seeds, which is why the ratio is below the horizon ratio of 10. The linear growth is asymptotic with a transient offset, not an exact proportionality. **Exception:** BC-only seed 3 raw does not settle (sustained oscillation, |theta| <= 1.5e-4, phi in [-0.483, -0.466], spectral radius 1.001); the manuscript text was corrected from "all policies" to 14 of 15.

## Table S6. Steady-state theory versus simulation (R1-4, R1-5)

Constant correction bias b (action units), theta = theta-dot = phi-dot = 0 requires zero total voltage: `Ku * FLC(-Ne * thetaref_ss, 0) + 10 b = 0`, `phi_ss = -thetaref_ss / Kp_phi`. Solutions exist iff `|10 b| < Ku * FLC(0.0679, 0) (0.0679 = Ne*0.035)` = **5.8886 V**, i.e. `|b| < 0.5889`. The number 0.035/Kp_phi = 7.994 rad is the value of phi at that limit, not a bound on phi.
Script `steady_state_check.py` -> `results/steady_state_check.csv`; injection sweep `bias_injection_sweep.csv`, `bias_steady_state_theory.csv` (11 values of b, constant action replacing the actor).
''')
w(md(pd.read_csv(R+'steady_state_check.csv')[['policy','seed','variant','a_ss_residual_action','phi_end_600s','phi_ss_predicted','abs_error']]))
w('''
**Result.** Closed form vs simulated phi(600 s): |error| <= 2.2e-13 for 19 of 20 policy variants; the exception is BC-only seed 3 raw (error 0.034, non-settling, see above). Injection: phi_ss = -/+2.7317 rad at b = +/-0.3; log-log slope 0.976; Pearson r(|b|, IAE_phi(60 s)) = %.4f over 11 injected values; injection ratio IAE_phi(600)/IAE_phi(60) = %.2f-%.2f (the earlier manuscript quoted 9.14 as the lower end; 9.14 is the b = +0.02 value here, but b = -0.02 gives 8.12, so the range was widened; the original `bias_injection_experiment.py` is not available, so the authors should confirm).
Local stability of the 10 ms closed-loop map (`local_closed_loop.csv`): all 10 corrected policies have the origin as a fixed point (phi ~ 1e-14) with spectral radius 0.9965-0.9977; raw fixed points have radius 0.9969-0.9979 (BC-only seed 3 raw: 1.001). Small-amplitude unbias runs (`unbias_small_amplitude.csv`) converge to 0.

## Items not reproducible from the current inputs

`bias_injection_experiment.py`, `fis_verify.py`, `rwip_full_sim.py`, `make_fig_*.py`, `fis_surface_201.npy`, `citation_mapping.json` (named in earlier notes but absent); the Python PSO script/log that produced the earlier PSO particle; the training-run log; MATLAB and toolbox versions; an SMC-ESO port and any LQR/state-feedback code.
''' % (S['injection_pearson_r_absb_vs_IAEphi60_11pts'],*S['injection_ratio_range']))
open('/home/claude/out2/Round2_PostHoc_Report.md','w').write('\n'.join(o))
