import pandas as pd, json, numpy as np, os, sys
ROOT=sys.argv[1] if len(sys.argv)>1 else '/mnt/user-data/uploads/WRIP_2'
R='results/'
S=json.load(open(R+'summary.json'))
def md(df,fmt=None):
    df=df.copy()
    return df.to_markdown(index=False,floatfmt=fmt or '.4g')
o=[]
w=o.append
w('''# Round 3 post-hoc analysis report (updates Round 2; Tables S1-S6 unchanged, S7-S9 new)

Manuscript: *Physics-Structured PSO-Fuzzy Control and Residual TD3 for a RWIP: Equilibrium Bias and Performance Limits* (IJIES Paper ID 20266940).
Status: **DRAFT prepared for the authors to verify.** Nothing here is a retraining experiment. Every number below is computed from the stored checkpoints by a Python re-implementation of the plant and loop, and nothing was re-run in MATLAB/Simulink.

## 0. Scope, inputs, validity

| Item | Content |
|---|---|
| Inputs (read only) | `rl_v13_repro/agent_bc_seed{1..5}.mat` (BC+TD3), `rl_v13_repro/agent_bconly_seed{1..5}.mat` (BC-only), `fis_rwip_v13_best.fis`, plant constants and gains as in `sim_rwip_v13.m` |
| Simulator | `rwipsim.py` + `actorx.py` (your audit scripts, unchanged); `phlib.py` (new, float re-implementation of the same loop that records clipping events and actor outputs) |
| Validation | `phlib.py` is bit-identical to `rwipsim.py` on the checked runs and reproduces the MATLAB Table 9 values and the 1.0 and 2.0 rad/s impulse rows printed in the manuscript (`t_val.py`, `t_dist.py`; logs `run.log`, `dist.log`) |
| Plant / solver | RK4, 1 ms; agent period 10 ms (ZOH); 201x201 LUT with bilinear interpolation; nominal IC theta0 = 0.10 rad, other states 0 |
| Cross-check with stored MATLAB output (Round 3) | 60 s trajectories of all 10 raw policies and the 600 s PSO-FLC trajectory agree with the stored MATLAB results (Table S8). The 600 s RL runs have no stored MATLAB trajectories; they are checked only against the printed Table 11 medians (111.57 / 1144.59). BC-only impulse runs and all `unbias` runs (except the printed Table 9 values) have no MATLAB reference |
| Not covered | No SMC-ESO port exists in Python, so no SMC-ESO number in this report is new; no Simulink run was made |
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
**Result.** Closed form vs simulated phi(600 s): |error| <= 2.2e-13 for 19 of 20 policy variants; the exception is BC-only seed 3 raw (error 0.034, non-settling, see above). Injection: phi_ss = -/+2.7317 rad at b = +/-0.3; log-log slope 0.976; Pearson r(|b|, IAE_phi(60 s)) = %.4f over 11 injected values; injection ratio IAE_phi(600)/IAE_phi(60) = %.2f-%.2f (the earlier manuscript quoted 9.14-10.78 for every b != 0; here the positive-b side b = +0.02 ... +0.3 gives exactly 9.14-10.78, whereas the negative side gives 8.12-10.31 (b = -0.02: 8.12; -0.05: 9.39), so the range over all ten non-zero b of the 11-point sweep is 8.12-10.78. The original `bias_injection_experiment.py` and its grid of b are not available, so the authors must confirm the grid).
Local stability of the 10 ms closed-loop map (`local_closed_loop.csv`): all 10 corrected policies have the origin as a fixed point (phi ~ 1e-14) with spectral radius 0.9965-0.9977; raw fixed points have radius 0.9969-0.9979 (BC-only seed 3 raw: 1.001). Small-amplitude unbias runs (`unbias_small_amplitude.csv`) converge to 0.


## Table S7. BC+TD3 minus BC-only, per seed, after removing pi(0) (post-hoc; proposal for Table 10 or a supplementary table)

Script `posthoc_round3_bco_td3.py` -> `results/bco_td3_paired.csv`. Unit of analysis: seed k pairs the BC-only and BC+TD3 networks that share seed and BC initialization (n = 5). d_k = TD3_k - BCO_k. Exact two-sided sign-flip test over 2^5 = 32 sign patterns (smallest attainable p = 0.0625); Hodges-Lehmann estimate = median of the 15 Walsh averages; the interval [min d, max d] is the exact Walsh interval with confidence 1 - 2/32 = 93.75 %% (assumes a symmetric distribution of d). 16 contrasts are listed; **no correction for multiple comparisons** was applied; all of this is post-hoc. "Impulse" rows use Python only: the BC-only impulse values have no MATLAB reference (the BC+TD3 impulse rows at 1.0 and 2.0 rad/s reproduce the printed MATLAB values).

@@S7@@
**How to read it.** A large p-value at n = 5 is not evidence that the two methods are equivalent: for nominal IAE_phi after correction the paired differences range from -6.33 to +7.98 around a BC-only median of 4.22, i.e. the data are compatible with differences of either sign and of a size comparable to the level itself. Nominal IAE_theta differences are small (median +0.0009, range -0.0031 to +0.0023). In the impulse scenarios the corrected BC+TD3 policies show the higher IAE_theta in 5/5 (1.0 rad/s, median +0.0040 on 0.128, p = 0.0625) and 4/5 (2.0 rad/s) seeds and the lower rho in 5/5 seeds; with n = 5 these are suggestive, not established. None of this supports the statement "TD3 helps" or "TD3 does not matter" in general; it applies to the five stored seeds of one training configuration.

Proposed text for the authors (not inserted in the manuscript; decision item 8 of the checklist): *"After removing pi(0) from both groups, the paired nominal IAE_phi difference (BC+TD3 minus BC-only) has a median of +1.15 rad s and ranges from -6.33 to +7.98 over the five seeds (post-hoc, Python re-implementation; supplementary Table S7)."* A table row set for Table 10 (nominal, unbias): BC-only IAE_theta @@BCO_TH@@, IAE_phi @@BCO_PH@@; BC+TD3 IAE_theta @@TD_TH@@, IAE_phi @@TD_PH@@ (median [Q1-Q3], MATLAB quantile convention).

## Table S8. Python re-implementation versus stored MATLAB output

Script `matlab_crosscheck.py` -> `results/matlab_crosscheck.csv`. Inputs (read only): `BC_vs_TD3_5seed_20260811_174452.mat` (60 s nominal trajectories of the ten raw policies and PSO-FLC) and `Test_LongHorizon_v13_results.mat` (600 s PSO-FLC). Metric: maximum absolute difference between the two trajectories over all samples.

@@S8@@

Also: raw IAE_phi, IAE_theta and rho of the ten raw policies agree with `BC_vs_TD3_5seed_paired_20260811_174452.csv` to a relative difference of at most 1.4e-6; the PSO-FLC 600 s IAE_phi agrees to 1e-13 relative (3.2110413201947967 vs 3.211041320194251). The remaining differences (about 1e-7 in phi) are consistent with single- versus double-precision network weights; this explanation was not tested.

## Table S9. Cross-check of numbers printed in the Round 3 manuscript

Script `crosscheck_round3.py` -> `results/crosscheck_round3.csv` (one row per number: location, manuscript value, recomputed value, source file, status).

@@S9@@

## Items not reproducible from the current inputs

`bias_injection_experiment.py`, `fis_verify.py`, `rwip_full_sim.py`, `make_fig_*.py`, `fis_surface_201.npy`, `citation_mapping.json` (named in earlier notes but absent); the Python PSO script/log that produced the earlier PSO particle; the training-run log; MATLAB and toolbox versions; an SMC-ESO port and any LQR/state-feedback code.
''' % (S['injection_pearson_r_absb_vs_IAEphi60_11pts'],*S['injection_ratio_range']))

bt=pd.read_csv(R+'bco_td3_paired.csv')
w2=lambda d: md(d)
disp=bt[['scenario','variant','metric','bco_median','td3_median','n_td3_higher','d_median','d_HL','interval_lo','interval_hi','signflip_p']]
o=[x.replace('@@S7@@',md(disp,'.4g')) for x in o]
mc=pd.read_csv(R+'matlab_crosscheck.csv')
o=[x.replace('@@S8@@',md(mc,'.3g')) for x in o]
cc=pd.read_csv(R+'crosscheck_round3.csv')
summ=cc.groupby('status').size().reset_index(name='count')
o=[x.replace('@@S9@@','Result: %d numbers checked, %d match, %d mismatch.\n\n'%(len(cc),(cc.status=='PASS').sum(),(cc.status!='PASS').sum())+md(cc.groupby('location').status.agg(n='count').reset_index())) for x in o]
# quartiles (MATLAB convention) for proposal rows
def q(x,p):
    x=np.sort(np.asarray(x));n=len(x);pos=p*n+0.5
    lo=int(np.floor(pos)); fr=pos-lo
    return x[lo-1]+fr*(x[min(lo,n-1)]-x[lo-1]) if lo>=1 and lo<n else (x[0] if lo<1 else x[-1])
rr=pd.read_csv(R+'runs_600s.csv')
def fmt(pol,col,d):
    v=rr[(rr.policy==pol)&(rr.variant=='unbias')][col].values
    return ('%.'+str(d)+'f [%.'+str(d)+'f-%.'+str(d)+'f]')%(np.median(v),q(v,0.25),q(v,0.75))
ph={'@@BCO_TH@@':fmt('bco','IAEtheta_60',4),'@@BCO_PH@@':fmt('bco','IAEphi_60',2),'@@TD_TH@@':fmt('td3','IAEtheta_60',4),'@@TD_PH@@':fmt('td3','IAEphi_60',2)}
for k,v in ph.items(): o=[x.replace(k,v) for x in o]
print(ph)
# self-check of quartile convention against MATLAB statistics (raw BC-only IAEtheta)
ms=pd.read_csv(os.path.join(ROOT,'BC_vs_TD3_5seed_statistics_20260811_174452.csv')).set_index('Metric')
v=rr[(rr.policy=='bco')&(rr.variant=='raw')].IAEtheta_60.values
print('quartile convention check', q(v,0.25),ms.loc['IAEtheta','BC_Q25'], q(v,0.75),ms.loc['IAEtheta','BC_Q75'])

open('Round3_PostHoc_Report.md','w').write('\n'.join(o))
