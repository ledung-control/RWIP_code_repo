"""build_report4.py -- Round 4 post-hoc report = Round 3 report (r3 md) with S9 recomputed and S10, S11 and proposal P1 added."""
import pandas as pd, numpy as np, json, re, sys
R='results/'; SRC=sys.argv[1]; OUT=sys.argv[2]
S=json.load(open(R+'summary.json'))
md=lambda df,f='.4g': df.to_markdown(index=False,floatfmt=f)
t=open(SRC,encoding='utf8').read()
t=t.replace("# Round 3 post-hoc analysis report (updates Round 2; Tables S1-S6 unchanged, S7-S9 new)","# Round 4 post-hoc analysis report (updates Round 3; Tables S1-S8 as in Round 3, S9 recomputed; S10, S11 and proposal P1 new)")
# ---- S9 recomputed from crosscheck_round4.csv
cc=pd.read_csv(R+'crosscheck_round4.csv')
s9=("## Table S9. Cross-check of numbers printed in the Round 4 working manuscript\n\nScript `crosscheck_round4.py` -> `results/crosscheck_round4.csv` (one row per number: location, manuscript value, recomputed value, source file, status). Round 3 checks plus the Round 4 additions (Figure 4 values, injection ratios per b, proposed Table 10 rows).\n\nResult: %d numbers checked, %d match, %d mismatch.\n\n"%(len(cc),(cc.status=='PASS').sum(),(cc.status!='PASS').sum())+md(cc.groupby('location').status.agg(n='count').reset_index())+"\n\n")
i=t.index('## Table S9.'); j=t.index('## Items not reproducible')
t=t[:i]+s9+'@@NEW@@'+t[j:]
# ---- S10
sw=pd.read_csv(R+'bias_injection_sweep.csv'); sw['ratio']=(sw.IAEphi_600/sw.IAEphi_60).where(sw.b!=0)
t10=sw[['b','IAEphi_60','IAEphi_600','ratio','phi_600']]
pdsp=pd.read_csv(R+'param_displacement.csv'); al=pdsp[pdsp.tensor=='ALL'].sort_values('seed')
rng_s='%.2f-%.2f'%(al.L2_change.min(),al.L2_change.max())
per=', '.join('seed %d: %.3f'%(a,b) for a,b in zip(al.seed,al.L2_change)); rel=', '.join('%.1f %%'%(100*a/b) for a,b in zip(al.L2_change,al.L2_norm_BCinit))
new=f'''## Table S10. Injection sweep: IAE_phi(600 s)/IAE_phi(60 s) per b (R1-8, A7)

Script `steady_state_check.py` -> `results/bias_injection_sweep.csv` (11-point grid b = 0, +-0.02, +-0.05, +-0.10, +-0.20, +-0.30: the list printed in the original Section 2.3.5). A constant bias replaces the actor output, so no trained network is involved: only the plant constants, the official FIS and the validated simulator. The original script (`bias_injection_experiment.py`) is not available, so the authors must confirm that their original run used this grid.

{md(t10)}

**Result.** Over the ten non-zero b the ratio lies between {sw.ratio.min():.2f} (b = -0.02) and {sw.ratio.max():.2f} (b = +0.10). The positive-b side alone gives {sw[sw.b>0].ratio.min():.2f}-{sw[sw.b>0].ratio.max():.2f}, the range printed in the earlier manuscript; the negative-b side gives {sw[sw.b<0].ratio.min():.2f}-{sw[sw.b<0].ratio.max():.2f}. The earlier statement "9.14-10.78 for every b != 0" therefore holds only for b > 0 in this regeneration. The Pearson correlation of |b| with IAE_phi(60 s) over the 11 points is {S['injection_pearson_r_absb_vs_IAEphi60_11pts']:.4f}.

## Table S11. Provenance of the quantities printed in the redrawn Figure 4 (R1-6, B9)

The original Figure 4 was a raster image whose source script (`make_fig_*.py`) is not available. Its Stage B box printed "(displacement ~ 0.36 units)", a nominal budget (learning rate x episodes x updates = 1e-5 x 18 x 2000), not a measured quantity. The figure was redrawn with `make_fig4_round4.py` (matplotlib; same boxes, text and colours; only the third line of the Stage B box changed). Quantities printed:

| Printed in the figure | Source |
|---|---|
| RWIP state; PSO-FLC fixed baseline; residual actor, 2 hidden layers, 256/layer | Sections 2.1.3, 2.3.1, Table 3 (unchanged from the original figure) |
| Stage A: behavior cloning from SMC-ESO (ZOH) | Section 2.3.3 (unchanged) |
| Stage B: 18 episodes, actor LR 1e-5 | `Train_RL_v13_repro.m` / Table 3 (unchanged) |
| Stage B: L2 change of the actor parameters vs. Stage A = {rng_s} | `results/param_displacement.csv`, rows tensor = ALL (67841 parameters = 6x256 + 256 + 256x256 + 256 + 256 + 1); for each seed k the difference between the weights of `agent_bc_seed{{k}}` and `agent_bconly_seed{{k}}`; seeds 1-5 |
| Stage C: a_bias = pi(0), pi'(s) = pi(s) - a_bias; u_corr = 10 V x a, a in [-1, 1]; u_tot = sat(u_FLC + u_corr, +-24 V) | Sections 2.3.1, 2.3.4 (unchanged) |

Per-seed L2 changes: {per}. Relative to the L2 norm of the BC-only weights: {rel}. The L2 norm is over network weights and has no physical unit.

'''
# ---- proposal P1
rr=pd.read_csv(R+'runs_600s.csv')
def q(x,p):
    x=np.sort(np.asarray(x,float));n=len(x);pos=p*n+0.5
    if pos<=1: return x[0]
    if pos>=n: return x[-1]
    k=int(np.floor(pos));f=pos-k;return x[k-1]+f*(x[k]-x[k-1])
def sel(pol,col): return rr[(rr.policy==pol)&(rr.variant=='unbias')].sort_values('seed')[col].values
def mq(pol,col,d): v=sel(pol,col); return ('%.'+str(d)+'f [%.'+str(d)+'f-%.'+str(d)+'f]')%(np.median(v),q(v,.25),q(v,.75))
bt=pd.read_csv(R+'bco_td3_paired.csv'); g=lambda m: bt[(bt.scenario=='nominal')&(bt.variant=='unbias')&(bt.metric==m)].iloc[0]
th,ph=g('IAEtheta'),g('IAEphi')
per_df=pd.DataFrame({'seed':[1,2,3,4,5],'BC-only IAE_theta':sel('bco','IAEtheta_60'),'BC+TD3 IAE_theta':sel('td3','IAEtheta_60'),'d IAE_theta':sel('td3','IAEtheta_60')-sel('bco','IAEtheta_60'),
 'BC-only IAE_phi':sel('bco','IAEphi_60'),'BC+TD3 IAE_phi':sel('td3','IAEphi_60'),'d IAE_phi':sel('td3','IAEphi_60')-sel('bco','IAEphi_60')})
dth=sel('td3','IAEtheta_60')-sel('bco','IAEtheta_60'); dph=sel('td3','IAEphi_60')-sel('bco','IAEphi_60')
prop=f'''## Proposal P1. BC-only after removal of pi(0): where to put it (A8, R2-2)

All numbers are post-hoc, n = 5 seeds, computed from the stored checkpoints by the Python re-implementation (no MATLAB reference for the corrected runs). Unit of analysis: seed k pairs the BC-only and the BC+TD3 network that share seed and BC initialization; d_k = BC+TD3_k - BC-only_k after subtracting the respective pi(0) from both (nominal scenario, 60 s).

{md(per_df,'.5g')}

| Statistic (BC+TD3 minus BC-only, pi(0) removed) | IAE_theta (rad s) | IAE_phi (rad s) |
|---|---|---|
| n seeds | 5 | 5 |
| median of d | {np.median(dth):+.4f} | {np.median(dph):+.2f} |
| Hodges-Lehmann estimate | {th.d_HL:+.4f} | {ph.d_HL:+.2f} |
| interval [min d, max d] (Walsh; confidence 1 - 2/2^5 = 93.75 %, assumes a symmetric distribution of d) | [{dth.min():+.4f}, {dth.max():+.4f}] | [{dph.min():+.2f}, {dph.max():+.2f}] |
| seeds with BC+TD3 higher | {(dth>0).sum()} of 5 | {(dph>0).sum()} of 5 |
| exact two-sided sign-flip p (32 patterns; smallest attainable 0.0625) | {th.signflip_p:.4f} | {ph.signflip_p:.4f} |

**What this does and does not show.** With n = 5 a large p is not evidence of equality: the interval for IAE_phi runs from {dph.min():+.2f} to {dph.max():+.2f} around a BC-only median of {np.median(sel('bco','IAEphi_60')):.2f}, so differences of either sign and of the size of the level itself are compatible with the data. The comparison concerns one training configuration. Table S7 lists 16 contrasts and no correction for multiple comparisons was applied. Both groups remain above PSO-FLC in median IAE_phi (3.211).

**Option A (main table, Table 10).** Add four rows. Measured on a scratch copy of the working manuscript: 22 pages before and after in LibreOffice, about 0.15 page of extra vertical space (Word may paginate differently):

| Metric | PSO-FLC | PSO-FLC + SMC-ESO | BC-only, n=5 | BC+TD3, n=5 |
|---|---|---|---|---|
| Nominal IAE_theta, pi(0) removed (post hoc) | 0.02446 | 0.02662 | {mq('bco','IAEtheta_60',4)} | {mq('td3','IAEtheta_60',4)} |
| Nominal IAE_phi, pi(0) removed (post hoc) | 3.211 | 4.949 | {mq('bco','IAEphi_60',2)} | {mq('td3','IAEphi_60',2)} |
| Paired BC+TD3 - BC-only, IAE_theta; median [min, max] (post hoc) | - | - | - | {np.median(dth):+.4f} [{dth.min():+.4f}, {dth.max():+.4f}] |
| Paired BC+TD3 - BC-only, IAE_phi; median [min, max] (post hoc) | - | - | - | {np.median(dph):+.2f} [{dph.min():+.2f}, {dph.max():+.2f}] |

The first two rows give median [Q1-Q3] (MATLAB quantile convention). The existing Table 10 columns hold raw policies, so the caption must say that these rows are after removal of pi(0).

**Option B (supplement only).** Keep Table S7 (all scenarios, 16 contrasts) in the supplementary report, with the sentences already in Section 3.7. No page effect in the manuscript, but the IJIES pages found give no channel for supplementary files (see `Round4_IJIES_Pages_and_Supplementary`).

**Option C (both).** Option A in the main text and S7 in the supplement.

The authors choose the option and approve the table position and the page effect. Impulse-scenario rows (BC-only impulse runs have no MATLAB reference) exist only in S7.

'''
t=t.replace('@@NEW@@',new+prop)
open(OUT,'w',encoding='utf8').write(t)
print(len(cc),(cc.status=='PASS').sum()); print(per_df.round(4).to_string())
