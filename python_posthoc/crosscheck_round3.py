"""crosscheck_round3.py -- verify each number printed in the Round 2 manuscript/letter/report against its source.
Sources: results/*.csv, results/summary.json (post-hoc, Python), and stored MATLAB outputs (read only):
  Compare_v13_robust_results.mat (Table 5), Test_LongHorizon_v13_results.mat (PSO-FLC 600 s),
  pso_rwip_v13_result.mat (PSO/Simulink evaluation), BC_vs_TD3_5seed_paired_*.csv (60 s raw).
Output: results/crosscheck_round3.csv"""
import os, sys, json, csv
import numpy as np, pandas as pd, scipy.io as sio
ROOT = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/WRIP_2'
HERE = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(HERE, 'results')
S = json.load(open(os.path.join(RS, 'summary.json')))
runs = pd.read_csv(os.path.join(RS, 'runs_600s.csv'))
pdisp = pd.read_csv(os.path.join(RS, 'param_displacement.csv'))
nb = pd.read_csv(os.path.join(RS, 'pi_minus_pi0_neighbourhood.csv')); nb = nb[nb.policy == 'td3']
jac = pd.read_csv(os.path.join(RS, 'pi_jacobian_at_origin.csv')); jac = jac[jac.policy == 'td3']
ss = pd.read_csv(os.path.join(RS, 'steady_state_check.csv'))
hr = pd.read_csv(os.path.join(RS, 'horizon_ratio_per_seed.csv'))
cl = pd.read_csv(os.path.join(RS, 'clipping_60s.csv'))
dc = pd.read_csv(os.path.join(RS, 'disturbance_clipping_and_metrics.csv'))
lh = sio.loadmat(os.path.join(ROOT, 'Test_LongHorizon_v13_results.mat'), squeeze_me=True, struct_as_record=False)
rb = sio.loadmat(os.path.join(ROOT, 'Compare_v13_robust_results.mat'), squeeze_me=True, struct_as_record=False)['R1']
pso = sio.loadmat(os.path.join(ROOT, 'pso_rwip_v13_result.mat'), squeeze_me=True, struct_as_record=False)
rows = []
def chk(cid, loc, claim, text_val, comp, tol, src, mode='abs'):
    ok = (abs(text_val - comp) <= tol) if mode == 'abs' else (abs(text_val - comp) / abs(comp) <= tol)
    rows.append([cid, loc, claim, text_val, comp, 'PASS' if ok else 'MISMATCH', src])
def rng(cid, loc, claim, lo, hi, vals, d, src):
    chk(cid+'a', loc, claim+' (min)', lo, min(vals), 0.5*10**-d+1e-12, src)
    chk(cid+'b', loc, claim+' (max)', hi, max(vals), 0.5*10**-d+1e-12, src)
T = lambda k, v: runs[(runs.policy == k) & (runs.variant == v)].sort_values('seed')
# --- Table 5 / Section 3.2 (MATLAB source)
nomS, ktS, advS = rb[0,1].IAEtheta, rb[6,1].IAEtheta, rb[7,1].IAEtheta
src5 = 'Compare_v13_robust_results.mat R1'
chk('T5-1','3.2/Table 5','SMC-ESO nominal IAEtheta',0.13692,nomS,5e-6,src5)
chk('T5-2','3.2/Table 5','SMC-ESO Kt-30% IAEtheta',5.4401,ktS,5e-5,src5)
chk('T5-3','3.2/Table 5','SMC-ESO adverse IAEtheta',7.2367,advS,5e-5,src5)
chk('T5-4','3.2','Kt-30% increase, stored (%)',3873.1,100*(ktS/nomS-1),0.05,src5)
chk('T5-5','3.2','Kt-30% increase from rounded printed values (reviewer 3873.20)',3873.196,100*(5.4401/0.13692-1),0.001,'arithmetic')
chk('T5-6','3.2/Table 5','adverse increase, stored (%)',5185.2,100*(advS/nomS-1),0.05,src5)
chk('T5-7','3.2','adverse increase, rounded printed values (%)',5185.3,100*(7.2367/0.13692-1),0.05,'arithmetic')
chk('T5-8','3.2','IAE ratio Kt-30%',39.7,ktS/nomS,0.05,src5)
chk('T5-9','3.2','PSO-FLC Kt-30% increase (%)',13.27,100*(rb[6,0].IAEtheta/rb[0,0].IAEtheta-1),0.005,src5)
chk('T5-10','3.2','hybrid mgl-30% increase (%)',9.46,100*(rb[2,2].IAEtheta/rb[0,2].IAEtheta-1),0.005,src5)
chk('T5-11','Table 5','PSO-FLC Kt-30% value',0.0277,rb[6,0].IAEtheta,5e-5,src5)
chk('T5-12','Table 5','PSO-FLC adverse value',0.0263,rb[7,0].IAEtheta,5e-5,src5)
chk('T5-13','Table 5','hybrid Kt-30% value',0.0274,rb[6,2].IAEtheta,5e-5,src5)
chk('T5-14','Table 5','hybrid adverse value',0.0270,rb[7,2].IAEtheta,5e-5,src5)
hy = [100*(rb[i,2].IAEtheta/rb[0,2].IAEtheta-1) for i in range(1,8)]
pf = [100*(rb[i,0].IAEtheta/rb[0,0].IAEtheta-1) for i in range(1,8)]
chk('T5-15','3.2','hybrid largest of 7 cases is the mgl-30% case (+9.46)',9.46,max(hy),0.005,src5)
chk('T5-16','3.2','PSO-FLC largest of 7 cases is Kt-30% (+13.27)',13.27,max(pf),0.005,src5)
# --- OS / settling (Section 2.1.4, Table 4)
R1 = lh['R1'][0]
chk('OS-1','2.1.4/Table 4','OS RK4 (MATLAB sim_rwip_v13, 600 s file)',0.005894,R1.OS_theta,5e-7,'Test_LongHorizon_v13_results.mat R1(1)')
chk('OS-2','2.1.4','OS RK4 Python re-evaluation',0.005894,S['pso_flc_rk4_1ms']['OS'],5e-7,'summary.json')
chk('OS-3','2.1.4','OS Simulink/PSO evaluation (stored Mbest)',0.005987,pso['Mbest'].OS_theta,5e-7,'pso_rwip_v13_result.mat')
chk('OS-4','2.1.4','ts_theta Simulink',2.78,pso['Mbest'].ts_theta,1e-9,'pso_rwip_v13_result.mat')
chk('OS-5','2.1.4','phi peak Simulink',0.519,pso['Mbest'].phi_peak,5e-4,'pso_rwip_v13_result.mat')
chk('OS-6','2.1.4','ts_phi Simulink',10.72,pso['Mbest'].ts_phi,1e-9,'pso_rwip_v13_result.mat')
chk('OS-7','2.1.4/Table 4','ts_phi RK4',10.629,R1.ts_phi,1e-9,'Test_LongHorizon_v13_results.mat R1(1)')
chk('OS-8','2.1.4/Table 4','ts_theta RK4',2.777,R1.ts_theta,1e-9,'Test_LongHorizon_v13_results.mat R1(1)')
chk('OS-9','2.1.4/Table 4','phi peak RK4',0.517,R1.phi_max,5e-4,'Test_LongHorizon_v13_results.mat R1(1)')
chk('OS-10','2.1.4','J of PSO result',0.314771,pso['GBestCost'],5e-7,'pso_rwip_v13_result.mat')
chk('OS-11','MATLAB vs Python','IAEphi(600) PSO-FLC relative difference',0.0,abs(R1.IAEphi-S['pso_flc_rk4_1ms']['IAEphi_600'])/R1.IAEphi,1e-10,'both')
# --- 2.3.3 parameter displacement
L2=[v['L2'] for v in S['param_change_ALL'].values()]; rel=[100*v['rel'] for v in S['param_change_ALL'].values()]; li=[v['Linf'] for v in S['param_change_ALL'].values()]
rng('PD-L2','2.3.3','L2 change',3.83,4.06,L2,2,'summary.json'); rng('PD-rel','2.3.3','relative change %',17.7,19.3,rel,1,'summary.json'); rng('PD-inf','2.3.3','Linf',0.018,0.024,li,3,'summary.json')
# --- clipping 2.3.1
chk('CL-1','2.3.1','largest pre-clip |a| in 60 evaluated runs',0.86,max(dc.max_abs_residual_pre_clip.max(),cl.max_abs_residual_pre_clip.max()),0.005,'clipping_60s.csv + disturbance csv')
chk('CL-2','2.3.1','max residual clip steps',0,max(dc.residual_clip_steps.max(),cl.residual_clip_steps.max()),0,'csv')
chk('CL-3','2.3.1','max total-clip ms steps in impulse runs',8,dc.total_clip_ms.max(),0,'disturbance csv')
chk('CL-4','2.3.1','total clip in nominal',0,cl.total_clip_ms.max(),0,'clipping_60s.csv')
chk('CL-5','2.3.1','number of evaluated runs = 10 policies x raw/corr x 3 scen (impulse file rows 40 + nominal 20)',60,len(dc[dc.policy!='pso_flc'])+len(cl),0,'csv rows')
# --- 2.3.4 steady state
chk('SS-1','2.3.4','bias limit V',5.89,S['steady_state']['Ku_FLC_at_sat_V'],0.005,'summary.json')
chk('SS-2','2.3.4','bias limit action',0.589,S['steady_state']['b_max_action'],0.0005,'summary.json')
chk('SS-3','2.3.4','0.035/Kp rad',8.0,S['steady_state']['phi_scale_0p035_over_Kp'],0.01,'summary.json')
good = ss[ss.abs_error < 1e-12]
chk('SS-4','2.3.4/S6','variants (of 20) with |error|<1e-12',19,len(good),0,'steady_state_check.csv')
rows.append(['SS-5','S6','max error among the 19 is <= 2.2e-13 (inequality)',2.2e-13,good.abs_error.max(),'PASS' if good.abs_error.max()<=2.2e-13 else 'MISMATCH','steady_state_check.csv'])
sw = pd.read_csv(os.path.join(RS,'bias_steady_state_theory.csv')); sim = pd.read_csv(os.path.join(RS,'bias_injection_sweep.csv'))
chk('SS-6','3.8','phi_ss at b=+0.3 (sim)',-2.7317,float(sim[np.isclose(sim.b,0.3)].phi_600.iloc[0]),5e-5,'bias_injection_sweep.csv')
chk('SS-7','3.8','phi_ss at b=-0.3 (sim)',2.7317,float(sim[np.isclose(sim.b,-0.3)].phi_600.iloc[0]),5e-5,'bias_injection_sweep.csv')
chk('SS-8','3.8','injection r (11 pts)',0.996,S['injection_pearson_r_absb_vs_IAEphi60_11pts'],5e-4,'summary.json')
chk('SS-9','3.9','injection ratio min',8.12,S['injection_ratio_range'][0],0.005,'summary.json')
chk('SS-10','3.9','injection ratio max',10.78,S['injection_ratio_range'][1],0.005,'summary.json')
# --- 3.9 ratios
t11 = S['table11_td3_raw']; b11 = S['table11_bco_raw']
chk('H-1','3.9/Table 11','median IAEphi 60 s raw BC+TD3',111.57,t11['med_60'],0.005,'summary.json')
chk('H-2','3.9/Table 11','median IAEphi 600 s raw BC+TD3',1144.59,t11['med_600'],0.005,'summary.json')
chk('H-3','3.9','ratio of medians',10.26,t11['ratio_of_medians'],0.005,'summary.json')
rng('H-4','3.9','BC+TD3 raw per-seed ratios',7.01,10.70,t11['per_seed_ratio'],2,'summary.json')
rng('H-5','3.9','BC-only raw per-seed ratios',5.67,10.83,b11['per_seed_ratio'],2,'summary.json')
chk('H-6','3.9','negative intercepts count (BC+TD3 raw)',4,sum(x<0 for x in t11['per_seed_intercept']),0,'summary.json')
chk('H-7','3.9','gap raw 60 s vs PSO-FLC',34.75,t11['med_60']/S['pso_flc_rk4_1ms']['IAEphi'],0.005,'summary.json')
chk('H-8','3.9','gap raw 600 s vs PSO-FLC',356.46,t11['med_600']/S['pso_flc_rk4_1ms']['IAEphi_600'],0.005,'summary.json')
chk('H-9','3.9','gap unbias',1.73,S['table11_td3_unbias']['med_60']/S['pso_flc_rk4_1ms']['IAEphi'],0.005,'summary.json')
c = runs[(runs.policy=='bco')&(runs.variant=='raw')&(runs.seed==3)].iloc[0]
chk('H-10','3.9','BC-only s3 raw IAEtheta(600)',0.0770,c.IAEtheta_600,5e-5,'runs_600s.csv')
chk('H-11','3.9','ratio IAEtheta 600/60',2.1,c.IAEtheta_600/c.IAEtheta_60,0.05,'runs_600s.csv')
# --- 3.7 / 3.6.1 / Tables 9, 10
u = T('bco','unbias'); chk('M-1','3.7','BC-only max |aBias|',0.0054,T('bco','raw').aBias.abs().max(),5e-5,'runs_600s.csv')
chk('M-2','3.7','BC-only unbias median IAEphi',4.22,u.IAEphi_60.median(),0.005,'runs_600s.csv')
chk('M-3','3.7','BC-only unbias min IAEphi',3.99,u.IAEphi_60.min(),0.005,'runs_600s.csv'); chk('M-4','3.7','BC-only unbias max IAEphi',11.88,u.IAEphi_60.max(),0.005,'runs_600s.csv')
chk('M-5','3.7','BC-only unbias median IAEtheta',0.0288,u.IAEtheta_60.median(),5e-5,'runs_600s.csv')
chk('M-6','3.7','BC-only unbias min IAEtheta',0.0280,u.IAEtheta_60.min(),5e-5,'runs_600s.csv'); chk('M-7','3.7','BC-only unbias max IAEtheta',0.0314,u.IAEtheta_60.max(),5e-5,'runs_600s.csv')
chk('M-8','3.7','BC-only raw median IAEphi',12.94,T('bco','raw').IAEphi_60.median(),0.005,'runs_600s.csv')
pf = S['paired_signflip']
chk('M-9','3.7','paired median dIAEphi after unbias',1.15,pf['TD3_minus_BCO_unbias:IAEphi']['median'],0.005,'summary.json')
chk('M-10','3.7','sign-flip p IAEphi',0.63,pf['TD3_minus_BCO_unbias:IAEphi']['exact_signflip_two_sided_p'],0.0051,'summary.json')
chk('M-11','3.7','dIAEtheta median',0.0009,pf['TD3_minus_BCO_unbias:IAEtheta']['median'],5e-5,'summary.json')
chk('M-12','3.7','sign-flip p IAEtheta',0.94,pf['TD3_minus_BCO_unbias:IAEtheta']['exact_signflip_two_sided_p'],0.005,'summary.json')
chk('M-13','3.7','median IAEphi unbias BC+TD3 above PSO-FLC (5.55 vs 3.211)',5.55,T('td3','unbias').IAEphi_60.median(),0.005,'runs_600s.csv')
chk('M-14','3.6.1','paired change min raw->unbias (rad s)',-285.3,pf['unbias_effect_TD3:IAEphi']['diffs'][2],0.05,'summary.json')
chk('M-15','3.6.1','paired change max raw->unbias',-5.5,pf['unbias_effect_TD3:IAEphi']['diffs'][1],0.05,'summary.json')
q = T('td3','unbias').IAEphi_60/S['pso_flc_rk4_1ms']['IAEphi']
chk('M-16','3.6.1','unbias/PSO-FLC factor min',1.10,q.min(),0.005,'runs_600s.csv'); chk('M-17','3.6.1','factor max',3.80,q.max(),0.005,'runs_600s.csv'); chk('M-18','3.6.1','factor median',1.73,q.median(),0.005,'runs_600s.csv')
tt = T('td3','unbias').IAEtheta_60/S['pso_flc_rk4_1ms']['IAEtheta']-1
chk('M-19','3.6.1','seeds meeting +20% criterion',2,int((tt<=0.20).sum()),0,'runs_600s.csv'); chk('M-20','3.6.1','median % above PSO-FLC (corrected in Round 3 from 21.0)',21.1,100*tt.median(),0.05,'runs_600s.csv')
sc = S['stats_corr']
chk('ST-1','3.6.1/Abstract','Pearson |aBias|',0.991,sc['pearson_abs'],5e-4,'summary.json'); chk('ST-2','3.6.1','signed r',0.640,sc['signed_pearson'],5e-4,'summary.json')
chk('ST-3','3.6.1','LOO min',0.9893,min(sc['loo']),5e-5,'summary.json'); chk('ST-4','3.6.1','LOO max',0.9987,max(sc['loo']),5e-5,'summary.json')
chk('ST-5','4.3','Fisher CI low',0.87,sc['fisher_ci95'][0],0.005,'summary.json'); chk('ST-6','4.3','permutation p',0.008,sc['perm_p'],0.0005,'summary.json')
# --- pi - pi(0)
for box,lo,hi in ((0.01,0.010,0.014),(0.1,0.084,0.110)):
    v = nb[np.isclose(nb.box_halfwidth_normalized_obs,box)].std_dev.values
    rng('PI-%g'%box,'3.9','std pi-pi0 box %g'%box,lo,hi,v,3,'pi_minus_pi0_neighbourhood.csv (BC+TD3 only)')
rng('PI-J','S3','Jacobian norm',1.7,2.4,jac.norm.values,1,'pi_jacobian_at_origin.csv')
chk('PI-ss','3.9','a_ss seed1 raw',-0.226,float(ss[(ss.policy=='td3')&(ss.seed==1)&(ss.variant=='raw')].a_ss_residual_action.iloc[0]),5e-4,'steady_state_check.csv')
# --- MATLAB 60 s cross-check (stored csv, 10 policies)
mp = pd.read_csv(os.path.join(ROOT,'BC_vs_TD3_5seed_paired_20260811_174452.csv')).sort_values('Seed')
dd=max(np.max(np.abs(T('td3','raw').IAEphi_60.values-mp.TD3_IAEphi.values)/mp.TD3_IAEphi.values),np.max(np.abs(T('bco','raw').IAEphi_60.values-mp.BC_IAEphi.values)/mp.BC_IAEphi.values))
chk('MX-1','Python vs MATLAB','max relative diff raw IAEphi 60 s (10 policies)',0.0,dd,1e-5,'BC_vs_TD3_5seed_paired csv')

# --- Table 9 (per seed) and Table 10 (medians/quartiles/aBias)
pc = pd.read_csv(os.path.join(RS,'paired_changes.csv'))
T9 = {1:(-0.1199,0.03442,0.03122,111.570,5.975,0.3314,0.2400),2:(-0.0038,0.02498,0.02560,9.013,3.541,0.1226,0.1150),
      3:(0.2768,0.07721,0.02962,297.475,12.194,0.5185,0.2841),4:(-0.0653,0.02593,0.02562,68.817,5.443,0.2325,0.1754),
      5:(-0.1713,0.04270,0.03374,153.408,5.550,0.4169,0.2804)}
for sd,(ab,thr,thu,phr,phu,rr,ru) in T9.items():
    r_=T('td3','raw').set_index('seed').loc[sd]; u_=T('td3','unbias').set_index('seed').loc[sd]
    pr=pc[(pc.contrast=='unbias_effect_TD3')&(pc.seed==sd)].iloc[0]
    for nm,tv,cv,tl in (('aBias',ab,r_.aBias,5e-5),('IAEtheta raw',thr,r_.IAEtheta_60,5e-6),('IAEtheta unbias',thu,u_.IAEtheta_60,5e-6),
                        ('IAEphi raw',phr,r_.IAEphi_60,5e-4),('IAEphi unbias',phu,u_.IAEphi_60,5e-4),('rho raw',rr,pr.rho_from,5e-5),('rho unbias',ru,pr.rho_to,5e-5)):
        chk('T9-s%d-%s'%(sd,nm.replace(' ','_')),'Table 9','seed %d %s'%(sd,nm),tv,cv,tl+1e-12,'runs_600s.csv / paired_changes.csv')
ms = pd.read_csv(os.path.join(ROOT,'BC_vs_TD3_5seed_statistics_20260811_174452.csv')).set_index('Metric')
for nm,tv,cv,tl in (('BC median IAEtheta',0.0283,ms.loc['IAEtheta','BC_Median'],5e-5),('BC Q1 IAEtheta',0.0265,ms.loc['IAEtheta','BC_Q25'],5e-5),('BC Q3 IAEtheta',0.0335,ms.loc['IAEtheta','BC_Q75'],5e-5),
    ('TD3 median IAEtheta',0.0344,ms.loc['IAEtheta','TD3_Median'],5e-5),('TD3 Q1 IAEtheta',0.0257,ms.loc['IAEtheta','TD3_Q25'],5e-5),('TD3 Q3 IAEtheta',0.0513,ms.loc['IAEtheta','TD3_Q75'],5e-5),
    ('BC median IAEphi',12.94,ms.loc['IAEphi','BC_Median'],5e-3),('BC Q1 IAEphi',10.82,ms.loc['IAEphi','BC_Q25'],5e-3),('BC Q3 IAEphi',27.10,ms.loc['IAEphi','BC_Q75'],5e-3),
    ('TD3 median IAEphi',111.57,ms.loc['IAEphi','TD3_Median'],5e-3),('TD3 Q1 IAEphi',53.87,ms.loc['IAEphi','TD3_Q25'],5e-3),('TD3 Q3 IAEphi',189.43,ms.loc['IAEphi','TD3_Q75'],5e-3),
    ('BC median rho',0.2282,ms.loc['rho','BC_Median'],5e-5),('TD3 median rho',0.3314,ms.loc['rho','TD3_Median'],5e-5)):
    chk('T10-'+nm.replace(' ','_'),'Table 10','%s (stored MATLAB statistics csv)'%nm,tv,cv,tl+1e-12,'BC_vs_TD3_5seed_statistics csv')
ab_b=T('bco','raw').aBias.abs(); ab_t=T('td3','raw').aBias.abs()
for nm,tv,cv in (('BC median',0.00256,ab_b.median()),('BC min',0.00091,ab_b.min()),('BC max',0.00538,ab_b.max()),('TD3 median',0.11992,ab_t.median()),('TD3 min',0.00377,ab_t.min()),('TD3 max',0.27676,ab_t.max())):
    chk('T10-ab-'+nm.replace(' ','_'),'Table 10','|aBias| '+nm,tv,cv,5e-6,'runs_600s.csv')
chk('T10-ratio','3.7','IAEphi ratio TD3/BC medians (8.6)',8.6,ms.loc['IAEphi','TD3_Median']/ms.loc['IAEphi','BC_Median'],0.05,'statistics csv')
chk('T10-pct1','3.7','BC-only above PSO-FLC IAEtheta (%) 15.8',15.8,100*(ms.loc['IAEtheta','BC_Median']/S['pso_flc_rk4_1ms']['IAEtheta']-1),0.05,'statistics csv')
chk('T10-pct2','3.7','BC+TD3 above PSO-FLC IAEtheta (%) 40.7',40.7,100*(ms.loc['IAEtheta','TD3_Median']/S['pso_flc_rk4_1ms']['IAEtheta']-1),0.05,'statistics csv')

bt = pd.read_csv(os.path.join(RS,'bco_td3_paired.csv'))
r_ = bt[(bt.scenario=='nominal')&(bt.variant=='unbias')&(bt.metric=='IAEphi')].iloc[0]
chk('M-21','3.7','HL estimate IAEphi',0.83,r_.d_HL,0.005,'bco_td3_paired.csv'); chk('M-22','3.7','range low',-6.33,r_.interval_lo,0.005,'bco_td3_paired.csv'); chk('M-23','3.7','range high',7.98,r_.interval_hi,0.005,'bco_td3_paired.csv')
r_ = bt[(bt.scenario=='nominal')&(bt.variant=='unbias')&(bt.metric=='IAEtheta')].iloc[0]
chk('M-24','3.7','IAEtheta range low',-0.0031,r_.interval_lo,5e-5,'bco_td3_paired.csv'); chk('M-25','3.7','IAEtheta range high',0.0023,r_.interval_hi,5e-5,'bco_td3_paired.csv')

out = pd.DataFrame(rows,columns=['id','location','claim','manuscript_value','recomputed','status','source'])
out.to_csv(os.path.join(RS,'crosscheck_round3.csv'),index=False)
print(out.status.value_counts().to_string()); print(out[out.status!='PASS'].to_string())
