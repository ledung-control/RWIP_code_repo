import pandas as pd, numpy as np
R='results/'
e3=pd.read_csv(R+'E3_MATLAB_2026a.csv'); e3=e3[e3.Rate=='all100']
ev=pd.read_csv(R+'E3c_final/E3c_eval_MATLAB_2026a.csv'); ps=pd.read_csv(R+'E3c_final/E3c_pso_MATLAB_2026a.csv')
for d in (e3,ev): d['C']=d.Case+np.where(d.Case=='kick',' '+d.Kick.astype(str),'')
out=[]; P=lambda *a: out.append(' '.join(str(x) for x in a))
P('E3c: SMC-ESO re-optimized with the PSO-FLC budget (5 PSO runs) at 100 Hz vs authors SMC-ESO gains, PSO-FLC, BC+TD3 (unbiased)')
P(ps[['Run','J','gv','OS_theta','ts_theta','phi_peak','ts_phi']].to_string(index=False))
cases=list(dict.fromkeys(ev.C)); au=ev[ev.Variant=='authors']; rt=ev[ev.Variant.str.startswith('retuned')]
pf=e3[e3.Controller=='PSO-FLC']; ag=e3[e3.Controller.str.startswith('BC+TD3')&(e3.Variant=='unbiased')]
for m in ['IAEphi','IAEtheta']:
    rows=[]
    for c in cases:
        r=rt[rt.C==c][m]
        rows.append((c,au[au.C==c][m].mean(),r.median(),r.min(),r.max(),pf[pf.C==c][m].mean(),ag[ag.C==c][m].median()))
    P('\n==',m); P(pd.DataFrame(rows,columns=['case','SMC authors','SMC retuned med','min','max','PSO-FLC','BC+TD3 med']).round(4).to_string(index=False))
P('\nRecovered: SMC authors',int(au.Recovered.sum()),'/',len(au),'; SMC retuned',int(rt.Recovered.sum()),'/',len(rt),'; diverged retuned',int(rt.Diverged.sum()),'; PSO-FLC',int(pf.Recovered.sum()),'/',len(pf),'; BC+TD3',int(ag.Recovered.sum()),'/',len(ag))
P('Recovered per retuned run:',rt.groupby('Variant').Recovered.sum().to_dict())
open(R+'E3c_final_analysis.txt','w').write('\n'.join(out)); print('\n'.join(out))
