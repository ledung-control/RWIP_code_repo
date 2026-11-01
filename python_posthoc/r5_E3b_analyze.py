import pandas as pd, numpy as np, sys
R='results/'
e3=pd.read_csv(R+'E3_MATLAB_2026a.csv'); e3=e3[e3.Rate=='all100']
ev=pd.read_csv(R+'E3b_final/E3b_eval_MATLAB_2026a.csv')
ps=pd.read_csv(R+'E3b_final/E3b_pso_MATLAB_2026a.csv')
out=[]
P=lambda *a: out.append(' '.join(str(x) for x in a))
P('E3b: PSO-FLC retuned under 100 Hz sampling (5 PSO runs) vs stored agents at 100 Hz')
P(ps[['Run','J','OS_theta','ts_theta','phi_peak','ts_phi']].to_string(index=False))
e3['Case2']=e3.Case+np.where(e3.Case=='kick',' '+e3.Kick.astype(str),'')
ev['Case2']=ev.Case+np.where(ev.Case=='kick',' '+ev.Kick.astype(str),'')
cases=list(dict.fromkeys(ev.Case2))
agall=e3[e3.Controller.str.startswith('BC+TD3')]
ag=agall[agall.Variant=='unbiased']  # agents with pi(0) subtracted (primary comparison); 'raw' nominal reported separately
orig=e3[e3.Controller=='PSO-FLC']
rt=ev[ev.Variant.str.startswith('retuned')]
for m in ['IAEphi','IAEtheta']:
    P('\n==',m,'(median over runs/agents; lower is better)')
    rows=[]
    for c in cases:
        o=orig[orig.Case2==c][m].mean(); r=rt[rt.Case2==c][m]; a=ag[ag.Case2==c][m]
        rows.append((c,o,r.median(),r.min(),r.max(),a.median(),a.min(),a.max()))
    P(pd.DataFrame(rows,columns=['case','PSO orig100','PSO retuned med','ret min','ret max','agents med','ag min','ag max']).round(4).to_string(index=False))
P('\n== Recovered counts (S1+S2 cases)')
P('PSO orig@100Hz:',int(orig.Recovered.sum()),'/',len(orig))
P('PSO retuned (5 runs):',int(rt.Recovered.sum()),'/',len(rt),' diverged',int(rt.Diverged.sum()))
P('BC+TD3 unbiased (5 agents):',int(ag.Recovered.sum()),'/',len(ag),' diverged',int(ag.Diverged.sum()))
raw=agall[agall.Variant=='raw']
P('BC+TD3 raw nominal (5 agents): recovered',int(raw.Recovered.sum()),'/',len(raw),' IAEphi per agent',raw.IAEphi.round(2).tolist())
# per-case: retuned run vs agent median on IAEphi
w=0;n=0
for c in cases:
    a=ag[ag.Case2==c].IAEphi.median()
    for _,r in rt[rt.Case2==c].iterrows(): n+=1; w+= r.IAEphi<a
P('retuned-run/case pairs with IAEphi below agent median:',w,'of',n)
open(R+'E3b_final_analysis.txt','w').write('\n'.join(out)); print('\n'.join(out))
