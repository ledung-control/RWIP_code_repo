import pandas as pd, numpy as np
R='results/'
e=pd.read_csv(R+'E6_eval/E6_eval_MATLAB_2026a.csv')
e3=pd.read_csv(R+'E3_MATLAB_2026a.csv'); e3=e3[e3.Rate=='all100']
e['C']=e.Case+np.where(e.Case=='kick',' '+e.Kick.astype(str),'')
e3['C']=e3.Case+np.where(e3.Case=='kick',' '+e3.Kick.astype(str),'')
out=[]; P=lambda *a: out.append(' '.join(str(x) for x in a))
P('E6: end-to-end TD3 (3 seeds, 800 episodes) vs PSO-FLC and BC+TD3 (unbiased) at 100 Hz, S1 (8 cases, 60 s) + S2 (kicks 1, 2 rad/s)')
P(e.pivot_table(index='C',columns='Controller',values='IAEphi',aggfunc='first').round(2).to_string())
P('\nIAEtheta'); P(e.pivot_table(index='C',columns='Controller',values='IAEtheta',aggfunc='first').round(4).to_string())
P('\nRecovered per seed (of 10):'); P(e.groupby('Controller').Recovered.sum().to_string())
P('Diverged per seed (of 10):'); P(e.groupby('Controller').Diverged.sum().to_string())
P('thEnd/phEnd medians per seed:'); P(e.groupby('Controller')[['thEnd','phEnd']].median().round(4).to_string())
pf=e3[e3.Controller=='PSO-FLC']; ag=e3[(e3.Controller.str.startswith('BC+TD3'))&(e3.Variant=='unbiased')]
P('\nReference at 100 Hz: PSO-FLC recovered',int(pf.Recovered.sum()),'/',len(pf),'; BC+TD3 unbiased',int(ag.Recovered.sum()),'/',len(ag))
P('PSO-FLC IAEphi median over cases',round(pf.IAEphi.median(),3),'IAEtheta',round(pf.IAEtheta.median(),4),'; BC+TD3 IAEphi',round(ag.IAEphi.median(),3),'IAEtheta',round(ag.IAEtheta.median(),4))
open(R+'E6_final_analysis.txt','w').write('\n'.join(out)); print('\n'.join(out))
