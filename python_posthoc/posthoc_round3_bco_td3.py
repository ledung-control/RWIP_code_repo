"""posthoc_round3_bco_td3.py -- paired BC+TD3 minus BC-only, per seed (post-hoc; no retraining).

Inputs : results/runs_600s.csv (nominal 60 s metrics, written by posthoc_round2.py),
         results/disturbance_clipping_and_metrics.csv (impulse 1.0 / 2.0 rad/s, 60 s; posthoc_disturbance_clip.py)
         Both come from the stored checkpoints agent_bc_seed{k}.mat (BC+TD3) and agent_bconly_seed{k}.mat (BC-only),
         evaluated in the Python re-implementation (phlib.py); 'unbias' = pi(o) - pi(0) with each network's own pi(0).
Unit   : seed k pairs the BC-only and BC+TD3 networks that share the same seed and BC initialization (n = 5).
Stats  : d_k = TD3_k - BCO_k.  Exact two-sided sign-flip test on d (2^5 = 32 sign patterns; min attainable p = 0.0625).
         Hodges-Lehmann estimate = median of the 15 Walsh averages (d_i + d_j)/2, i <= j.
         Interval [min d, max d] = exact Wilcoxon/Walsh interval with confidence 1 - 2/32 = 93.75 % (assumes symmetric
         distribution of d); with n = 5 no narrower exact interval exists.
Output : results/bco_td3_paired.csv
"""
import itertools, csv, os
import numpy as np, pandas as pd
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
a = pd.read_csv(os.path.join(R, 'runs_600s.csv'))
b = pd.read_csv(os.path.join(R, 'disturbance_clipping_and_metrics.csv'))
rows = []
for var in ('raw', 'unbias'):
    for scen in (0.0, 1.0, 2.0):
        for metric in ('IAEtheta', 'IAEphi', 'rho'):
            val = {}
            for pol in ('td3', 'bco'):
                if scen == 0.0:
                    q = a[(a.policy == pol) & (a.variant == var)].sort_values('seed')
                    col = {'IAEtheta': 'IAEtheta_60', 'IAEphi': 'IAEphi_60', 'rho': 'rho_600'}[metric]
                    if metric == 'rho':      # rho at 60 s, not 600 s: take from the disturbance file's nominal? not stored -> skip
                        val[pol] = None
                        continue
                else:
                    q = b[(b.policy == pol) & (b.variant == var) & (b.impulse_rad_s == scen)].sort_values('seed')
                    col = {'IAEtheta': 'IAEtheta_60', 'IAEphi': 'IAEphi_60', 'rho': 'rho'}[metric]
                val[pol] = q[col].values
            if val['td3'] is None or val['bco'] is None:
                continue
            d = val['td3'] - val['bco']
            n = len(d)
            absd = np.abs(d)
            pts = [np.sum(np.array(s) * absd) for s in itertools.product([-1, 1], repeat=n)]
            p = float(np.mean([abs(x) >= abs(d.sum()) - 1e-12 for x in pts]))
            walsh = [(d[i] + d[j]) / 2 for i in range(n) for j in range(i, n)]
            rows.append(dict(scenario=('nominal' if scen == 0 else '%.1f rad/s impulse' % scen), variant=var, metric=metric,
                             bco_median=float(np.median(val['bco'])), td3_median=float(np.median(val['td3'])),
                             d_per_seed=' '.join('%.5g' % x for x in d), n_td3_higher=int((d > 0).sum()), n=n,
                             d_median=float(np.median(d)), d_HL=float(np.median(walsh)),
                             interval_lo=float(d.min()), interval_hi=float(d.max()), interval_conf='0.9375',
                             signflip_p=p))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(R, 'bco_td3_paired.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(out.drop(columns=['d_per_seed']).round(5).to_string())
