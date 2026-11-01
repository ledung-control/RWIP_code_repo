"""Clipping frequencies and metrics in the two impulse scenarios (Table 8 protocol: theta0=0.10 rad,
velocity impulse on theta_dot at t=5 s, 60 s). Python re-implementation; validated against the
stored MATLAB rows for BC+TD3 (audit/audit_5seed.py n1_*/n2_*). BC-only rows are NEW Python values."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv, numpy as np, actorx, phlib as P
ROOT = sys.argv[1] if len(sys.argv) > 1 else '..'
RD = os.path.join(ROOT, 'rl_v13_repro')
S = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'surf201.npy'))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
rows = []
base = {}
for amp in (1.0, 2.0):
    r = P.run(S, 60.0, dist_t=5.0, dist_amp=amp)
    base[amp] = r
    rows.append(['pso_flc', 0, 'baseline', amp, P.iae(r['theta'], r['t'], 60), P.iae(r['phi'], r['t'], 60),
                 0.0, 0, 0, r['tot_clip'], 60000, 0.0])
for k, f in (('td3', 'agent_bc_seed%d.mat'), ('bco', 'agent_bconly_seed%d.mat')):
    for s in range(1, 6):
        net = actorx.load_actor(os.path.join(RD, f % s))
        pi = lambda o, n=net: actorx.pi(n, o)
        ab = actorx.pi(net, np.zeros(6))
        for v in ('raw', 'unbias'):
            for amp in (1.0, 2.0):
                r = P.run(S, 60.0, pi=pi, a_bias=(ab if v == 'unbias' else 0.0), dist_t=5.0, dist_amp=amp)
                rows.append([k, s, v, amp, P.iae(r['theta'], r['t'], 60), P.iae(r['phi'], r['t'], 60),
                             r['rho'], r['res_clip'], r['n_agent'], r['tot_clip'], r['n_ms'], r['amax_res']])
with open(os.path.join(OUT, 'disturbance_clipping_and_metrics.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['policy', 'seed', 'variant', 'impulse_rad_s', 'IAEtheta_60', 'IAEphi_60', 'rho', 'residual_clip_steps',
                'agent_steps', 'total_clip_ms', 'ms_steps', 'max_abs_residual_pre_clip'])
    w.writerows(rows)
print('ok')
