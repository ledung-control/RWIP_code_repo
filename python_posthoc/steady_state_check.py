"""For every raw (and unbias) policy: at the 600-s end state (theta=thetadot=phidot=0 to ~1e-17), the closed-form
steady-state relation  Ku*FLC(-Ne*thref,0) + 10*a_ss = 0,  phi_ss = -thref/Kp_phi  predicts phi_ss from the measured
steady action a_ss.  Compares with the simulated phi(600).  Excludes the non-converged BC-only seed 3 (raw)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv, numpy as np, actorx, rwipsim as R, phlib as P
ROOT = sys.argv[1] if len(sys.argv) > 1 else '..'
RD = os.path.join(ROOT, 'rl_v13_repro')
S = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'surf201.npy'))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
rows = []
for k, f in (('td3', 'agent_bc_seed%d.mat'), ('bco', 'agent_bconly_seed%d.mat')):
    for s in range(1, 6):
        net = actorx.load_actor(os.path.join(RD, f % s))
        pi = lambda o, n=net: actorx.pi(n, o)
        for v in ('raw', 'unbias'):
            ab = actorx.pi(net, np.zeros(6)) if v == 'unbias' else 0.0
            r = P.run(S, 600.0, pi=pi, a_bias=ab)
            a_ss = float(r['a_res'][-1])
            phi_end = float(r['phi'][-1])
            lo, hi = -R.THETA_REF_MAX, R.THETA_REF_MAX
            g = lambda tr: R.KU * P._interp(S, max(-1, min(1, R.KE * (0 - tr))), 0.0) + R.UCORR_MAX * a_ss
            if g(lo) * g(hi) > 0:
                pred = float('nan')
            else:
                for _ in range(90):
                    mid = 0.5 * (lo + hi)
                    if g(lo) * g(mid) <= 0: hi = mid
                    else: lo = mid
                pred = -(0.5 * (lo + hi)) / R.KP_PHI
            rows.append([k, s, v, a_ss, phi_end, pred, abs(pred - phi_end), float(abs(r['theta'][-1]))])
with open(os.path.join(OUT, 'steady_state_check.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['policy', 'seed', 'variant', 'a_ss_residual_action', 'phi_end_600s', 'phi_ss_predicted',
                'abs_error', 'abs_theta_end'])
    w.writerows(rows)
print('ok')
