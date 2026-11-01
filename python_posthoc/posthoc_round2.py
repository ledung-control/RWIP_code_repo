"""posthoc_round2.py -- post-hoc analyses of the STORED actor checkpoints (no retraining).

Inputs  : <ROOT>/rl_v13_repro/agent_bc_seed{1..5}.mat   (BC + TD3, raw)
          <ROOT>/rl_v13_repro/agent_bconly_seed{1..5}.mat (BC-only)
          <ROOT>/fis_rwip_v13_best.fis  (only via rwipsim.py constants / surf201.npy cache)
Outputs : results/*.csv, results/summary.json
Usage   : python3 -I posthoc_round2.py [ROOT]      (ROOT default: ..)
Nothing here is a retraining experiment; every number is a deterministic forward evaluation of a
stored network in the independent Python re-implementation (rwipsim.py / phlib.py).
"""
import sys, os, json, csv, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import actorx, rwipsim as R, phlib as P

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RD = os.path.join(ROOT, 'rl_v13_repro')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUT, exist_ok=True)
SURF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'surf201.npy')
if os.path.exists(SURF):
    S = np.load(SURF)
else:
    _, S = R.build_surface(201)
    np.save(SURF, S)

SEEDS = [1, 2, 3, 4, 5]
summ = {}


def wcsv(name, header, rows):
    with open(os.path.join(OUT, name), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def load(kind, s):
    f = 'agent_bc_seed%d.mat' if kind == 'td3' else 'agent_bconly_seed%d.mat'
    return actorx.load_actor(os.path.join(RD, f % s))


nets = {(k, s): load(k, s) for k in ('td3', 'bco') for s in SEEDS}
abias = {key: actorx.pi(n, np.zeros(6)) for key, n in nets.items()}
pis = {key: (lambda o, n=n: actorx.pi(n, o)) for key, n in nets.items()}

# ---------------------------------------------------------------- 0. PSO-FLC reference (RK4, 1 ms)
base = P.run(S, 600.0)
t, th, ph = base['t'], base['theta'], base['phi']
d60 = P.run(S, 60.0)


def settle(x, t, tol):
    idx = np.where(np.abs(x) > tol)[0]
    return float(t[idx[-1]]) if len(idx) else 0.0


m60 = dict(IAEtheta=P.iae(th, t, 60), IAEphi=P.iae(ph, t, 60),
           OS=float(max(0.0, -np.min(th[:60001]))),
           ts_theta=settle(th[:60001], t[:60001], 0.002), ts_phi=settle(ph[:60001], t[:60001], 0.020),
           phi_peak=float(np.max(np.abs(ph[:60001]))),
           IAEphi_600=P.iae(ph, t, 600), tot_clip_frac_60=base['tot_clip'] / 60000.0)
summ['pso_flc_rk4_1ms'] = m60

# ---------------------------------------------------------------- 1. per-policy runs (600 s)
rows = []
runs = {}
for (k, s), pi in pis.items():
    for variant in ('raw', 'unbias'):
        ab = abias[(k, s)] if variant == 'unbias' else 0.0
        r = P.run(S, 600.0, pi=pi, a_bias=ab)
        runs[(k, s, variant)] = r
        tt = r['t']
        i60 = 60000
        n60 = 6000  # agent steps in 60 s
        rows.append([k, s, variant, abias[(k, s)],
                     P.iae(r['theta'], tt, 60), P.iae(r['phi'], tt, 60),
                     P.iae(r['theta'], tt, 600), P.iae(r['phi'], tt, 600),
                     r['phi'][i60], r['phi'][-1], r['theta'][-1],
                     r['rho'],
                     int(np.sum(np.abs(r['a_res'][:n60]) > 1.0)), n60,
                     int(np.sum(np.abs(r['a_res']) > 1.0)), r['n_agent'], r['amax_res'],
                     r['tot_clip'], r['n_ms']])
wcsv('runs_600s.csv',
     ['policy', 'seed', 'variant', 'aBias', 'IAEtheta_60', 'IAEphi_60', 'IAEtheta_600', 'IAEphi_600',
      'phi_60', 'phi_600', 'theta_600', 'rho_600', 'res_clip_steps_60s', 'agent_steps_60s',
      'res_clip_steps_600s', 'agent_steps_600s', 'max_abs_residual_pre_clip', 'tot_clip_ms_600s',
      'ms_steps_600s'], rows)

# ---------------------------------------------------------------- 2. clipping table (60 s, nominal)
clip_rows = []
for (k, s, v), r in runs.items():
    n60 = 6000
    res = int(np.sum(np.abs(r['a_res'][:n60]) > 1.0))
    # total-voltage clipping within the first 60 s: re-run 60 s to count in that window only
    r60 = P.run(S, 60.0, pi=pis[(k, s)], a_bias=abias[(k, s)] if v == 'unbias' else 0.0)
    clip_rows.append([k, s, v, res, n60, 100.0 * res / n60, r60['tot_clip'], 60000,
                      100.0 * r60['tot_clip'] / 60000.0, r60['amax_res']])
wcsv('clipping_60s.csv', ['policy', 'seed', 'variant', 'residual_clip_steps', 'agent_steps',
                         'residual_clip_pct', 'total_clip_ms', 'ms_steps', 'total_clip_pct',
                         'max_abs_residual_pre_clip'], clip_rows)
summ['clipping_pso_flc_alone_total_clip_ms_60s'] = int(d60['tot_clip'])
summ['clipping_max_residual_pre_clip'] = max(r[9] for r in clip_rows)
summ['clipping_any_residual_clip'] = int(sum(r[3] for r in clip_rows))
summ['clipping_total_clip_ms_by_policy'] = {f'{r[0]}_s{r[1]}_{r[2]}': r[6] for r in clip_rows}

# ---------------------------------------------------------------- 3. parameter displacement BC-only -> BC+TD3
prow = []
for s in SEEDS:
    a, b = nets[('bco', s)], nets[('td3', s)]
    for nm in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
        d = (b[nm] - a[nm]).ravel()
        prow.append([s, nm, a[nm].size, float(np.linalg.norm(d)), float(np.linalg.norm(a[nm].ravel())),
                     float(np.max(np.abs(d)))])
    dall = np.concatenate([(b[n] - a[n]).ravel() for n in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')])
    wall = np.concatenate([a[n].ravel() for n in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')])
    prow.append([s, 'ALL', dall.size, float(np.linalg.norm(dall)), float(np.linalg.norm(wall)),
                 float(np.max(np.abs(dall)))])
wcsv('param_displacement.csv', ['seed', 'tensor', 'n_params', 'L2_change', 'L2_norm_BCinit',
                               'Linf_change'], prow)
allrows = [r for r in prow if r[1] == 'ALL']
summ['param_change_ALL'] = {f's{r[0]}': dict(L2=r[3], rel=r[3] / r[4], Linf=r[5]) for r in allrows}

# ---------------------------------------------------------------- 4. pi(s)-pi(0): neighbourhood + trajectory + Jacobian
rng = np.random.default_rng(20261025)
nb_rows = []
EPS = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
for (k, s), net in nets.items():
    p0 = abias[(k, s)]
    for eps in EPS:
        O = rng.uniform(-eps, eps, size=(4000, 6))
        d = np.array([actorx.pi(net, o) - p0 for o in O])
        nb_rows.append([k, s, eps, p0, float(d.mean()), float(d.std()), float(np.max(np.abs(d))),
                        float(np.percentile(np.abs(d), 99))])
wcsv('pi_minus_pi0_neighbourhood.csv',
     ['policy', 'seed', 'box_halfwidth_normalized_obs', 'pi0', 'mean_dev', 'std_dev', 'max_abs_dev',
      'p99_abs_dev'], nb_rows)
# Jacobian of pi at 0 (central differences, normalized obs units)
jac_rows = []
for (k, s), net in nets.items():
    h = 1e-4
    g = []
    for i in range(6):
        e = np.zeros(6)
        e[i] = h
        g.append((actorx.pi(net, e) - actorx.pi(net, -e)) / (2 * h))
    jac_rows.append([k, s, abias[(k, s)]] + g + [float(np.linalg.norm(g))])
wcsv('pi_jacobian_at_origin.csv', ['policy', 'seed', 'pi0', 'd/dtheta', 'd/dthetadot', 'd/dphi', 'd/dphidot',
                                  'd/dthetaref', 'd/duFLC', 'norm'], jac_rows)
# along the nominal closed-loop trajectory (raw policy): deviation pi(s_t)-pi(0) per window
tr_rows = []
for s in SEEDS:
    for k in ('td3', 'bco'):
        r = runs[(k, s, 'raw')]
        dev = r['a_raw'] - abias[(k, s)]          # one value per agent step (10 ms)
        for (a0, a1) in [(0, 5), (5, 20), (20, 60), (60, 600)]:
            seg = dev[int(a0 / 0.01):int(a1 / 0.01)]
            tr_rows.append([k, s, a0, a1, abias[(k, s)], float(seg.mean()), float(seg.std()),
                            float(seg.min()), float(seg.max())])
wcsv('pi_minus_pi0_trajectory.csv',
     ['policy', 'seed', 't_from_s', 't_to_s', 'pi0', 'mean_dev', 'std_dev', 'min_dev', 'max_dev'], tr_rows)

# ---------------------------------------------------------------- 5. local closed loop: fixed point + Jacobian of 10-ms map
loc_rows = []
for (k, s) in nets:
    for variant in ('raw', 'unbias'):
        ab = abias[(k, s)] if variant == 'unbias' else 0.0
        r = runs[(k, s, variant)]
        xs = np.array(r['state_end'])
        # fixed-point residual of the 10 ms map at the 600 s end state
        xn = P.step10(S, xs, pis[(k, s)], ab)
        res = float(np.max(np.abs(xn - xs)))
        J = np.zeros((4, 4))
        for i in range(4):
            e = np.zeros(4)
            e[i] = 1e-6
            J[:, i] = (P.step10(S, xs + e, pis[(k, s)], ab) - P.step10(S, xs - e, pis[(k, s)], ab)) / 2e-6
        ev = np.linalg.eigvals(J)
        loc_rows.append([k, s, variant, float(xs[0]), float(xs[2]), res, float(np.max(np.abs(ev))),
                         ' '.join('%.5f%+.5fj' % (z.real, z.imag) for z in ev)])
wcsv('local_closed_loop.csv', ['policy', 'seed', 'variant', 'theta_end', 'phi_end',
                              'fixed_point_residual_10ms', 'spectral_radius_10ms_map', 'eigenvalues'], loc_rows)
# small-amplitude responses (unbias): theta0 sweep
sm_rows = []
for s in SEEDS:
    for th0 in (0.002, 0.01, 0.05, 0.10):
        r = P.run(S, 120.0, IC=(th0, 0, 0, 0), pi=pis[('td3', s)], a_bias=abias[('td3', s)])
        sm_rows.append([s, th0, P.iae(r['phi'], r['t'], 120) / th0, r['phi'][-1], r['theta'][-1]])
wcsv('unbias_small_amplitude.csv', ['seed', 'theta0', 'IAEphi_120s_over_theta0', 'phi_120', 'theta_120'], sm_rows)

# ---------------------------------------------------------------- 6. horizon ratio (Table 11) per seed + intercept
h_rows = []
for (k, s, v), r in runs.items():
    tt, p = r['t'], r['phi']
    I = np.array([P.iae(p, tt, T) for T in (60, 100, 200, 300, 400, 500, 600)])
    Tg = np.array([60, 100, 200, 300, 400, 500, 600.0])
    sl, ic = np.polyfit(Tg[2:], I[2:], 1)
    h_rows.append([k, s, v, I[0], I[-1], I[-1] / I[0] if I[0] > 0 else float('nan'), float(sl), float(ic),
                   float(abs(p[-1]))])
wcsv('horizon_ratio_per_seed.csv', ['policy', 'seed', 'variant', 'IAEphi_60', 'IAEphi_600', 'ratio_600_over_60',
                                   'slope_T200_600', 'intercept_T200_600', 'abs_phi_600'], h_rows)
for k in ('td3', 'bco'):
    for v in ('raw', 'unbias'):
        a60 = [r[3] for r in h_rows if r[0] == k and r[2] == v]
        a600 = [r[4] for r in h_rows if r[0] == k and r[2] == v]
        summ[f'table11_{k}_{v}'] = dict(med_60=float(np.median(a60)), med_600=float(np.median(a600)),
                                       ratio_of_medians=float(np.median(a600) / np.median(a60)),
                                       per_seed_ratio=[r[5] for r in h_rows if r[0] == k and r[2] == v],
                                       per_seed_intercept=[r[7] for r in h_rows if r[0] == k and r[2] == v])

# ---------------------------------------------------------------- 7. constant-bias injection sweep + steady-state theory
bs = [0.0, 0.02, -0.02, 0.05, -0.05, 0.10, -0.10, 0.20, -0.20, 0.30, -0.30]
inj = []
for b in bs:
    r = P.run(S, 600.0, b_inject=b)
    tt, p = r['t'], r['phi']
    inj.append([b, P.iae(p, tt, 60), P.iae(p, tt, 600),
                P.iae(p, tt, 600) / P.iae(p, tt, 60) if P.iae(p, tt, 60) > 0 else float('nan'),
                p[60000], p[-1], abs(p[-1] - p[-60001]) if len(p) > 60001 else float('nan'),
                r['state_end'][1], r['state_end'][3]])
wcsv('bias_injection_sweep.csv', ['b', 'IAEphi_60', 'IAEphi_600', 'ratio', 'phi_60', 'phi_600',
                                 'abs_phi600_minus_phi540', 'thetadot_600', 'phidot_600'], inj)
x = np.array([abs(r[0]) for r in inj])
y = np.array([r[1] for r in inj])
summ['injection_pearson_r_absb_vs_IAEphi60_11pts'] = float(np.corrcoef(x, y)[0, 1])
summ['injection_pearson_r_b_vs_IAEphi60_11pts'] = float(np.corrcoef([r[0] for r in inj], y)[0, 1])
rat = [r[3] for r in inj if r[0] != 0]
summ['injection_ratio_range'] = [float(min(rat)), float(max(rat))]
# steady state: theta=0, phidot=0 -> V_total=0 -> Ku*FLC(-Ne*thref*,0) + 10*b = 0 with thref* in [-0.035, 0.035]
from phlib import _interp
v_at_sat = R.KU * _interp(S, max(-1, min(1, R.KE * 0.035)), 0.0)  # FLC at e=+Ne*0.035 (theta=0, thref=-0.035)
summ['steady_state'] = dict(theta_ref_sat=R.THETA_REF_MAX,
                            phi_scale_0p035_over_Kp=R.THETA_REF_MAX / R.KP_PHI,
                            e_at_sat=R.KE * R.THETA_REF_MAX, Ku_FLC_at_sat_V=float(v_at_sat),
                            b_max_action=float(v_at_sat / R.UCORR_MAX))
# solve phi_ss for each |b| < b_max by bisection on thref* (monotone map)
ss = []
for b in bs:
    if b == 0:
        continue
    lo, hi = -R.THETA_REF_MAX, R.THETA_REF_MAX
    f = lambda tr: R.KU * _interp(S, max(-1, min(1, R.KE * (0.0 - tr))), 0.0) + R.UCORR_MAX * b
    if f(lo) * f(hi) > 0:
        ss.append([b, 'no steady state with |thetaref|<0.035', float('nan'), float('nan')])
        continue
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    tr = 0.5 * (lo + hi)
    ss.append([b, 'exists', tr, -tr / R.KP_PHI])
wcsv('bias_steady_state_theory.csv', ['b', 'status', 'thetaref_ss', 'phi_ss_predicted'], ss)

# ---------------------------------------------------------------- 8. paired statistics
def perm_p_pearson(xv, yv):
    r0 = np.corrcoef(xv, yv)[0, 1]
    cnt = 0
    tot = 0
    for pm in itertools.permutations(range(len(yv))):
        r = np.corrcoef(xv, np.array(yv)[list(pm)])[0, 1]
        tot += 1
        if abs(r) >= abs(r0) - 1e-12:
            cnt += 1
    return r0, cnt, tot


def spearman(xv, yv):
    rx = np.argsort(np.argsort(xv))
    ry = np.argsort(np.argsort(yv))
    return float(np.corrcoef(rx, ry)[0, 1])


td3raw = {s: runs[('td3', s, 'raw')] for s in SEEDS}
ab_td3 = np.array([abs(abias[('td3', s)]) for s in SEEDS])
iaephi_raw = np.array([P.iae(td3raw[s]['phi'], td3raw[s]['t'], 60) for s in SEEDS])
r0, cnt, tot = perm_p_pearson(ab_td3, iaephi_raw)
z = np.arctanh(r0)
se = 1 / np.sqrt(len(SEEDS) - 3)
st = dict(pearson_abs=float(r0), perm_two_sided_count=cnt, perm_total=tot, perm_p=cnt / tot,
          spearman=spearman(ab_td3, iaephi_raw),
          fisher_ci95=[float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))],
          signed_pearson=float(np.corrcoef([abias[('td3', s)] for s in SEEDS], iaephi_raw)[0, 1]),
          loo=[float(np.corrcoef(np.delete(ab_td3, i), np.delete(iaephi_raw, i))[0, 1]) for i in range(5)])
rng2 = np.random.default_rng(1)
bs_r = []
for _ in range(20000):
    idx = rng2.integers(0, 5, 5)
    if len(set(idx)) < 3 or np.std(ab_td3[idx]) == 0 or np.std(iaephi_raw[idx]) == 0:
        continue
    bs_r.append(np.corrcoef(ab_td3[idx], iaephi_raw[idx])[0, 1])
st['bootstrap_pct_2p5_97p5'] = [float(np.percentile(bs_r, 2.5)), float(np.percentile(bs_r, 97.5))]
summ['stats_corr'] = st

# paired per-seed changes
def get(k, s, v, key):
    return [r for r in rows if r[0] == k and r[1] == s and r[2] == v][0]


pair_rows = []
for s in SEEDS:
    for (lab, ka, va, kb, vb) in [('unbias_effect_TD3', 'td3', 'raw', 'td3', 'unbias'),
                                  ('unbias_effect_BCO', 'bco', 'raw', 'bco', 'unbias'),
                                  ('TD3_minus_BCO_raw', 'bco', 'raw', 'td3', 'raw'),
                                  ('TD3_minus_BCO_unbias', 'bco', 'unbias', 'td3', 'unbias')]:
        A, B = get(ka, s, va, None), get(kb, s, vb, None)
        pair_rows.append([lab, s, A[5], B[5], B[5] - A[5], A[4], B[4], B[4] - A[4], A[11], B[11], B[11] - A[11]])
wcsv('paired_changes.csv', ['contrast', 'seed', 'IAEphi60_from', 'IAEphi60_to', 'dIAEphi', 'IAEtheta60_from',
                           'IAEtheta60_to', 'dIAEtheta', 'rho_from', 'rho_to', 'drho'], pair_rows)
# exact sign-flip (paired permutation) tests on the paired differences
sf = {}
for lab in ('unbias_effect_TD3', 'unbias_effect_BCO', 'TD3_minus_BCO_raw', 'TD3_minus_BCO_unbias'):
    for nm, col in (('IAEphi', 4), ('IAEtheta', 7)):
        d = np.array([r[col] for r in pair_rows if r[0] == lab])
        obs = abs(d.sum())
        c = 0
        for sg in itertools.product([1, -1], repeat=len(d)):
            if abs((np.array(sg) * d).sum()) >= obs - 1e-15:
                c += 1
        sf[f'{lab}:{nm}'] = dict(diffs=d.tolist(), n_negative=int((d < 0).sum()), n=len(d),
                                 median=float(np.median(d)), exact_signflip_two_sided_p=c / 2 ** len(d))
summ['paired_signflip'] = sf
d_ab = np.array([abs(abias[('td3', s)]) - abs(abias[('bco', s)]) for s in SEEDS])
summ['abias_td3_minus_bco'] = dict(diffs=d_ab.tolist(), n_positive=int((d_ab > 0).sum()),
                                  exact_two_sided_signflip_p=sum(1 for sg in itertools.product([1, -1], repeat=5)
                                                                if abs((np.array(sg) * d_ab).sum()) >= abs(d_ab.sum()) - 1e-15) / 32)
summ['abias'] = {f'{k}_s{s}': float(v) for (k, s), v in abias.items()}
json.dump(summ, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1, default=float)
print('done')
