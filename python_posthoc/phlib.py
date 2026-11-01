"""phlib.py -- instrumented re-run of the loop in rwipsim.py (Round 2 post-hoc analysis).

Same equations, constants and ordering as rwipsim.py (which follows sim_rwip_v13.m); it only
(i) uses plain floats for speed and (ii) records clipping events and the actor output.

Action order (identical to rwipsim.simulate / sim_rwip_v13.m):
    a_raw = pi(obs)                       (tanh output, |a_raw| <= 1)
    a_res = a_raw - aBias                 (post-hoc unbias; aBias = 0 for the raw policy)
    a_c   = clip(a_res, -1, 1)            (residual clip)          -> event 'res_clip'
    u_rl  = 10 V * a_c  (held 10 ms, ZOH)
    V     = clip(V_flc + u_rl, -24, 24)   (total clip, every 1 ms) -> event 'tot_clip'
"""
import math
import numpy as np
import rwipsim as R

TS = 1e-3
EVERY = 10
N_LUT = R.__dict__.get('N_LUT', 201)


def _interp(S, a, b):
    n = 201
    h = 2.0 / (n - 1)
    xa = (a + 1.0) / h
    xb = (b + 1.0) / h
    i = int(xa)
    j = int(xb)
    if i > n - 2:
        i = n - 2
    if j > n - 2:
        j = n - 2
    if i < 0:
        i = 0
    if j < 0:
        j = 0
    fa = xa - i
    fb = xb - j
    return (S[i, j] * (1 - fa) * (1 - fb) + S[i + 1, j] * fa * (1 - fb)
            + S[i, j + 1] * (1 - fa) * fb + S[i + 1, j + 1] * fa * fb)


def _der(th, thd, phd, V):
    tau = (R.KT / R.RA) * (V - R.KEM * phd)
    thdd = (-tau + R.BR * phd + R.MGL * math.sin(th) - R.BP * thd) / R.IP_EQ
    phdd = (tau - R.BR * phd) / R.IR - thdd
    return thd, thdd, phd, phdd


def flc_voltage(S, th, thd, ph, phd):
    tref = -(R.KP_PHI * ph + R.KV_PHI * phd)
    tref = max(-R.THETA_REF_MAX, min(R.THETA_REF_MAX, tref))
    i1 = max(-1.0, min(1.0, R.KE * (th - tref)))
    i2 = max(-1.0, min(1.0, R.KDE * thd))
    return R.KU * _interp(S, i1, i2) + R.KD_TH * thd + R.KD_PH * phd, tref


def run(S, Tf=60.0, IC=(0.10, 0.0, 0.0, 0.0), pi=None, a_bias=0.0, b_inject=None,
        keep_traj=True, dist_t=None, dist_amp=0.0):
    """pi: callable obs(6,)->a. b_inject: constant action (replaces pi). Returns dict."""
    N = int(round(Tf / TS))
    th, thd, ph, phd = IC
    TH = np.empty(N + 1)
    PH = np.empty(N + 1)
    UC = np.empty(N + 1)
    TH[0], PH[0] = th, ph
    u_rl = 0.0
    res_clip = tot_clip = n_agent = 0
    amax_res = 0.0
    use = (pi is not None) or (b_inject is not None)
    A_raw = []
    A_res = []
    ub_max = 0.0
    uc_max = 0.0
    done_dist = dist_t is None
    for n in range(1, N + 1):
        Vf, tref = flc_voltage(S, th, thd, ph, phd)
        V = Vf
        if use:
            if (n - 1) % EVERY == 0:
                n_agent += 1
                if pi is not None:
                    obs = np.array([th / R.S_TH, thd / R.S_THD, ph / R.S_PH,
                                    phd / R.S_PHD, tref / R.S_TREF, Vf / R.S_U])
                    a_raw = pi(obs)
                    a = a_raw - a_bias
                else:
                    a_raw = b_inject
                    a = b_inject
                if keep_traj:
                    A_raw.append(a_raw)
                    A_res.append(a)
                if abs(a) > 1.0:
                    res_clip += 1
                amax_res = max(amax_res, abs(a))
                u_rl = R.UCORR_MAX * max(-1.0, min(1.0, a))
            V = Vf + u_rl
        UC[n - 1] = u_rl
        if abs(V) > 24.0:
            tot_clip += 1
            V = 24.0 if V > 0 else -24.0
        ub_max = max(ub_max, abs(V - u_rl))
        uc_max = max(uc_max, abs(u_rl))
        # RK4
        k1 = _der(th, thd, phd, V)
        k2 = _der(th + TS / 2 * k1[0], thd + TS / 2 * k1[1], phd + TS / 2 * k1[3], V)
        k3 = _der(th + TS / 2 * k2[0], thd + TS / 2 * k2[1], phd + TS / 2 * k2[3], V)
        k4 = _der(th + TS * k3[0], thd + TS * k3[1], phd + TS * k3[3], V)
        th += TS / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        thd += TS / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        ph += TS / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        phd += TS / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        if not (math.isfinite(th) and math.isfinite(phd)) or abs(th) > 1.5:
            return dict(diverged=True)
        if not done_dist and n * TS >= dist_t:
            thd += dist_amp
            done_dist = True
        TH[n], PH[n] = th, ph
    UC[-1] = UC[-2]
    t = np.arange(N + 1) * TS
    out = dict(diverged=False, t=t, theta=TH, phi=PH, u_corr=UC,
               state_end=(th, thd, ph, phd), n_agent=n_agent, n_ms=N,
               res_clip=res_clip, tot_clip=tot_clip, amax_res=amax_res,
               rho=(uc_max / max(ub_max, 1e-12)) if use else 0.0,
               a_raw=np.array(A_raw), a_res=np.array(A_res))
    return out


def iae(x, t, T):
    k = int(round(T / TS))
    return float(np.trapezoid(np.abs(x[:k + 1]), t[:k + 1]))


def step10(S, x, pi, a_bias):
    """One agent period (10 ms, 10 RK4 substeps of 1 ms) of the closed loop. x=(th,thd,ph,phd)."""
    th, thd, ph, phd = x
    Vf, tref = flc_voltage(S, th, thd, ph, phd)
    obs = np.array([th / R.S_TH, thd / R.S_THD, ph / R.S_PH, phd / R.S_PHD,
                    tref / R.S_TREF, Vf / R.S_U])
    a = max(-1.0, min(1.0, pi(obs) - a_bias))
    u_rl = R.UCORR_MAX * a
    for _ in range(EVERY):
        Vf, _t = flc_voltage(S, th, thd, ph, phd)
        V = max(-24.0, min(24.0, Vf + u_rl))
        k1 = _der(th, thd, phd, V)
        k2 = _der(th + TS / 2 * k1[0], thd + TS / 2 * k1[1], phd + TS / 2 * k1[3], V)
        k3 = _der(th + TS / 2 * k2[0], thd + TS / 2 * k2[1], phd + TS / 2 * k2[3], V)
        k4 = _der(th + TS * k3[0], thd + TS * k3[1], phd + TS * k3[3], V)
        th += TS / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        thd += TS / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        ph += TS / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        phd += TS / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
    return np.array([th, thd, ph, phd])
