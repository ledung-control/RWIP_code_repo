"""Cai dat lai doc lap vong lap RWIP + PSO-FLC (+ lop hieu chinh residual).

Bam sat sim_rwip_v13.m / make_P_v13.m / rl_obs_v13.m / fis_rwip_v13_best.fis.
Kiem chung: IAEtheta danh nghia phai khop 0.024463 (Bang 3 cua bai bao).
"""
import numpy as np

# ---------------- tham so doi tuong (make_P_v13.m) ----------------
mp, mr, Ip0, Lp, Lr, g = 0.5297, 0.29225, 11.57e-3, 0.256, 0.385, 9.80665
IP_EQ = Ip0 + mp * Lp**2 + mr * Lr**2
IR = 82.1953125e-3
BP, BR = 1e-3, 5e-4
RA, KEM, KT = 2.0, 0.5, 0.5
MGL = g * (mp * Lp + mr * Lr)
THETA_REF_MAX = 0.035
KE, KDE, KU = 1.93975, 0.678232, 23.4172
KD_TH, KD_PH = 2.91123, 0.190429
KP_PHI, KV_PHI = 0.00437824, 0.0183369
UCORR_MAX = 10.0
# rl_obs_v13.m (LUU Y: S_PH = 10.0 trong ma nguon, bai bao ghi 2.0)
S_TH, S_THD, S_PH, S_PHD, S_TREF, S_U = 0.15, 3.0, 10.0, 20.0, 0.035, 24.0


# ---------------- FIS Mamdani (fis_rwip_v13_best.fis) ----------------
def _trap(x, a, b, c, d):
    x = np.asarray(x, float)
    y = np.zeros_like(x)
    if b > a:
        m = (x > a) & (x < b); y[m] = (x[m] - a) / (b - a)
    m = (x >= b) & (x <= c); y[m] = 1.0
    if d > c:
        m = (x > c) & (x < d); y[m] = (d - x[m]) / (d - c)
    y[(x <= a) | (x >= d)] = 0.0
    m = (x >= b) & (x <= c); y[m] = 1.0
    return y


def _tri(x, a, b, c):
    x = np.asarray(x, float)
    y = np.zeros_like(x)
    if b > a:
        m = (x > a) & (x < b); y[m] = (x[m] - a) / (b - a)
    if c > b:
        m = (x > b) & (x < c); y[m] = (c - x[m]) / (c - b)
    y[x == b] = 1.0
    y[(x <= a) | (x >= c)] = 0.0
    y[x == b] = 1.0
    return y


def _mfset(s):
    """5 MF doi xung: trap, tri, tri, tri, trap voi nua-do-rong vung giua = s."""
    return [lambda x, s=s: _trap(x, -1, -1, -1, -s),
            lambda x, s=s: _tri(x, -1, -s, 0),
            lambda x, s=s: _tri(x, -s, 0, s),
            lambda x, s=s: _tri(x, 0, s, 1),
            lambda x, s=s: _trap(x, s, 1, 1, 1)]


S_E, S_DE, S_U_MF = 0.209689627964486, 0.783640055701184, 0.290554149732575
MF1, MF2, MFO = _mfset(S_E), _mfset(S_DE), _mfset(S_U_MF)
RULES = [(i, j, min(max(i + j - 3, 1), 5) - 1)               # u = clamp(i+j-3,1,5)
         for i in range(1, 6) for j in range(1, 6)]
YOUT = np.linspace(-1, 1, 101)                               # luoi giai mo cua evalfis
MFO_Y = np.array([f(YOUT) for f in MFO])                     # (5,101)


def evalfis(a, b):
    """Suy luan min-max, giai mo trong tam -- mot cap dau vao."""
    ma = np.array([f(np.array([a]))[0] for f in MF1])
    mb = np.array([f(np.array([b]))[0] for f in MF2])
    agg = np.zeros_like(YOUT)
    for i, j, k in RULES:
        w = min(ma[i - 1], mb[j - 1])
        if w > 0:
            agg = np.maximum(agg, np.minimum(w, MFO_Y[k]))
    s = agg.sum()
    return 0.0 if s == 0 else float((YOUT * agg).sum() / s)


def build_surface(n=201):
    g = np.linspace(-1, 1, n)
    S = np.empty((n, n))
    for i, a in enumerate(g):
        for j, b in enumerate(g):
            S[i, j] = evalfis(a, b)
    return g, S


def interp2lin(S, a, b):
    n = S.shape[0]; h = 2.0 / (n - 1)
    xa, xb = (a + 1) / h, (b + 1) / h
    i = min(max(int(np.floor(xa)), 0), n - 2)
    j = min(max(int(np.floor(xb)), 0), n - 2)
    fa, fb = xa - i, xb - j
    return (S[i, j] * (1 - fa) * (1 - fb) + S[i + 1, j] * fa * (1 - fb)
            + S[i, j + 1] * (1 - fa) * fb + S[i + 1, j + 1] * fa * fb)


def deriv(x, V):
    th, thd, phd = x[0], x[1], x[3]
    tau = (KT / RA) * (V - KEM * phd)
    thdd = (-tau + BR * phd + MGL * np.sin(th) - BP * thd) / IP_EQ
    phdd = (tau - BR * phd) / IR - thdd
    return np.array([thd, thdd, x[3], phdd])


def simulate(SURF, Tf=60.0, Ts=1e-3, IC=(0.10, 0, 0, 0),
             policy=None, a_bias=0.0, b_inject=0.0, rlTs=0.01,
             dist_t=None, dist_amp=0.0):
    """policy: ham nhan obs(6,) tra ve a in [-1,1]. b_inject: bias hang so (don vi a)."""
    N = int(round(Tf / Ts))
    every = max(1, int(round(rlTs / Ts)))
    x = np.array(IC, float)
    TH = np.empty(N + 1); PH = np.empty(N + 1); U = np.empty(N + 1); UC = np.empty(N + 1)
    TH[0], PH[0] = x[0], x[2]
    u_rl = 0.0
    done_dist = dist_t is None
    for n in range(1, N + 1):
        th, thd, ph, phd = x
        tref = -(KP_PHI * ph + KV_PHI * phd)
        tref = max(-THETA_REF_MAX, min(THETA_REF_MAX, tref))
        i1 = max(-1.0, min(1.0, KE * (th - tref)))
        i2 = max(-1.0, min(1.0, KDE * thd))
        V = KU * interp2lin(SURF, i1, i2) + KD_TH * thd + KD_PH * phd
        if policy is not None or b_inject:
            if (n - 1) % every == 0:
                if policy is not None:
                    obs = np.array([th / S_TH, thd / S_THD, ph / S_PH,
                                    phd / S_PHD, tref / S_TREF, V / S_U])
                    a = policy(obs) - a_bias
                else:
                    a = b_inject
                u_rl = UCORR_MAX * max(-1.0, min(1.0, a))
            V = V + u_rl
            UC[n - 1] = u_rl
        else:
            UC[n - 1] = 0.0
        V = max(-24.0, min(24.0, V))
        U[n - 1] = V
        k1 = deriv(x, V); k2 = deriv(x + Ts / 2 * k1, V)
        k3 = deriv(x + Ts / 2 * k2, V); k4 = deriv(x + Ts * k3, V)
        x = x + Ts / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if not done_dist and n * Ts >= dist_t:
            x[1] += dist_amp; done_dist = True
        if not np.all(np.isfinite(x)) or abs(x[0]) > 1.5:
            return dict(diverged=True)
        TH[n], PH[n] = x[0], x[2]
    U[-1] = U[-2]; UC[-1] = UC[-2]
    t = np.arange(N + 1) * Ts
    out = dict(diverged=False, t=t, theta=TH, phi=PH, u=U, u_corr=UC,
               IAEtheta=float(np.trapezoid(np.abs(TH), t)),
               IAEphi=float(np.trapezoid(np.abs(PH), t)),
               phi_end=float(PH[-1]), theta_end=float(TH[-1]),
               phi_max=float(np.max(np.abs(PH))),
               RMSEtheta=float(np.sqrt(np.mean(TH**2))))
    ub = U - UC
    out['rho'] = float(np.max(np.abs(UC)) / max(np.max(np.abs(ub)), 1e-12)) if np.any(UC) else 0.0
    out['u_base_max'] = float(np.max(np.abs(ub)))
    return out
