# -*- coding: utf-8 -*-
"""Rebuild the numbers of Tables 4-7 of the manuscript from the archived MATLAB result files.

Run from the repository root:   python python_posthoc/export_tables_4_7.py
Reads  results/archived_tables4-7/*.mat  (written by Compare_v13.m, Test_Disturbance_v13.m,
Benchmark_ComputeTime_v13.m) and, if present, Compare_v13_robust_results.mat (114 MB, not
stored in the repository; regenerate it with matlab_core/Compare_v13_robust.m, or put it in
results/archived_tables4-7/).  Writes results/archived_tables4-7/Table{4,5,6,7}.csv.
Needs NumPy and SciPy only.
"""
import os, sys, csv
import numpy as np, scipy.io as sio

D = os.path.join("results", "archived_tables4-7")
ld = lambda f, **k: sio.loadmat(os.path.join(D, f), squeeze_me=True, struct_as_record=False, **k)

def w(name, header, rows):
    with open(os.path.join(D, name), "w", newline="") as f:
        c = csv.writer(f); c.writerow(header); c.writerows(rows)
    print("wrote", name)

# ---- Table 4 (and the data of Fig. 5): Compare_v13_results.mat -------------------------
R = ld("Compare_v13_results.mat")["R"]
F = ["IAEtheta", "IAEphi", "IAEu", "RMSEtheta", "RMSEphi", "Urms", "OS_theta", "ts_theta", "ts_phi"]
def edu_after(r, t0=0.1):                      # as in Compare_v13.m: trapz(t, du^2) for t > 0.1 s
    t = np.asarray(r.t, float); u = np.asarray(r.u, float); dt = t[1] - t[0]
    du = np.r_[0.0, np.diff(u)] / dt; m = t > t0
    return float((getattr(np, "trapezoid", None) or np.trapz)(du[m] ** 2, t[m]))
rows = [[f] + [float(getattr(getattr(R, k), f)) for k in ("PSOFLC", "SMCESO", "HYBRID")] for f in F]
rows.append(["Edu_after_0.1s"] + [edu_after(getattr(R, k)) for k in ("PSOFLC", "SMCESO", "HYBRID")])
w("Table4.csv", ["metric", "PSO-FLC", "SMC-ESO", "PSO-FLC+SMC-ESO"], rows)

# ---- Table 5 and the threshold of Table 6: Compare_v13_robust_results.mat ---------------
rob = os.path.join(D, "Compare_v13_robust_results.mat")
if os.path.isfile(rob):
    d = ld("Compare_v13_robust_results.mat", variable_names=["R1", "pert", "kickMax", "names"])
    IA = np.array([[d["R1"][i, j].IAEtheta for j in range(3)] for i in range(d["R1"].shape[0])])
    rows = [[str(d["pert"][i, 0])] + list(IA[i]) + list(100 * (IA[i] / IA[0] - 1)) for i in range(IA.shape[0])]
    w("Table5_cases.csv", ["case", "IAEtheta_PSOFLC", "IAEtheta_SMCESO", "IAEtheta_Hybrid",
                           "pct_PSOFLC", "pct_SMCESO", "pct_Hybrid"], rows)
    w("Table6_threshold.csv", ["controller", "recovery_threshold_rad_s"],
      [[str(d["names"][m]), float(d["kickMax"][m])] for m in range(3)])
else:
    print("note: Compare_v13_robust_results.mat not found; Table5_cases.csv / Table6_threshold.csv "
          "are the copies archived with the repository.")

# ---- Table 6, recovery times: Test_Disturbance_v13_results.mat --------------------------
T = ld("Test_Disturbance_v13_results.mat")
rows = []
for a, k in enumerate(T["kicks"]):
    row = [float(k)]
    for m in range(3):
        tr = np.atleast_1d(T["tRec"][a, :, m]); tr = tr[np.isfinite(tr)]
        row.append(round(float(tr.mean()), 2) if tr.size else "no recovery")
    rows.append(row)
w("Table6_recovery_time.csv", ["impulse_rad_s", "PSO-FLC", "SMC-ESO", "PSO-FLC+SMC-ESO"], rows)

# ---- Table 7: Benchmark_ComputeTime_v13_results.mat -------------------------------------
B = ld("Benchmark_ComputeTime_v13_results.mat")
us = lambda x: 1e6 * float(x)
outer, look, smc, ev = us(B["t_outer"]), us(B["t_lookup"]), us(B["t_smc"]), us(B["t_evalfis"])
w("Table7.csv", ["configuration", "us_per_step"],
  [["PSO-FLC (201x201 look-up) = lookup + outer loop", look + outer],
   ["SMC-ESO cascade = SMC step + outer loop", smc + outer],
   ["PSO-FLC + SMC-ESO = lookup + SMC step + outer loop", look + smc + outer],
   ["PSO-FLC (direct evalfis) = evalfis + outer loop", ev + outer]])
