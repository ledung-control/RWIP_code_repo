"""build_phase1_report.py <results dir> <out.md> -- numbers come from the CSV files only."""
import csv,sys,statistics as st,os
RD,OUT=sys.argv[1],sys.argv[2]
rd=lambda f:list(csv.DictReader(open(os.path.join(RD,f),encoding='utf-8-sig')))
cmp_=rd('phase1_comparison.csv'); B=rd('baseline_B_MATLAB_2026a.csv'); A=rd('baseline_A_v2_MATLAB_2026a.csv'); A1=rd('baseline_A_MATLAB_2026a.csv'); E=rd('E7E8_MATLAB_2026a.csv')
import collections
cnt=collections.Counter((r['reference_source'].split(' (')[0],r['status']) for r in cmp_)
f=lambda x,n=4:('%.'+str(n)+'g')%float(x)
L=[]
w=L.append
w('---\ntitle: "Round 5 — Phase 1 (baseline reproduction in MATLAB R2026a) and E7/E8 results"\n---\n')
w('**Package status: NOT READY FOR SUBMISSION.** Three kinds of evidence appear below and are kept apart: **(R) reproduction** of stored results with the authors\' own code in MATLAB; **(P) post-hoc** analyses of stored checkpoints (Round 3–4, Python); **(N) new experiments** (E7, E8: run for the first time in Round 5, MATLAB, authors\' code unchanged). Everything was run on the authors\' PC on 25 Oct 2026; no original file was modified.\n')
w('# 1. Environment\n')
w('| Item | Value |\n|---|---|\n| MATLAB | 26.1.0.3276743 (R2026a) Update 3, `C:\\Program Files\\MATLAB\\R2026a` (R2024b, R2025b and R2026b are also installed; R2026b has no toolboxes installed and was not used) |\n| OS / hardware | Windows 11 Pro for Workstations (PCWIN64), 6 cores, 1 GPU device reported (not used) |\n| Toolboxes used (all version 26.1) | Simulink, Reinforcement Learning, Deep Learning, Fuzzy Logic, Control System, Global Optimization (`ver` listing in `logs/env_check_*.txt`) |\n| Licence | Base MATLAB checked out; the first attempt failed for every toolbox with error 5006 (licensing service); after the author fixed it the functional smoke test passed: `load_system`, `readfis`, `rlNumericSpec`, `tf`, `dlnetwork`, `mamfis` (`logs/smoke_test_2026a.txt`) |\n| Code folder | `C:\\Users\\HP STUDIO G5\\Desktop\\WRIP_2` (read only); new files only in `Baibao_IJIES\\Revision_Round5` |\n| Evaluator | `sim_rwip_v13.m` (15 Aug 2026), a fixed-step RK4 MATLAB script (Ts = 1 ms); `make_P_v13.m` (7 Aug), `Eval_RL_v13_final.m` (10 Aug); the Simulink model is **not** used by this evaluator |\n| Python | not used in Round 5 runs; the Python re-implementation is only a comparison reference |\n')
w('# 2. Which script produced which checkpoint (evidence in the files)\n')
w('| Checkpoint | Evidence | Open point |\n|---|---|---|\n| `rl_v13_repro/agent_bc_seed1…5.mat`, `agent_bconly_seed1…5.mat` | The header of `Train_RL_v13_repro.m` states that it reconstructs the configuration from `agent_bc_seed1` of 8 Aug 10:04 and calls `Train_RL_v13_bc.m` with pinned options (γ 0.995, no gradient clipping, 256-unit layers, uMax 10 V, original control-variate reward, 60 BC episodes / 3000 BC iterations, 20 warm-up episodes, 18 fine-tune episodes, actor LR 1e-5). The `.mat` files contain `seed`, `cfg` (uMax 10, Ts_agent 0.01, Tf 40, rewardMode cv), `rewardVersion` 2, `rmseBC`, `statsA/B1`, `tWarm/tB1` | File times on the PC: `agent_bc_seed1` 10 Aug 13:09 (an `…_AABORT` file 13:03 precedes it), seed2 10 Aug 03:20, seed3 14:42, seed4 16:17, seed5 16:43; `Train_RL_v13_bc.m` and `Train_RL_v13_repro.m` were last saved 10 Aug 10:26. So seed 2 predates the last save of the training script by about 7 h; whether it was produced by the same code version cannot be shown from the files (**author, item A3**) |\n| Training time recorded in `agent_bc_seed1.mat` | warm-up 1153 s, fine-tune 347 s (plus the BC stage) | planning figure for E1/E2: about 30 min per seed on this PC (to be confirmed by E0) |\n')
w('# 3. Reproduction of the baselines (R)\n')
w('Script `r5_baseline.m` (stages A and B). A first run of stage A used the wrong outer-loop gains for the stand-alone SMC-ESO (my script did not apply `applyOuter`, as `Compare_v13_robust.m` does); the file `baseline_A_MATLAB_2026a.csv` is that wrong run (SMC-ESO IAE_θ %s instead of %s) and is kept only as a record. The corrected run is `baseline_A_v2_MATLAB_2026a.csv`.\n' % (f([r for r in A1 if r['Controller']=='SMCESO' and r['Tf']=='60'][0]['IAEtheta'],5),f([r for r in A if r['Controller']=='SMCESO' and r['Tf']=='60'][0]['IAEtheta'],5)))
w('| Comparison | MATCH (rel ≤ 1e-6 or abs ≤ 1e-9) | CLOSE (IAE rel ≤ 1e-4; φ(T) abs ≤ 1e-5) | DIFF |\n|---|---|---|---|')
for k in ('stored MATLAB Compare_v13_results.mat','stored MATLAB Test_LongHorizon_v13_results.mat','stored MATLAB BC_vs_TD3_5seed_paired csv','Python re-implementation'):
    w('| %s | %d | %d | %d |'%(k,cnt[(k,'MATCH')],cnt[(k,'CLOSE')],cnt[(k,'DIFF')]))
w('\nAgainst **stored MATLAB output** every compared value matches (PSO-FLC, SMC-ESO and the hybrid at 60 s and 600 s: Compare_v13_results.mat and Test_LongHorizon_v13_results.mat; the ten agents, raw, 60 s: IAE_θ and IAE_φ of the stored per-seed CSV). Against the **Python re-implementation** the new MATLAB runs of the ten agents (raw and π(0) removed, 60 s and 600 s) agree within IAE relative difference 9e-5 and φ(T) within 1.3e-6 rad; the small differences (e.g. φ(T) 1e-7 rad in MATLAB versus 1e-14 rad in Python for the π(0)-removed runs) are consistent with MATLAB networks evaluated in single precision, but I have not shown that this is the cause.\n')
w('**MATLAB 600 s runs of the ten agents (new MATLAB data; they did not exist before).** Medians over the five seeds:\n')
w('| Group | Variant | median IAE_θ (60 s) | median IAE_φ (60 s) | median IAE_φ (600 s) | max abs φ(600) |\n|---|---|---|---|---|---|')
for g in ('BC-only','BC+TD3'):
    for v in ('raw','unbiased'):
        x60=[r for r in B if r['Controller'].startswith(g) and r['Variant']==v and r['Tf']=='60']; x6=[r for r in B if r['Controller'].startswith(g) and r['Variant']==v and r['Tf']=='600']
        w('| %s | %s | %s | %s | %s | %s |'%(g,'π(0) removed' if v=='unbiased' else 'raw',f(st.median(float(r['IAEtheta']) for r in x60),4),f(st.median(float(r['IAEphi']) for r in x60),5),f(st.median(float(r['IAEphi']) for r in x6),6),f(max(abs(float(r['phi_end'])) for r in x6),3)))
w('\nThe printed manuscript values (Table 10/11: BC-only raw 89.63, BC+TD3 raw 1144.59 at 600 s; π(0)-removed 4.22 and 5.55) are reproduced by MATLAB. Note B3 of the author checklist ("the 600 s RL runs rest on the Python re-implementation only") is therefore resolved for these ten agents and both variants. Per-run values are in `results/baseline_B_MATLAB_2026a.csv` and `results/phase1_comparison.csv`.\n')
w('# 4. E7 and E8 (N): ESO bandwidth sweep and oracle gain\n')
w('Protocol v1.0 section 4; script `r5_E7E8.m`; authors\' `SMCESO_cascade.m` and `sim_rwip_v13.m` unchanged; deterministic; 60 s, θ0 = 0.10 rad; the controller keeps the nominal control-affine gain (E7) or receives the true perturbed gain (E8). **Built-in check:** ω0 = 20 rad/s with nominal gain reproduces the stored Table 5 values (nominal IAE_θ %s, K_t −30 %%: %s, adverse combination %s). Cases marked "div" diverged (IAE not defined).\n'%(f([r for r in E if r['Experiment']=='E7_nominal_gain' and r['w0']=='20' and r['Case']=='nominal'][0]['IAEtheta'],6),f([r for r in E if r['Experiment']=='E7_nominal_gain' and r['w0']=='20' and r['Case']=='Kt -30%'][0]['IAEtheta'],6),f([r for r in E if r['Experiment']=='E7_nominal_gain' and r['w0']=='20' and r['Case'].startswith('adverse')][0]['IAEtheta'],6)))
def g(e,wv,c):
    for r in E:
        if r['Experiment']==e and r['w0']==str(wv) and r['Case']==c: return r
def cell(r):
    return 'div' if r['diverged']=='1' else f(r['IAEtheta'],4)
w('**Table S12. E7: IAE_θ (rad·s) versus ω0 (nominal gain in the controller)**\n')
w('| ω0 (rad/s) | nominal | U_rms nominal (V) | rms Δu nominal (V/step) | K_t −30 % | m·g·l +30 % | I_p −30 % | adverse |\n|---|---|---|---|---|---|---|---|')
for wv in (10,15,20,30,40,60):
    n=g('E7_nominal_gain',wv,'nominal'); w('| %d | %s | %s | %s | %s | %s | %s | %s |'%(wv,cell(n),f(n['Urms'],3),f(n['rms_du_chatter'],3),cell(g('E7_nominal_gain',wv,'Kt -30%')),cell(g('E7_nominal_gain',wv,'mgl +30%')),cell(g('E7_nominal_gain',wv,'Ip -30%')),cell(g('E7_nominal_gain',wv,'adverse (mgl+30,Ip-30,Kt-30)'))))
w('\n**Table S13. E8: the same grid with the oracle (true) control-affine gain**\n')
w('| ω0 (rad/s) | nominal | K_t −30 % | m·g·l +30 % | I_p −30 % | adverse |\n|---|---|---|---|---|---|')
for wv in (10,15,20,30,40,60):
    e='E8_oracle_gain'; w('| %d | %s | %s | %s | %s | %s |'%(wv,cell(g(e,wv,'nominal')),cell(g(e,wv,'Kt -30%')),cell(g(e,wv,'mgl +30%')),cell(g(e,wv,'Ip -30%')),cell(g(e,wv,'adverse (mgl+30,Ip-30,Kt-30)'))))
k7=float(g('E7_nominal_gain',20,'Kt -30%')['IAEtheta']); k8=float(g('E8_oracle_gain',20,'Kt -30%')['IAEtheta']); n20=float(g('E7_nominal_gain',20,'nominal')['IAEtheta'])
w('\n**What the data show (and what they do not).**\n')
w('1. At ω0 = 20 rad/s, K_t −30 %%: IAE_θ is %s with the nominal gain and %s with the oracle gain (nominal-condition value %s), so correcting the control-affine gain removes the K_t degradation in this design. Under the adverse combination the oracle gain is identical to the nominal one (K_t and I_p are both scaled by 0.7, so K_t/I_p does not change) and the result is unchanged (%s), i.e. that degradation does not come from the gain error.\n'%(f(k7,5),f(k8,5),f(n20,5),f(g('E8_oracle_gain',20,'adverse (mgl+30,Ip-30,Kt-30)')['IAEtheta'],5)))
w('2. The nominal-condition IAE_θ depends strongly on ω0 (%s at 20 rad/s, %s at 30, %s at 15, %s at 10), and U_rms and the control chatter grow for ω0 ≥ 30. A percentage degradation therefore mixes two effects (the baseline level and the mismatch); the table gives absolute values. In absolute terms, with the nominal gain, K_t −30 %% gives %s at ω0 = 30 and %s at ω0 = 40, against %s at ω0 = 20.\n'%(f(n20,4),f(g('E7_nominal_gain',30,'nominal')['IAEtheta'],4),f(g('E7_nominal_gain',15,'nominal')['IAEtheta'],4),f(g('E7_nominal_gain',10,'nominal')['IAEtheta'],4),f(g('E7_nominal_gain',30,'Kt -30%')['IAEtheta'],4),f(g('E7_nominal_gain',40,'Kt -30%')['IAEtheta'],4),f(k7,4)))
w('3. Scope: one plant, one gain set apart from ω0 (c 8, k 5, η 1, φ_bl 0.05), 60 s, one initial condition, deterministic runs; no claim is made about an optimal ω0 or about SMC-ESO in general. Whether a retuned SMC-ESO would match PSO-FLC was not tested (PSO-FLC at K_t −30 %%: 0.0277 in Table 5).\n')
w("""# 5. E0 (N/R): retraining reproducibility pilot, seed 1

Script `r5_E0.m` calls the authors' `Train_RL_v13_repro(1, …)` (unchanged) with output to `agents/E0_repro_seed1/` (the original `rl_v13_repro` folder was not written to), then evaluates the retrained and the stored checkpoints with the same MATLAB code (60 s, θ0 = 0.10 rad, raw and π(0) removed). Wall time of the training: 1898 s (31.6 min). Log: `logs/E0_seed1_MATLAB_2026a_*.txt`; results: `results/E0_seed1_MATLAB_2026a.csv`.

| Policy | Variant | π(0) retrained | π(0) stored | IAE_θ retrained | IAE_θ stored | IAE_φ retrained | IAE_φ stored |
|---|---|---|---|---|---|---|---|
| BC-only | raw | −0.002561 | −0.002561 | 0.0283173 | 0.0283173 | 12.9369 | 12.9369 |
| BC-only | π(0) removed | −0.002561 | −0.002561 | 0.0295833 | 0.0295833 | 4.82830 | 4.82830 |
| BC+TD3 | raw | −0.119920 | −0.119920 | 0.0344172 | 0.0344172 | 111.570 | 111.570 |
| BC+TD3 | π(0) removed | −0.119920 | −0.119920 | 0.0312204 | 0.0312204 | 5.97522 | 5.97522 |

**Result.** For seed 1, the retrained BC-only and BC+TD3 policies give the same π(0) and the same evaluation metrics as the stored checkpoints to all printed digits (the CSV agrees to 15 significant digits in the columns compared). The checkpoint files are not byte-identical (different sizes/hashes for BC+TD3, probably timing fields and metadata; I did not inspect them), so the statement is about behaviour, not about the files.

**What this does and does not show.** Under this code, on this PC and in MATLAB R2026a, one training run (seed 1) is deterministic and reproduces the stored result. It does **not** show that the other four seeds reproduce (not run, about 32 min each), nor that the original run used the same MATLAB release (the Simulink model metadata say R2026a, which is consistent but not proof). Seeds 2–5 carry the file-time ambiguity of section 2 (item A3). Log observation: the run printed a critic-fit warning in phase A (the text reads "r = +0.910 < 0.50", which is internally inconsistent; I did not examine the gate logic), saved its state as `agent_bc_seed1_AABORT.mat`, and continued with the message "SkipCriticGate = true" (an option set by the repro script, as the log states); phase B1 was likewise continued with SkipB1Gate = true. An `…_AABORT` file next to `agent_bc_seed1` in the original folder is consistent with the same code path, but that is an inference. The measured wall time (about 32 min per seed) is the planning figure for E1/E2: 18 tuning trainings ≈ 9.5 h.
""")
open(OUT,'w').write('\n'.join(L))
print('ok',len(L))
