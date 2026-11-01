"""compare_phase1.py <Round5 results dir> <project dir (read-only inputs)> -> prints and writes phase1_comparison.csv
Compares MATLAB R2026a re-runs (baseline_A_v2 / baseline_B) with (i) stored MATLAB data and (ii) the verified Python re-implementation."""
import sys,csv,os,scipy.io as sio
RD,PD=sys.argv[1],sys.argv[2]; rows=[]
def rec(item,metric,new,ref,refsrc,tol=1e-6):
    if ref is None or new is None: return
    d=new-ref; rel=abs(d)/max(abs(ref),1e-12)
    if abs(d)<=1e-9 or rel<=tol: st='MATCH'
    elif (metric=='phi_end' and abs(d)<=1e-5) or (metric!='phi_end' and rel<=1e-4): st='CLOSE'   # float32 networks in MATLAB vs float64 Python; documented
    else: st='DIFF'
    rows.append([item,metric,new,ref,d,rel,st,refsrc])
# (i) stored MATLAB: Compare_v13_results.mat (60 s) and Test_LongHorizon_v13_results.mat (600 s)
m=sio.loadmat(os.path.join(PD,'Compare_v13_results.mat'),squeeze_me=True,struct_as_record=False)['R']
t=sio.loadmat(os.path.join(PD,'Test_LongHorizon_v13_results.mat'),squeeze_me=True,struct_as_record=False)['R1']
mp={'PSOFLC':(m.PSOFLC,t[0]),'SMCESO':(m.SMCESO,t[1]),'PSOFLC_SMCESO':(m.HYBRID,t[2])}
A=[r for r in csv.DictReader(open(os.path.join(RD,'baseline_A_v2_MATLAB_2026a.csv')))] if os.path.exists(os.path.join(RD,'baseline_A_v2_MATLAB_2026a.csv')) else []
for r in A:
    s60,s600=mp[r['Controller']]; ref=s60 if r['Tf']=='60' else s600
    for k in ('IAEtheta','IAEphi','OS_theta','ts_theta','ts_phi','Urms','RMSEtheta','RMSEphi'):
        if k in r and hasattr(ref,k): rec('%s %ss'%(r['Controller'],r['Tf']),k,float(r[k]),float(getattr(ref,k)),'stored MATLAB '+('Compare_v13_results.mat' if r['Tf']=='60' else 'Test_LongHorizon_v13_results.mat'))
# (i-b) stored MATLAB per-seed 60 s: BC_vs_TD3_5seed_paired csv ; (ii) Python re-implementation
P={(x['policy'],int(x['seed']),x['variant']):x for x in csv.DictReader(open(os.path.join(PD,'..','..','..','..','pkg','results','runs_600s.csv')))} if False else {}
pcsv=os.environ.get('PY_RUNS','/home/claude/pkg/results/runs_600s.csv')
if os.path.exists(pcsv): P={(x['policy'],int(x['seed']),x['variant']):x for x in csv.DictReader(open(pcsv))}
pair={int(x['Seed']):x for x in csv.DictReader(open(os.path.join(PD,'BC_vs_TD3_5seed_paired_20260811_174452.csv')))}
bf=os.path.join(RD,'baseline_B_MATLAB_2026a.csv')
if os.path.exists(bf):
    for r in csv.DictReader(open(bf)):
        grp,seed=r['Controller'].split('_s'); seed=int(seed); pol='bco' if grp=='BC-only' else 'td3'
        var='raw' if r['Variant']=='raw' else 'unbias'; tf=r['Tf']; nm='%s s%d %s %ss'%(grp,seed,r['Variant'],tf)
        if var=='raw' and tf=='60':
            pre='BC' if pol=='bco' else 'TD3'
            rec(nm,'IAEtheta',float(r['IAEtheta']),float(pair[seed][pre+'_IAEtheta']),'stored MATLAB BC_vs_TD3_5seed_paired csv')
            rec(nm,'IAEphi',float(r['IAEphi']),float(pair[seed][pre+'_IAEphi']),'stored MATLAB BC_vs_TD3_5seed_paired csv')
        x=P.get((pol,seed,var))
        if x:
            rec(nm,'IAEtheta',float(r['IAEtheta']),float(x['IAEtheta_'+tf]),'Python re-implementation (runs_600s.csv)')
            rec(nm,'IAEphi',float(r['IAEphi']),float(x['IAEphi_'+tf]),'Python re-implementation (runs_600s.csv)')
            rec(nm,'phi_end',float(r['phi_end']),float(x['phi_'+tf]),'Python re-implementation (runs_600s.csv)',tol=1e-4)
w=csv.writer(open(os.path.join(RD,'phase1_comparison.csv'),'w',newline='')); w.writerow(['item','metric','MATLAB_R2026a_rerun','reference','abs_diff','rel_diff','status','reference_source']); w.writerows(rows)
n=len(rows); mm=sum(r[6]=='MATCH' for r in rows); cc=sum(r[6]=='CLOSE' for r in rows); print(n,'comparisons;',mm,'MATCH (rel<=1e-6 or abs<=1e-9);',cc,'CLOSE (IAE rel<=1e-4, phi_end abs<=1e-5);',n-mm-cc,'DIFF')
import collections
c=collections.Counter((r[7].split(' (')[0],r[6]) for r in rows); print(dict(c))
for r in rows:
    if r[6]=='DIFF': print('DIFF',r[0],r[1],r[2],r[3],'rel=%.2e'%r[5],r[7])
