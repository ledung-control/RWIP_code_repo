import csv,glob,sys,statistics as st,re,os
d=sys.argv[1]; out=sys.argv[2]
rows=[]
for f in sorted(glob.glob(d+'/E1_c*_MATLAB_2026a.csv')):
    rows+=list(csv.DictReader(open(f)))
CF={1:(10,.995),2:(10,.999),3:(10,.9995),4:(40,.995),5:(40,.999),6:(40,.9995)}
def sel(c,grp,var,col):
    return [float(r[col]) for r in rows if int(r['Config'])==c and r['Group']==grp and r['Variant']==var]
seeds={c:sorted({int(r['Seed']) for r in rows if int(r['Config'])==c}) for c in CF}
lines=['config,S_PH,gamma,n,seeds,group,variant,median_IAEtheta,median_IAEphi,median_abs_pi0,wall_s_mean']
for c in CF:
    for grp in ('BC-only','BC+TD3'):
        for var in ('raw','unbiased'):
            th=sel(c,grp,var,'IAEtheta'); ph=sel(c,grp,var,'IAEphi'); p0=[abs(x) for x in sel(c,grp,var,'pi0')]; w=sel(c,grp,var,'wall_s')
            if th: lines.append('c%d,%g,%g,%d,%s,%s,%s,%.6g,%.6g,%.5g,%.0f'%(c,CF[c][0],CF[c][1],len(th),'/'.join(map(str,seeds[c])),grp,var,st.median(th),st.median(ph),st.median(p0),st.mean(w)))
# selection rule (protocol E1): BC+TD3, pi(0) removed; reference = c1
ref=st.median(sel(1,'BC+TD3','unbiased','IAEphi'))
res=['','# Selection rule: lowest median 60 s IAEtheta (BC+TD3, pi(0) removed) among configs whose median IAEphi (same variant) is not worse than c1 (%.5g)'%ref]
cand=[]
for c in CF:
    th=sel(c,'BC+TD3','unbiased','IAEtheta'); ph=sel(c,'BC+TD3','unbiased','IAEphi')
    if not th: continue
    ok=st.median(ph)<=ref
    res.append('# c%d n=%d median IAEtheta %.6g median IAEphi %.5g -> %s'%(c,len(th),st.median(th),st.median(ph),'eligible' if ok else 'not eligible'))
    if ok: cand.append((st.median(th),c))
res.append('# selected (ties -> fewer deviations from paper config): '+(('c%d'%min(cand)[1]) if cand else 'none eligible'))
open(out,'w').write('\n'.join(lines+res)+'\n'); print('\n'.join(lines+res))
