import csv, collections, math, statistics as st
from scipy.stats import beta
rows=list(csv.DictReader(open('results/E5_MATLAB_2026a.csv')))
def cp(k,n):
    lo=0 if k==0 else beta.ppf(0.025,k,n-k+1); hi=1 if k==n else beta.ppf(0.975,k+1,n-k); return lo,hi
pol=[];
for r in rows:
    if r['Policy'] not in pol: pol.append(r['Policy'])
groups=['IC','mismatch','noise','combined','delay0','delay5','delay10']
rec={}  # (pol,group,draw)->bool
iae={}
for r in rows:
    rec[(r['Policy'],r['Group'],int(r['Draw']))]=r['Recovered'] in ('1','True','true')
    iae[(r['Policy'],r['Group'],int(r['Draw']))]=float(r['IAEtheta'])
n={g:len({k[2] for k in rec if k[1]==g and k[0]==pol[0]}) for g in groups}
out=['# Round 5 — E5 (distributions of initial state, mismatch, noise, delay): results','',
f'Policies done: {len(pol)} of 13. Recovery criterion of `Compare_v13_robust.m` (|θ|<0.01, |φ|<0.05 in the last 10 s, not diverged); exact Clopper-Pearson 95% intervals; π(0) removed for the agents. Draws per group: '+', '.join(f'{g} {n[g]}' for g in groups),'','## 1. Recovery rate by group (k/n)','','| Policy | '+' | '.join(groups)+' | all 310 |','|---|'+'---|'*(len(groups)+1)]
for p in pol:
    cells=[];tk=0;tn=0
    for g in groups:
        k=sum(v for (pp,gg,d),v in rec.items() if pp==p and gg==g); m=n[g]; tk+=k;tn+=m; cells.append(f'{k}/{m}')
    lo,hi=cp(tk,tn); out.append(f'| {p} | '+' | '.join(cells)+f' | {tk}/{tn} [{lo:.2f}, {hi:.2f}] |')
# agent summary
def agg(prefix):
    ps=[p for p in pol if p.startswith(prefix)]; ks=[sum(v for (pp,gg,d),v in rec.items() if pp==p) for p in ps]; return ps,ks
out+=['','## 2. Agents versus PSO-FLC (paired per configuration)','','Per agent: configurations recovered by the agent but not by PSO-FLC (+), by PSO-FLC but not by the agent (−), exact two-sided sign test on the discordant pairs.','','| Agent | + | − | sign-test p | median ΔIAE_θ (agent−PSO-FLC) on configurations both recovered |','|---|---|---|---|---|']
from math import comb
def sign_p(a,b):
    nn=a+b
    if nn==0: return 1.0
    m=min(a,b); return min(1.0,2*sum(comb(nn,i) for i in range(m+1))/2**nn)
keys=[k for k in rec if k[0]==pol[0]]
for p in pol:
    if p in ('PSO-FLC',): continue
    plus=sum(1 for k in keys if rec[(p,)+k[1:]] and not rec[('PSO-FLC',)+k[1:]]); minus=sum(1 for k in keys if (not rec[(p,)+k[1:]]) and rec[('PSO-FLC',)+k[1:]])
    dl=[iae[(p,)+k[1:]]-iae[('PSO-FLC',)+k[1:]] for k in keys if rec[(p,)+k[1:]] and rec[('PSO-FLC',)+k[1:]]]
    out.append(f'| {p} | {plus} | {minus} | {sign_p(plus,minus):.3g} | {st.median(dl) if dl else float("nan"):+.4f} |')
# BC+TD3 minus BC-only per seed
out+=['','## 3. BC+TD3 versus BC-only per seed (recovered configurations)','','| Seed | BC-only | BC+TD3 | BC+TD3 − BC-only |','|---|---|---|---|']
diffs=[]
for s in range(1,6):
    a,b=f'BC-only s{s}',f'BC+TD3 s{s}'
    if a in pol and b in pol:
        ka=sum(v for (pp,gg,d),v in rec.items() if pp==a); kb=sum(v for (pp,gg,d),v in rec.items() if pp==b); diffs.append(kb-ka); out.append(f'| {s} | {ka} | {kb} | {kb-ka:+d} |')
if diffs: out.append(f'\nBC+TD3 recovers more than BC-only in {sum(d>0 for d in diffs)} of {len(diffs)} seeds (exact sign-flip p = {sign_p(sum(d>0 for d in diffs),sum(d<0 for d in diffs)):.3g} on the seed-level differences; n = {len(diffs)}).')
# where failures: per-group failure of agents
out+=['','## 4. Notes','- All runs are open-loop to the criterion; no run diverged unless the CSV `Diverged` column says so (see counts in the log).','- Observational: the criterion is the authors\' recovery test; IAE columns are in the CSV.','- Controller parameters are nominal; noise is added to θ and φ̇ only; delay is a pure input delay.']
open('docs/Round5_E5_Report.md','w').write('\n'.join(out)+'\n'); print('\n'.join(out))
