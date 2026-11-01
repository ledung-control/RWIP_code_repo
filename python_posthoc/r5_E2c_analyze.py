import csv,glob,itertools,statistics as st
ref={}
for r in csv.DictReader(open('/tmp/ph/Round4_PostHoc_Package/results/runs_600s.csv')):
    if r['variant'] in('unbias','raw'): ref[(r['policy'],int(r['seed']),r['variant'])]=r
E={}
for f in sorted(glob.glob('results/E2c_final/E2c_s*_MATLAB_2026a.csv')):
    for r in csv.DictReader(open(f)): E[(r['Group'],int(r['Seed']),int(r['Tf']))]=r
def signflip(d):
    n=len(d);obs=sum(d);c=0
    for s in itertools.product([1,-1],repeat=n):
        if sum(a*b for a,b in zip(s,d))<=obs: c+=1
    return c/2**n
out=[]
for grp,pol in(('BC-only','bco'),('BC+TD3','td3')):
    for T in(60,600):
        for m,k in(('IAEphi','IAEphi'),('IAEtheta','IAEtheta')):
            a=[float(E[(grp,s,T)][m]) for s in range(1,6)]
            ub=[float(ref[(pol,s,'unbias')][f'{k}_{T}']) for s in range(1,6)]
            rw=[float(ref[(pol,s,'raw')][f'{k}_{T}']) for s in range(1,6)]
            d=[x-y for x,y in zip(a,ub)]
            out.append((grp,T,m,a,ub,rw,st.median(a),st.median(ub),st.median(rw),sum(x<y for x,y in zip(a,ub)),signflip(d)))
for o in out: print(o[0],o[1],o[2],'E2c med %.4g | unbias med %.4g | raw med %.4g | E2c<unbias in %d/5 | p(sign-flip,E2c-unbias<=obs)=%.4f'%(o[6],o[7],o[8],o[9],o[10]))
print('pi0',{k:v['pi0'] for k,v in E.items() if k[2]==60})
for grp in('BC-only','BC+TD3'):
    print(grp,'phi_end max',max(abs(float(E[(grp,s,600)]['phi_end'])) for s in range(1,6)),'rho',[round(float(E[(grp,s,60)]['rho']),3) for s in range(1,6)])
print('per seed E2c td3 IAEphi60',[round(float(E[('BC+TD3',s,60)]['IAEphi']),3) for s in range(1,6)])
print('per seed unbias td3',[round(float(ref[('td3',s,'unbias')]['IAEphi_60']),3) for s in range(1,6)])
print('per seed raw td3',[round(float(ref[('td3',s,'raw')]['IAEphi_60']),3) for s in range(1,6)])
print('per seed E2c bco',[round(float(E[('BC-only',s,60)]['IAEphi']),3) for s in range(1,6)])
print('per seed unbias bco',[round(float(ref[('bco',s,'unbias')]['IAEphi_60']),3) for s in range(1,6)])
