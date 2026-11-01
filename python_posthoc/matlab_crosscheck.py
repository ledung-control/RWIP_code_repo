"""matlab_crosscheck.py -- compare the Python re-implementation (phlib) with the stored MATLAB outputs.
Inputs (read only): BC_vs_TD3_5seed_20260811_174452.mat (60 s nominal trajectories, raw policies),
                    Test_LongHorizon_v13_results.mat (600 s PSO-FLC), checkpoints in rl_v13_repro.
Output: results/matlab_crosscheck.csv"""
import sys, os, csv
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import numpy as np, h5py, scipy.io as sio
import actorx, phlib as P
ROOT=sys.argv[1] if len(sys.argv)>1 else '/mnt/user-data/uploads/WRIP_2'
S=np.load(os.path.join(HERE,'surf201.npy'))
f=h5py.File(os.path.join(ROOT,'BC_vs_TD3_5seed_20260811_174452.mat'),'r')
nom=f['R_BConly_TD3']['nom']
rows=[]
def cmp(label,r,th_m,ph_m,mat_iae=None):
    n=min(len(th_m),len(r['theta']))
    dth=np.max(np.abs(r['theta'][:n]-th_m[:n])); dph=np.max(np.abs(r['phi'][:n]-ph_m[:n]))
    rows.append([label,n,dth,dph,float(np.max(np.abs(th_m[:n]))),float(np.max(np.abs(ph_m[:n])))])
ref=f['R_BConly_TD3']['ref']['PSOFLC']
r=P.run(S,60.0); cmp('PSO-FLC 60s',r,np.array(ref['theta']).ravel(),np.array(ref['phi']).ravel())
for kind,key,fn in (('td3','BC_TD3S%d','agent_bc_seed%d.mat'),('bco','BC_onlyS%d','agent_bconly_seed%d.mat')):
    for s in range(1,6):
        net=actorx.load_actor(os.path.join(ROOT,'rl_v13_repro',fn%s))
        r=P.run(S,60.0,pi=lambda o,n=net: actorx.pi(n,o))
        g=nom[key%s]; cmp('%s seed%d raw 60s'%(kind,s),r,np.array(g['theta']).ravel(),np.array(g['phi']).ravel())
m=sio.loadmat(os.path.join(ROOT,'Test_LongHorizon_v13_results.mat'),squeeze_me=True,struct_as_record=False)
R1=m['R1'][0]; r=P.run(S,600.0)
cmp('PSO-FLC 600s',r,np.asarray(R1.theta),np.asarray(R1.phi))
rows.append(['PSO-FLC 600s IAEphi MATLAB vs Python',0,float(R1.IAEphi),P.iae(r['phi'],r['t'],600.0),0,0])
os.makedirs(os.path.join(HERE,'results'),exist_ok=True)
with open(os.path.join(HERE,'results','matlab_crosscheck.csv'),'w',newline='') as fo:
    w=csv.writer(fo); w.writerow(['case','n_samples','max_abs_dtheta_or_value1','max_abs_dphi_or_value2','max_abs_theta_matlab','max_abs_phi_matlab']); w.writerows(rows)
for x in rows: print(x)
