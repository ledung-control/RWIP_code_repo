import sys; sys.path.insert(0,".")
import time, numpy as np, actorx, rwipsim as R, phlib as P
S=np.load('surf201.npy')
RD='/mnt/user-data/uploads/WRIP_2/rl_v13_repro/'
net=actorx.load_actor(RD+'agent_bc_seed3.mat'); pi=lambda o: actorx.pi(net,o); ab=actorx.pi(net,np.zeros(6))
t0=time.time(); a=P.run(S,60,pi=pi); print('time',time.time()-t0)
b=R.simulate(S,Tf=60,policy=pi)
print(P.iae(a['theta'],a['t'],60), b['IAEtheta'], P.iae(a['phi'],a['t'],60), b['IAEphi'], a['rho'], b['rho'])
c=P.run(S,60,pi=pi,a_bias=ab); d=R.simulate(S,Tf=60,policy=pi,a_bias=ab)
print(P.iae(c['theta'],c['t'],60), d['IAEtheta'], P.iae(c['phi'],c['t'],60), d['IAEphi'], c['rho'], d['rho'], c['res_clip'],c['tot_clip'],c['n_agent'])
e=P.run(S,60); f=R.simulate(S,Tf=60); print(P.iae(e['theta'],e['t'],60), f['IAEtheta'], e['tot_clip'])
