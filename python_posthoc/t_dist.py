import sys; sys.path.insert(0,".")
import numpy as np, actorx, phlib as P
S=np.load('surf201.npy'); RD='/mnt/user-data/uploads/WRIP_2/rl_v13_repro/'
ref={1:{1.0:((0.16034,137.68539),(0.13791,29.72898)),2.0:((0.35483,235.68847),(0.30611,96.87215))},
     3:{1.0:((0.13542,282.12529),(0.12942,29.58904)),2.0:((0.23947,267.90176),(0.27542,76.85841))}}
for s in (1,3):
    net=actorx.load_actor(RD+f'agent_bc_seed{s}.mat'); pi=lambda o,n=net: actorx.pi(n,o); ab=actorx.pi(net,np.zeros(6))
    for amp in (1.0,2.0):
        a=P.run(S,60,pi=pi,dist_t=5,dist_amp=amp); b=P.run(S,60,pi=pi,a_bias=ab,dist_t=5,dist_amp=amp)
        print(s,amp,'raw %.5f %.3f'%(P.iae(a['theta'],a['t'],60),P.iae(a['phi'],a['t'],60)),ref[s][amp][0],'| unb %.5f %.3f'%(P.iae(b['theta'],b['t'],60),P.iae(b['phi'],b['t'],60)),ref[s][amp][1], 'clip',a['res_clip'],a['tot_clip'],b['res_clip'],b['tot_clip'])
