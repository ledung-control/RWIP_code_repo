"""make_fig7_traj.py -- 600 s trajectories of the 15 stored policies for Fig. 7 (no retraining).
Policies: BC+TD3 raw, BC-only (raw), BC+TD3 with pi(0) subtracted ('unb'); seeds 1-5.
Output: results/fig7_traj.pkl  {(seed, 'raw'|'bco'|'unb'): dict(t, phi, Tg, cum, abias)}
 t,phi : flywheel angle every 0.1 s;  Tg,cum : accumulated IAE_phi at horizons Tg (s).
Usage: python3 -I make_fig7_traj.py [ROOT]   (ROOT holds rl_v13_repro/agent_*.mat; default ../..)
Resumable: one pickle per run in results/_fig7_cache/ (delete the folder to recompute)."""
import sys, os, pickle
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import numpy as np
import actorx, rwipsim as R, phlib as P
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', '..')
RD = os.path.join(ROOT, 'rl_v13_repro')
OUT = os.path.join(HERE, '..', 'results'); CACHE = os.path.join(OUT, '_fig7_cache')
os.makedirs(CACHE, exist_ok=True)
SURF = os.path.join(HERE, '..', 'surf201.npy')
S = np.load(SURF) if os.path.exists(SURF) else R.build_surface(201)[1]
Tg = np.concatenate([np.arange(5, 61, 1.0), np.arange(65, 601, 5.0)])      # 164 horizons
res = {}
for s in range(1, 6):
    for tag in ('raw', 'bco', 'unb'):
        f = os.path.join(CACHE, '%d_%s.pkl' % (s, tag))
        if not os.path.exists(f):
            net = actorx.load_actor(os.path.join(RD, ('agent_bconly_seed%d.mat' if tag == 'bco' else 'agent_bc_seed%d.mat') % s))
            ab = actorx.pi(net, np.zeros(6))
            pol = lambda o, n=net: actorx.pi(n, o)
            r = P.run(S, 600.0, pi=pol, a_bias=ab if tag == 'unb' else 0.0)
            t, ph = r['t'], r['phi']
            c = np.concatenate([[0.0], np.cumsum(0.5 * (np.abs(ph[1:]) + np.abs(ph[:-1])) * np.diff(t))])
            d = dict(t=t[::100], phi=ph[::100], Tg=Tg, cum=np.interp(Tg, t, c), abias=ab)
            pickle.dump(d, open(f, 'wb'))
        res[(s, tag)] = pickle.load(open(f, 'rb'))
pickle.dump(res, open(os.path.join(OUT, 'fig7_traj.pkl'), 'wb'))
print('done', len(res))
