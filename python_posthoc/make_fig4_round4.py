"""Round 4: redraw Figure 4 (architecture diagram) WITHOUT the 'displacement ~ 0.36 units' text.
The original figure script is not available; this re-creates the layout of the original raster
(same boxes, text and colours) and replaces only the Stage B third line with the measured value read
from results/param_displacement.csv (row tensor=ALL: L2 change of all 67841 actor parameters
between agent_bconly_seed{k} and agent_bc_seed{k}, k=1..5)."""
import sys, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
plt.rcParams['font.family']='Liberation Serif'; plt.rcParams['mathtext.fontset']='stix'
csv=sys.argv[1]; out=sys.argv[2]
d=pd.read_csv(csv); a=d[d.tensor=='ALL']; assert len(a)==5 and (a.n_params==67841).all()
lo,hi=a.L2_change.min(),a.L2_change.max()
rng=f"{lo:.2f}\u2013{hi:.2f}"; print('L2 range',lo,hi,rng)
W,H=693,1316; fig=plt.figure(figsize=(W/200,H/200),dpi=400)
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(H,0); ax.axis('off')
BL='#1f3a6e'; OR='#c8601a'; RD='#b01c2e'
def box(x0,y0,x1,y1,fc,ec,txt,fs=12,ls=1.3):
    ax.add_patch(FancyBboxPatch((x0,y0),x1-x0,y1-y0,boxstyle='round,pad=0,rounding_size=10',fc=fc,ec=ec,lw=1.6))
    ax.text((x0+x1)/2,(y0+y1)/2,txt,ha='center',va='center',fontsize=fs,linespacing=ls,color='black')
def arr(p0,p1,c,lw=1.6):
    ax.add_patch(FancyArrowPatch(p0,p1,arrowstyle='-|>',mutation_scale=11,color=c,lw=lw,shrinkA=0,shrinkB=0))
box(152,20,541,90,'#f0f0f0','#555555',r'RWIP state: $\theta,\ \dot\theta,\ \phi,\ \dot\phi$')
box(13,148,202,333,'#dbe3f1',BL,'PSO-FLC\nbaseline\n(Section 2,\nfixed)')
box(221,148,680,297,'#dbe3f1',BL,'Residual actor $\\pi(s)$\n(neural network, 2 hidden\nlayers, 256/layer)')
box(221,342,680,490,'#fdeede',OR,'Stage A \u2014 Behavior Cloning\nRegression on (obs, action)\nfrom SMC-ESO (ZOH)')
box(221,535,680,683,'#fdeede',OR,'Stage B \u2014 TD3 fine-tuning\n18 episodes, actor LR $10^{-5}$\n$\\|\\Delta\\theta\\|_2$ = '+rng+' vs. Stage A')
box(221,727,680,913,'#fbdde0',RD,'Stage C \u2014 Bias correction\n(post hoc)\n$a_{\\mathrm{bias}}=\\pi(0)$\n$\\pi^{\\prime}(s)=\\pi(s)-a_{\\mathrm{bias}}$',ls=1.3)
box(242,958,659,1068,'#dbe3f1',BL,'$u_{corr}=u_{corr,\\mathrm{max}}\\cdot a,\\ \\ a\\in[-1,\\ 1]$\n$u_{corr,\\mathrm{max}}=10$ V',fs=11)
box(27,1222,666,1296,'#dbe3f1',BL,'$u_{tot}=\\mathrm{sat}(u_{FLC}+u_{corr},\\ \\pm24$ V$)$',fs=11)
ax.plot([347,347],[90,122],c='k',lw=1.6); ax.plot([105,450],[122,122],c='k',lw=1.6)
arr((105,122),(105,148),'k'); arr((450,122),(450,148),'k')
for y0,y1 in [(297,342),(490,535),(683,727)]: arr((450,y0),(450,y1),OR)
arr((450,913),(450,958),RD)
ax.plot([107,107],[333,1141],c=BL,lw=1.6); arr((107,1141),(107,1141.5),BL)
ax.add_patch(Circle((105,1158),17,fc='white',ec='k',lw=1.6)); ax.text(105,1158,'+',ha='center',va='center',fontsize=12)
arr((105,1175),(105,1222),BL)
ax.plot([450,450],[1068,1158],c=RD,lw=1.6); arr((450,1158),(124,1158),RD)
ax.text(148,745,'$u_{FLC}$',color=BL,fontsize=11,ha='center'); ax.text(492,1112,'$u_{corr}$',color=RD,fontsize=11,ha='center')
fig.savefig(out,dpi=400); print('saved',out)
