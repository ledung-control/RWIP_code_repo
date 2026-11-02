"""make_fig6.py -- Fig. 6 (17 cm, two plots, no panel letters). Reads results/runs_600s.csv and
results/bias_injection_sweep.csv. Output: Fig6.pdf, Fig6_600dpi.png (run from this folder)."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, '..', 'results')
W_CM, FS = 17.0, 11.0
plt.rcParams.update({"font.family":"serif","font.serif":["Liberation Serif","DejaVu Serif"],"mathtext.fontset":"stix",
 "font.size":FS,"axes.labelsize":FS,"xtick.labelsize":FS,"ytick.labelsize":FS,"legend.fontsize":FS,"axes.linewidth":0.8,"grid.linewidth":0.6})
R=pd.read_csv(os.path.join(RES,"runs_600s.csv")); R=R[(R.policy=="td3")&(R.variant=="raw")].sort_values("seed")
x=np.abs(R.aBias.values); y=R.IAEphi_60.values
B=pd.read_csv(os.path.join(RES,"bias_injection_sweep.csv")).sort_values("b")
w=W_CM/2.54
fig,(a1,a2)=plt.subplots(1,2,figsize=(w,w*0.44))
a1.scatter(x,y,s=60,color="#1f6fb2",zorder=3,label="5 independent seeds")
for xi,yi,s in zip(x,y,R.seed): a1.annotate("s%d"%s,(xi,yi),xytext=(6,5),textcoords="offset points",fontsize=FS)
p=np.polyfit(x,y,1); r=np.corrcoef(x,y)[0,1]; xx=np.linspace(0,0.3,50)
a1.plot(xx,np.polyval(p,xx),"--",color="#c0392b",lw=1.4,label="Regression ($r$ = %+.3f)"%r)
a1.set_xlim(0,0.31); a1.set_ylim(0,340)
a1.set_xlabel(r"$|a_{bias}|=|\pi(0)|$ (normalized)"); a1.set_ylabel(r"Nominal IAE$_\phi$ (raw)")
a1.grid(True,color="#CCCCCC"); a1.set_axisbelow(True); a1.legend(loc="upper left",frameon=False,handlelength=1.8)
hi=x.max()
a2.axvspan(-hi,hi,color="#f7dfa5",alpha=0.85,zorder=0,label=r"Measured $|a_{bias}|$ range (5 seeds)")
a2.plot(B.b,B.IAEphi_60,"-o",color="#1f6fb2",ms=5,lw=1.5,label="Synthetic bias sweep",zorder=3)
a2.set_xlim(-0.32,0.32); a2.set_ylim(0,185)
a2.set_xlabel(r"$b$ = injected bias (normalized, [$-$1, 1])"); a2.set_ylabel(r"IAE$_\phi$ (60 s, PSO-FLC baseline)")
a2.grid(True,color="#CCCCCC"); a2.set_axisbelow(True); a2.legend(loc="upper center",frameon=False,handlelength=1.6)
for a in (a1,a2): a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
fig.subplots_adjust(left=0.085,right=0.99,top=0.98,bottom=0.17,wspace=0.28)
fig.savefig(os.path.join(HERE,"Fig6.pdf"),facecolor="white"); fig.savefig(os.path.join(HERE,"Fig6_600dpi.png"),dpi=600,facecolor="white")
