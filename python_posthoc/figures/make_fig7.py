"""make_fig7.py -- Fig. 7 (17 cm, two plots, no panel letters). Reads results/fig7_traj.pkl
(made by make_fig7_traj.py). Output: Fig7.pdf, Fig7_600dpi.png."""
import os, pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
W_CM, FS = 17.0, 11.0
plt.rcParams.update({"font.family":"serif","font.serif":["Liberation Serif","DejaVu Serif"],"mathtext.fontset":"stix",
 "font.size":FS,"axes.labelsize":FS,"xtick.labelsize":FS,"ytick.labelsize":FS,"legend.fontsize":FS,"axes.linewidth":0.8})
T=pickle.load(open(os.path.join(HERE,"..","results","fig7_traj.pkl"),"rb"))
BL,OR,GR,MU,GRID="#1f6fb2","#d55e00","#0b7a5e","#555555","#CCCCCC"
w=W_CM/2.54
fig,(a1,a2)=plt.subplots(1,2,figsize=(w,w*0.46))
for s in range(1,6):
    for tag,c,ls,lab,lw in [("raw",BL,"-","BC+TD3 raw",1.1),("bco",OR,(0,(5,2)),"BC-only",1.0),("unb",GR,(0,(1,1.4)),r"BC+TD3 minus $\pi(0)$",1.3)]:
        d=T[(s,tag)]; a1.plot(d["t"],d["phi"],color=c,lw=lw,ls=ls,alpha=.9,label=lab if s==1 else None)
a1.axhline(0,color=GRID,lw=.7,zorder=0)
a1.set_xlabel("Time $t$ (s)"); a1.set_ylabel(r"Flywheel angle $\phi$ (rad)")
a1.set_xlim(0,600); a1.set_ylim(-6.2,5.2); a1.grid(True,color=GRID,lw=.6); a1.set_axisbelow(True)
a1.annotate(r"s3: $\phi_{ss}$ = $-$5.34 rad",xy=(520,-5.343),xytext=(250,-4.0),arrowprops=dict(arrowstyle="-",color=MU,lw=.6))
a1.legend(frameon=False,loc="upper left",handlelength=2.2)
tg=T[(1,"raw")]["Tg"]
for tag,c,ls,lab in [("raw",BL,"-","BC+TD3 raw"),("bco",OR,(0,(5,2)),"BC-only"),("unb",GR,(0,(1,1.4)),r"BC+TD3 minus $\pi(0)$")]:
    m=np.median(np.array([T[(s,tag)]["cum"] for s in range(1,6)]),axis=0); a2.plot(tg,m,color=c,lw=1.5,ls=ls,label=lab)
a2.plot(tg,np.full_like(tg,3.211,dtype=float),color=MU,lw=1.0,ls=(0,(2,2)),label="PSO-FLC baseline")
a2.axvline(60,color=MU,lw=.7,ls=(0,(1,2)))
a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xlim(5,1150); a2.set_ylim(1,6000)
a2.set_xlabel("Evaluation horizon $T$ (s)"); a2.set_ylabel(r"Accumulated IAE$_\phi$ (rad$\cdot$s)")
a2.grid(True,color=GRID,lw=.6,which="both"); a2.set_axisbelow(True); a2.legend(frameon=False,loc="upper left",handlelength=2.2)
for a in (a1,a2): a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
fig.subplots_adjust(left=.085,right=.99,top=.98,bottom=.17,wspace=.28)
fig.savefig(os.path.join(HERE,"Fig7.pdf"),facecolor="white"); fig.savefig(os.path.join(HERE,"Fig7_600dpi.png"),dpi=600,facecolor="white")
