import sys
sys.path.insert(0,'/home/claude/ms5')
import tc
tc.AUTHOR='Round5 working draft (authors to verify)'
from tc import Doc
d=Doc('/home/claude/ms5/x/word/document.xml'); d.nid=300000

import copy
from lxml import etree
q=tc.q
def _mk_ins_like(src):
    n=etree.Element(q('ins')); n.set(q('id'),d._id()); n.set(q('author'),src.get(q('author'))); n.set(q('date'),src.get(q('date'))); return n
def _find_run_in_ins(old):
    for r in d.root.iter(q('r')):
        t=r.find(q('t'))
        if t is not None and t.text and old in t.text and r.getparent().tag==q('ins'):
            return r,t
    return None,None
def edit(old,new,mode='replace'):
    """mode replace: tracked deletion of old + insertion of new; after: insert new after old"""
    r,t=_find_run_in_ins(old)
    if r is None:
        return d.replace(old,new) if mode=='replace' else d.insert_after(old,new)
    ins=r.getparent(); k=t.text.index(old); a,b,c=t.text[:k],old,t.text[k+len(old):]
    def mk(txt):
        x=copy.deepcopy(r); tt=x.find(q('t')); tt.text=txt; tt.set('{http://www.w3.org/XML/1998/namespace}space','preserve'); return x
    kids=list(ins); idx=kids.index(r)
    before=kids[:idx]; after=kids[idx+1:]
    ins_a=_mk_ins_like(ins); ins_a.extend(before+([mk(a)] if a else []))
    mid=_mk_ins_like(ins); rb=mk(b)
    if mode=='replace':
        dl=etree.SubElement(mid,q('del')); dl.set(q('id'),d._id()); dl.set(q('author'),tc.AUTHOR); dl.set(q('date'),tc.DATE)
        tt=rb.find(q('t')); tt.tag=q('delText'); dl.append(rb)
    else:
        mid.append(rb)
    nw=d._ins(r.find(q('rPr')),new)
    ins_b=_mk_ins_like(ins); ins_b.extend(([mk(c)] if c else [])+after)
    parent=ins.getparent(); pos=parent.index(ins); parent.remove(ins)
    seq=[x for x in (ins_a,mid,nw,ins_b) if len(x) or x is nw]
    for j,x in enumerate(seq): parent.insert(pos+j,x)
    d.log.append(('ins-edit',old[:40],new[:60]))

R=edit
I=lambda a,n: edit(a,n,'after')

# 4.1 mechanism: untested -> limited new evidence (N)
R("and a plausible but untested explanation is the following:",
  "and a plausible explanation, partly examined in Round 5 (see the last sentences of this section), is the following:")
R("no isolating experiment (an ESO-bandwidth sweep, or an oracle-gain comparator that supplies the correct Kt to the sliding mode law) was run in this study, so the mechanism is not confirmed.",
  "an ESO-bandwidth sweep with an oracle-gain comparator that supplies the true control-affine gain to the sliding-mode law was run in the same simulator (ω0 in {10, 15, 20, 30, 40, 60} rad/s, 60 s, θ0 = 0.10 rad, deterministic; supplementary report, Tables S12 and S13). With the nominal gain and ω0 = 20 rad/s the stored Table 5 values are reproduced (nominal IAEθ 0.1369; Kt −30%: 5.440; adverse combination: 7.237). With the oracle gain at ω0 = 20 rad/s the Kt −30% value falls to 0.1271, so in this design the gain inversion, rather than the observer bandwidth alone, accounts for that degradation; for the adverse combination (Kt and Ip both scaled by 0.7, so Kt/Ip is unchanged) the oracle gain changes nothing (7.237), and that degradation is therefore not due to the gain error. The nominal-condition IAEθ also depends strongly on ω0 (0.1369 at 20, 0.4985 at 30, 4.006 at 15 rad/s), and at ω0 ≥ 30 rad/s the control effort and chatter grow, so these results are specific to this design, this grid and this single initial condition and do not establish a general property of sliding-mode control.")
# Limitations
R("an ESO-bandwidth sweep with an oracle-gain comparator, and distributions of initial states, noise and model mismatch) were not run.",
  "and distributions of initial states, noise and model mismatch) were not run; the ESO-bandwidth sweep with an oracle-gain comparator was run in Round 5 only for this plant and grid (Section 4.1).")
# 3.9 MATLAB reproduction
I("reproducing the nominal metrics of Tables 4, 9 and 10 to within ≤0.1%.",
  " The ten policies (raw and π(0)-removed) were additionally re-evaluated at 60 s and 600 s with the authors' MATLAB evaluator (MATLAB R2026a); the 600 s medians over the five seeds reproduce Table 11 (IAEϕ 89.63 and 1144.59 for the raw BC-only and BC+TD3 policies, 4.22 and 5.55 after π(0) removal) and agree with the stored MATLAB 60 s results of the authors' evaluation. This is a reproduction of stored results, not a new experiment.")
# Data availability
I("Fuzzy Logic, Reinforcement Learning and Deep Learning toolboxes)",
  " — in Round 5 the baselines were re-run in release 26.1.0.3276743 (R2026a, Update 3) with toolbox version 26.1")
d.save()
print('\n'.join('%s | %s | %s'%x for x in d.log)); print(len(d.log),'edits')
