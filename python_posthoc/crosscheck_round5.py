import csv,re,sys,statistics as st
R='/home/claude/r5/results/'
e=list(csv.DictReader(open(R+'E7E8_MATLAB_2026a.csv')))
def g(exp,w,case): 
    r=[x for x in e if x['Experiment']==exp and float(x['w0'])==w and x['Case']==case][0]; return float(r['IAEtheta'])
b=list(csv.DictReader(open(R+'baseline_B_MATLAB_2026a.csv')))
def med(grp,var,tf,col):
    v=[float(x[col]) for x in b if x['Controller'].startswith(grp) and x['Variant']==var and float(x['Tf'])==tf]
    return st.median(v),len(v)
# (label, computed value, tolerance-format, [files to find in])
checks=[]
def add(label,val,fmt,*texts): checks.append((label,fmt%val,texts))
ms=open('/home/claude/ms5/clean5.txt',encoding='utf8').read()
lt=open('/home/claude/r5/out/Round5_Response_to_Reviewers.md',encoding='utf8').read()
for lab,exp,w,case,fmt in [('nominal ω0=20 nominal gain','E7_nominal_gain',20,'nominal','%.4f'),('Kt-30 nominal gain','E7_nominal_gain',20,'Kt -30%','%.3f'),('adverse nominal gain','E7_nominal_gain',20,'adverse (mgl+30,Ip-30,Kt-30)','%.3f'),('Kt-30 oracle','E8_oracle_gain',20,'Kt -30%','%.4f'),('adverse oracle','E8_oracle_gain',20,'adverse (mgl+30,Ip-30,Kt-30)','%.3f'),('nominal ω0=30','E7_nominal_gain',30,'nominal','%.4f'),('nominal ω0=15','E7_nominal_gain',15,'nominal','%.3f')]:
    try: v=g(exp,w,case)
    except IndexError:
        print('case name not found',lab,set(x['Case'] for x in e)); sys.exit(1)
    add(lab,v,fmt,ms,lt)
for grp,var,tf,col,fmt,lab in [('BC-only','raw',600,'IAEphi','%.2f','BC-only raw 600 IAEphi'),('BC+TD3','raw',600,'IAEphi','%.2f','BC+TD3 raw 600'),('BC-only','unbiased',600,'IAEphi','%.2f','BC-only unb 600'),('BC+TD3','unbiased',600,'IAEphi','%.2f','BC+TD3 unb 600')]:
    try: v,n=med(grp,var,tf,col)
    except Exception as ex: print('skip',lab,ex); continue
    add(lab+' (n=%d)'%n,v,fmt,ms,lt)
bad=0
out=['check,computed,in_manuscript,in_letter']
for lab,s,texts in checks:
    flags=[s in t for t in texts]
    out.append('%s,%s,%s,%s'%(lab,s,flags[0],flags[1]))
    if not all(flags): bad+=1
open('/home/claude/r5/out/crosscheck_round5.csv','w').write('\n'.join(out)); print('\n'.join(out)); print('missing:',bad)
