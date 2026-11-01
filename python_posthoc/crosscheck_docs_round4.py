"""crosscheck_docs_round4.py -- document-level consistency checks for the Round 4 file set.
usage: python3 crosscheck_docs_round4.py <dir with the Round 4 files> <cum.log>  -> results/crosscheck_docs_round4.csv"""
import sys, os, re, subprocess, zipfile, hashlib, csv, glob
D=sys.argv[1]; CUM=sys.argv[2]; HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,'results','crosscheck_docs_round4.csv')
def txt(f,extra=()): return subprocess.run(['pandoc',os.path.join(D,f),'-t','plain','--wrap=none',*extra],capture_output=True,text=True).stdout
def rd(f): return open(os.path.join(D,f),encoding='utf-8-sig').read()
rows=[]
def chk(i,what,ok,detail=''): rows.append([i,what,'PASS' if ok else 'FAIL',detail])
clean=txt('RWIP_Round4_WORKING_CLEAN.docx'); acc=txt('RWIP_Round4_WORKING_TRACKED.docx',('--track-changes=accept',))
pc=[p.strip() for p in clean.split('\n\n') if p.strip()]; pa=[p.strip() for p in acc.split('\n\n') if p.strip()]
eqp=lambda p: len(p)<300 and bool(re.search(r'\(\d+\)\s*$',p))
dif=[p for p in set(pc)^set(pa) if not eqp(p)]
chk('D1','tracked file with changes accepted == clean file (non-empty paragraphs; equation-number paragraphs excluded because LibreOffice re-renders equations)',not dif,'%d vs %d paragraphs; %d non-equation differences'%(len(pc),len(pa),len(dif)))
# D2 figure
def doc_xml(f): return zipfile.ZipFile(os.path.join(D,f)).read('word/document.xml').decode('utf8')
def rels(f): return zipfile.ZipFile(os.path.join(D,f)).read('word/_rels/document.xml.rels').decode('utf8')
tx=doc_xml('RWIP_Round4_WORKING_TRACKED.docx')
ORIG=sys.argv[3]
def media(f): z=zipfile.ZipFile(f); return {n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('word/media/')}
fig=os.path.join(D,'Figure4_round4.png'); hfig=hashlib.sha256(open(fig,'rb').read()).hexdigest()
hold=media(ORIG)['word/media/image5.png']
mc=media(os.path.join(D,'RWIP_Round4_WORKING_CLEAN.docx')); mt=media(os.path.join(D,'RWIP_Round4_WORKING_TRACKED.docx'))
chk('D2a','clean: the redrawn Figure 4 is embedded, byte-identical to Figure4_round4.png',hfig in mc.values())
chk('D2b','clean: the old Figure 4 raster (original image5.png) is not in the file',hold not in mc.values())
chk('D2c','tracked: both the old and the new Figure 4 are present, the old one in a tracked deletion and the new one in a tracked insertion',hfig in mt.values() and hold in mt.values() and bool(re.search(r'<w:del [^>]*>(?:(?!</w:del>).)*<w:drawing>',tx,re.S)) and bool(re.search(r'<w:ins [^>]*>(?:(?!</w:ins>).)*<w:drawing>',tx,re.S)))
# D3 strings
alltxt={f:txt(f) for f in sorted(os.listdir(D)) if f.endswith('.docx') and f.startswith(('RWIP_Round4','Round4'))}
for f,t in alltxt.items():
    chk('D3-0.111-'+f[:24],'%s: no "0.111" (0.110 is the recomputed value)'%f,'0.111' not in t.replace('0.1110','').replace('0.1111','') or '0.111' not in re.sub(r'\d\.\d*0\.111\d*','',t))
for f in ('RWIP_Round4_WORKING_CLEAN.docx','RWIP_Round4_WORKING_TRACKED.docx'):
    t=alltxt[f] if f in alltxt else ''
    chk('D3-prereg-'+f[11:20],'%s: no "pre-registered"/"pre-specified"/"fixed before training"'%f,not re.search(r'pre-?registered|pre-?specified|fixed before (training|optimi)',re.sub(r'\[AUTHOR CONFIRM[^\]]*\]','',acc if 'TRACKED' in f else clean),re.I))
chk('D3-disp','accepted manuscript text: "displacement ≈ 0.36 units" absent (0.36 appears only as a nominal budget)','displacement ≈ 0.36 units' not in clean and '0.36 was used during design only as a nominal budget' in clean)
chk('D3-S19','no document still says "S1–S9" as the supplementary range',all(not re.search(r'S1[–-]S9\b',t) for t in alltxt.values()))
chk('D4a','manuscript carries exactly 7 [AUTHOR CONFIRM] markers',len(re.findall(r'\[AUTHOR CONFIRM',clean))==7,str(len(re.findall(r'\[AUTHOR CONFIRM',clean))))
for f,t in alltxt.items():
    if any(k in f for k in ('Response','Matrix','Decisions','Experiment','Pages','PostHoc')):
        chk('D4b-'+f[7:20],'%s states NOT READY'%f,bool(re.search(r'NOT READY|CHƯA SẴN SÀNG|chưa sẵn sàng',t,re.I)) )
        chk('D4c-'+f[7:20],'%s never says "ready for submission" without NOT'%f,not re.search(r'(?<!NOT )(?<!not )(?<!chưa )(READY FOR SUBMISSION|ready to submit)',t))
# D5 numbering
cap_t=[int(x) for x in re.findall(r'^Table (\d+)\.',clean,re.M)]; cap_f=[int(x) for x in re.findall(r'^Figure\.? ?(\d+)',clean,re.M)]
chk('D5a','Table captions 1..11 contiguous',sorted(cap_t)==list(range(1,12)),str(sorted(cap_t)))
chk('D5b','Figure captions 1..7 contiguous',sorted(cap_f)==list(range(1,8)),str(sorted(cap_f)))
body=re.sub(r'^(Table \d+\.|Figure\.? ?\d+).*$','',clean,flags=re.M)
chk('D5c','every table is cited in the text',all(re.search(r'Tables? [^.\n]*\b%d\b'%n,body) for n in range(1,12)))
chk('D5d','every figure is cited in the text',all(re.search(r'Fig(ure|s)?\.? ?[^.\n]*\b%d\b'%n,body) for n in range(1,8)))
# D6 supplementary refs
ph=rd('Round4_PostHoc_Report.md') if os.path.exists(os.path.join(D,'Round4_PostHoc_Report.md')) else ''
sup_heads=set(re.findall(r'^## Table (S\d+)\.',ph,re.M))
cited=set(re.findall(r'\bS(\d+)\b',' '.join(re.findall(r'[Tt]ables? S\d+(?:[^.;)]*?S\d+)*',clean))))
cited={'S'+c for c in cited}
chk('D6','supplementary tables cited in the manuscript exist in the post-hoc report',cited<=sup_heads,'cited %s; defined %s'%(sorted(cited),sorted(sup_heads)))
chk('D6b','Table S5 and Table S10 are cited in the manuscript',{'S5','S10'}<=cited)
# D7 section refs
secs=set(re.findall(r'^(\d(?:\.\d){0,2})\.? [A-Z]',clean,re.M))
heads=set(re.findall(r'^(\d(?:\.\d){0,2})\.? ',clean,re.M))
for f in ('Round4_Response_to_Reviewers.docx','Round4_Matrix.docx','Round4_Author_Decisions.docx'):
    t=alltxt[f]; refs=set(re.findall(r'(?:Sections?|Mục|Secs?\.)\s+(\d\.\d(?:\.\d)?)',t))
    miss=sorted(r for r in refs if r not in heads and not any(h.startswith(r) for h in heads))
    chk('D7-'+f[7:18],'%s: section numbers cited exist in the manuscript'%f,not miss,'missing: %s'%miss)
# D8 numbers in letter exist in manuscript/report
ref_txt=clean+'\n'+ph
letter=alltxt['Round4_Response_to_Reviewers.docx']
toks=sorted(set(re.findall(r'(?<![\w.])\d+\.\d{2,}(?![\w.])',letter)))
csvt=open(os.path.join(HERE,'results','crosscheck_round4.csv'),encoding='utf-8-sig').read()
refnums=[(float(x),len(x.split('.')[1])) for x in set(re.findall(r'\d+\.\d+',ref_txt+' '+csvt))]
def found(t):
    if t in ref_txt or t.lstrip('0') in ref_txt: return True
    v=float(t); d=len(t.split('.')[1]); return any(abs(abs(v)-r)<=0.5*10**-d+1e-12 for r,_ in refnums)
nf=[t for t in toks if not found(t)]
chk('D8','decimal numbers (≥2 decimals) in the letter occur in the manuscript or post-hoc report',len(nf)<=0,'%d checked; not found: %s'%(len(toks),nf))
# D9 matrix/letter codes
codes=['R1-%d'%i for i in range(1,12)]+['R2-%d'%i for i in range(1,7)]
mt=rd('Round4_Matrix.csv')
chk('D9a','matrix lists each of 17 reviewer codes once',all(len(re.findall(r'^%s,'%c,mt,re.M))==1 for c in codes))
chk('D9b','letter answers each of the 17 codes',all(re.search(r'\*?\*?%s'%c,letter) for c in codes))
dec=alltxt['Round4_Author_Decisions.docx']
refs=set(re.findall(r'\b([AB]\d)\b',mt)); ex=set('A%d'%i for i in range(1,9))|set('B%d'%i for i in range(1,10))
chk('D9c','decision codes cited in the matrix exist in the decisions list',refs<=ex and all(re.search(r'\b%s\b'%c,dec) for c in ex),str(sorted(refs-ex)))
# D10 pages doc arithmetic
pg=alltxt['Round4_IJIES_Pages_and_Supplementary.docx']
cum=[tuple(l.split()[-1:]) for l in open(CUM) if re.match(r'\d+ ',l)]
cum_pages=[int(l.split()[-1]) for l in open(CUM) if re.match(r'\d+ ',l)]
fees=[50*max(0,p-10) for p in (22,19,17,14,12,10)]
chk('D10a','fee table arithmetic (22,19,17,14,12,10 pages -> %s)'%fees,fees==[600,450,350,200,100,0])
rowsmd=re.findall(r'^\| (\d+) \| \+? ?[^|]+\| (\d+)\*? \| (\d+) \|',rd('Round4_IJIES_Pages_and_Supplementary.md') if os.path.exists(os.path.join(D,'Round4_IJIES_Pages_and_Supplementary.md')) else '',re.M)
ok=len(rowsmd)>=16 and all(int(p)==cum_pages[int(n)-1] and int(f)==50*max(0,int(p)-10) for n,p,f in rowsmd if int(n)>=1)
chk('D10b','Plan-10 table page counts equal the measured log and fees = 50 x (pages-10)',ok,'%d rows'%len(rowsmd))
with open(OUT,'w',newline='',encoding='utf8') as f:
    w=csv.writer(f); w.writerow(['id','check','status','detail']); w.writerows(rows)
bad=[r for r in rows if r[2]!='PASS']
print(len(rows),'checks;',len(bad),'FAIL')
for r in bad: print(r)
