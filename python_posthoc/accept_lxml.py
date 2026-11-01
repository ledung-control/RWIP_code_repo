"""accept_lxml.py in.docx out.docx -- accept all tracked changes (ins/del only) without touching math or drawings."""
import sys,zipfile
from lxml import etree
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'; q=lambda t:'{%s}%s'%(W,t)
zi=zipfile.ZipFile(sys.argv[1]); zo=zipfile.ZipFile(sys.argv[2],'w',zipfile.ZIP_DEFLATED)
for it in zi.infolist():
    d=zi.read(it.filename)
    if it.filename=='word/document.xml':
        r=etree.fromstring(d)
        for e in list(r.iter(q('del'))):
            p=e.getparent()
            if p.tag==q('rPr'): p.remove(e)       # deleted paragraph mark (none expected)
            else: p.remove(e)
        for e in list(r.iter(q('ins'))):
            p=e.getparent()
            if p.tag==q('rPr'): p.remove(e); continue
            i=p.index(e)
            for c in list(e): p.insert(i,c); i+=1
            p.remove(e)
        d=etree.tostring(r,xml_declaration=True,encoding='UTF-8',standalone=True)
    zo.writestr(it,d)
