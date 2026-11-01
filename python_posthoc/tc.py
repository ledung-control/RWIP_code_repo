"""tc.py -- tiny tracked-changes editor for word/document.xml (lxml).
Mini-markup in NEW text: *italic*  ~subscript~  ^superscript^   (no nesting)."""
import copy, re
from lxml import etree

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
XML = 'http://www.w3.org/XML/1998/namespace'
def q(t): return '{%s}%s' % (W, t)

AUTHOR = 'Round2 draft (authors to verify)'
DATE = '2026-10-25T12:00:00Z'


class Doc:
    def __init__(self, path):
        self.path = path
        self.tree = etree.parse(path)
        self.root = self.tree.getroot()
        self.nid = 100000
        # normalise: drop lastRenderedPageBreak, merge multiple w:t per run
        for e in self.root.iter(q('lastRenderedPageBreak')):
            e.getparent().remove(e)
        for r in self.root.iter(q('r')):
            ts = r.findall(q('t'))
            if len(ts) > 1:
                ts[0].text = ''.join(t.text or '' for t in ts)
                ts[0].set('{%s}space' % XML, 'preserve')
                for t in ts[1:]:
                    r.remove(t)
        self.log = []

    def _id(self):
        self.nid += 1
        return str(self.nid)

    # ---- helpers
    @staticmethod
    def runs(p):
        out = []
        for r in p:
            if r.tag == q('r') and r.find(q('t')) is not None:
                out.append(r)
        return out

    def ptext(self, p):
        return ''.join(r.find(q('t')).text or '' for r in self.runs(p))

    def paras(self):
        return list(self.root.iter(q('p')))

    def find(self, old, nth=0):
        hits = [p for p in self.paras() if old in self.ptext(p)]
        if not hits:
            raise KeyError('NOT FOUND: ' + old[:80])
        if len(hits) > 1:
            print('WARNING multiple hits (%d) for: %s' % (len(hits), old[:60]))
        if len(hits) > 1 and nth == 'unique':
            raise KeyError('AMBIGUOUS: ' + old[:80])
        return hits[nth if isinstance(nth, int) else 0]

    def _split(self, p, pos):
        """ensure a run boundary at char offset pos in paragraph p"""
        acc = 0
        for r in self.runs(p):
            t = r.find(q('t'))
            L = len(t.text or '')
            if acc < pos < acc + L:
                k = pos - acc
                r2 = copy.deepcopy(r)
                t.text = t.text[:k]
                t.set('{%s}space' % XML, 'preserve')
                t2 = r2.find(q('t'))
                t2.text = t2.text[k:]
                t2.set('{%s}space' % XML, 'preserve')
                r.addnext(r2)
                return
            acc += L

    def _mkrun(self, base, text):
        """runs from mini-markup, base = rPr element or None"""
        res = []
        parts = re.split(r'(\*[^*]+\*|~[^~]+~|\^[^^]+\^)', text)
        for part in parts:
            if not part:
                continue
            style = None
            if part[0] == '*' and part[-1] == '*' and len(part) > 2:
                style, part = 'i', part[1:-1]
            elif part[0] == '~' and part[-1] == '~' and len(part) > 2:
                style, part = 'sub', part[1:-1]
            elif part[0] == '^' and part[-1] == '^' and len(part) > 2:
                style, part = 'sup', part[1:-1]
            r = etree.Element(q('r'))
            rpr = copy.deepcopy(base) if base is not None else etree.Element(q('rPr'))
            for tag in ('i', 'iCs', 'vertAlign', 'b', 'bCs'):
                for e in rpr.findall(q(tag)):
                    rpr.remove(e)
            if style == 'i':
                etree.SubElement(rpr, q('i')); etree.SubElement(rpr, q('iCs'))
            elif style in ('sub', 'sup'):
                va = etree.SubElement(rpr, q('vertAlign'))
                va.set(q('val'), 'subscript' if style == 'sub' else 'superscript')
            # schema order: put rFonts first, others after -- keep simple/valid ordering
            order = ['rStyle', 'rFonts', 'b', 'bCs', 'i', 'iCs', 'color', 'sz', 'szCs', 'vertAlign']
            kids = sorted(list(rpr), key=lambda e: order.index(etree.QName(e).localname)
                          if etree.QName(e).localname in order else 99)
            for k in list(rpr):
                rpr.remove(k)
            for k in kids:
                rpr.append(k)
            if len(rpr):
                r.append(rpr)
            t = etree.SubElement(r, q('t'))
            t.text = part
            t.set('{%s}space' % XML, 'preserve')
            res.append(r)
        return res

    def _ins(self, base, text):
        ins = etree.Element(q('ins'))
        ins.set(q('id'), self._id()); ins.set(q('author'), AUTHOR); ins.set(q('date'), DATE)
        for r in self._mkrun(base, text):
            ins.append(r)
        return ins

    # ---- operations
    def replace(self, old, new, nth=0, para_hint=None):
        p = self.find(para_hint or old, nth) if para_hint else self.find(old, nth)
        txt = self.ptext(p)
        i = txt.find(old)
        if i < 0:
            raise KeyError('old not in hinted paragraph: ' + old[:60])
        self._split(p, i)
        self._split(p, i + len(old))
        acc = 0
        target = []
        for r in self.runs(p):
            L = len(r.find(q('t')).text or '')
            if acc >= i and acc + L <= i + len(old) and L > 0:
                target.append(r)
            acc += L
        assert ''.join(r.find(q('t')).text for r in target) == old, (old[:50], [r.find(q('t')).text for r in target])
        base = target[0].find(q('rPr'))
        dl = etree.Element(q('del'))
        dl.set(q('id'), self._id()); dl.set(q('author'), AUTHOR); dl.set(q('date'), DATE)
        target[0].addprevious(dl)
        for r in target:
            t = r.find(q('t'))
            t.tag = q('delText')
            dl.append(r)
        if new:
            dl.addnext(self._ins(base, new))
        self.log.append(('replace', old[:60], new[:60]))

    def insert_after(self, anchor, new, nth=0):
        """tracked insertion right after the anchor text (anchor must end at desired position)"""
        p = self.find(anchor, nth)
        txt = self.ptext(p)
        i = txt.find(anchor) + len(anchor)
        self._split(p, i)
        acc = 0
        last = None
        base = None
        for r in self.runs(p):
            L = len(r.find(q('t')).text or '')
            acc += L
            if acc == i:
                last = r
                base = r.find(q('rPr'))
                break
        last.addnext(self._ins(base, new))
        self.log.append(('insert', anchor[-40:], new[:60]))

    def new_para_after(self, anchor, text, nth=0, like=None):
        """tracked new paragraph after the paragraph containing `anchor`; formatting copied from `like` (or anchor) paragraph"""
        pa = self.find(anchor, nth) if isinstance(anchor, str) else anchor
        pl = (self.find(like) if isinstance(like, str) else like) if like is not None else pa
        np_ = etree.Element(q('p'))
        ppr = copy.deepcopy(pl.find(q('pPr'))) if pl.find(q('pPr')) is not None else etree.Element(q('pPr'))
        rp = ppr.find(q('rPr'))
        if rp is None:
            rp = etree.Element(q('rPr'))
            tail = [e for e in ppr if etree.QName(e).localname in ('sectPr', 'pPrChange')]
            if tail:
                tail[0].addprevious(rp)
            else:
                ppr.append(rp)
        for e in ppr.findall(q('sectPr')):   # a new paragraph must not carry the section break
            ppr.remove(e)
        m = etree.Element(q('ins'))
        m.set(q('id'), self._id()); m.set(q('author'), AUTHOR); m.set(q('date'), DATE)
        rp.insert(0, m)
        np_.append(ppr)
        rs = self.runs(pl)
        base = rs[0].find(q('rPr')) if rs else None
        np_.append(self._ins(base, text))
        pa.addnext(np_)
        self.log.append(('newpara', str(anchor)[:40], text[:60]))
        return np_

    def save(self, path=None):
        self.tree.write(path or self.path, xml_declaration=True, encoding='UTF-8', standalone=True)
