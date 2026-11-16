"""Build deterministic, original Evater practice PDFs. No bank or database writes."""
import json, hashlib
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT=Path(__file__).resolve().parents[1]
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
pdfmetrics.registerFont(TTFont('DejaVu',FONT));pdfmetrics.registerFont(TTFont('DejaVuBold',FONT.replace('Sans.ttf','Sans-Bold.ttf')))
S=getSampleStyleSheet();S.add(ParagraphStyle(name='Title2',fontName='DejaVuBold',fontSize=21,leading=28,textColor=colors.HexColor('#143e38'),spaceAfter=16))
S.add(ParagraphStyle(name='Body2',fontName='DejaVu',fontSize=10,leading=15,spaceAfter=7));S.add(ParagraphStyle(name='Small2',fontName='DejaVu',fontSize=8,leading=12,textColor=colors.HexColor('#555555'),spaceAfter=8));S.add(ParagraphStyle(name='Head2',fontName='DejaVuBold',fontSize=13,leading=19,spaceAfter=12))
def para(t,style='Body2'):return Paragraph(escape(t).replace('\n','<br/>'),S[style])
def page(c,d):
 c.saveState();c.setFillColor(colors.HexColor('#1b6257'));c.setFillAlpha(.055);c.translate(298,420);c.rotate(38);c.setFont('DejaVuBold',54);c.drawCentredString(0,0,'EVATER');c.restoreState()
 c.saveState();c.setStrokeColor(colors.HexColor('#d7e5e1'));c.line(44,46,551,46);c.setFillColor(colors.HexColor('#555555'));c.setFont('DejaVu',8);c.drawString(44,32,'Evater • Original practice paper • evater.xyz');c.drawRightString(551,32,f'Page {d.page}');c.restoreState()
def build(p):
 out=ROOT/'frontend/public'/p['filename'];doc=SimpleDocTemplate(str(out),pagesize=(595.28,841.89),leftMargin=46,rightMargin=46,topMargin=42,bottomMargin=62,title='Class 8 '+p['title']+' - Evater sample paper',author='Evater',pageCompression=1,invariant=1)
 story=[para('CLASS 8 · '+p['subject'].upper(),'Small2'),para(p['title'],'Title2'),para('Sample paper · 15 questions · 22 marks','Head2'),para('10 objective questions × 1 mark. Five written-response questions: 12 marks. Write your working where needed. Suggested time: 45 minutes. Answers and marking guidance follow the questions.'),para('Original Evater practice material aligned with these school topics. Not an official NCERT paper or marking scheme. No automatic grading is implied.','Small2'),Spacer(1,10),para('Section A · Objective questions (10 marks)','Head2')]
 for i,q in enumerate(p['objective'],1):
  block=[para(f'{i}. {q[0]}')]+[para(f'{chr(65+j)}. {o}','Small2') for j,o in enumerate(q[1])]+[Spacer(1,7)]
  story.append(KeepTogether(block))
 story += [PageBreak(),para('Section B · Written responses (12 marks)','Title2'),para('Attempt all five. Use a separate sheet if you need more space. The marks for each question are shown.')]
 for i,q in enumerate(p['subjective'],11):story+=[KeepTogether([para(f'{i}. {q[0]} [{q[1]} marks]'),Spacer(1,44)])]
 story += [PageBreak(),para('Answers & marking guidance','Title2'),para('Use these after attempting the paper. Accept equivalent correct reasoning and sensible units. These are Evater model answers, not an official marking scheme.','Small2'),para('Section A · 1 mark each','Head2')]
 for i,q in enumerate(p['objective'],1):story.append(para(f'{i}. {q[2]} — {q[3]}'))
 story += [Spacer(1,12),para('Section B · Model answers','Head2')]
 for i,q in enumerate(p['subjective'],11):story.append(KeepTogether([para(f'{i}. {q[2]}'),para('Marking: '+q[3],'Small2'),Spacer(1,9)]))
 story += [Spacer(1,10),KeepTogether([para('Topic references','Head2')]+[para(x,'Small2') for x in p['references']]+[para('Questions, contexts, values, options and model answers in this edition were written for Evater. Source references support the concepts; textbook exercises, illustrations and third-party question-bank wording are not reproduced. Edition: 16 November 2026.','Small2')])]
 doc.build(story,onFirstPage=page,onLaterPages=page)
 return {'filename':p['filename'],'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'bytes':out.stat().st_size}
# These already-published editions are immutable. Building the expanded collection
# must not silently rewrite files that learners may already have downloaded.
PRESERVED = {
 '1-class-8-light-sample-paper.pdf': 'b7e10fa7fef5ba7c008f336ce10a3a905a5fac870f04d862901e6f70c1ca963e',
 '2-class-8-electricity-sample-paper.pdf': '26b420269895a8e68b6016ec6007c61bf056890322795a1550ac22cf89e5a332',
 '3-class-8-pythagoras-sample-paper.pdf': '5d8af32824c74bd2208ed55ec481eec6c0b378d33d11786e5d2d91b7cd8ab56c',
}
def build_collection():
 papers=json.loads((ROOT/'content/sample-papers/papers.json').read_text())
 result=[]
 for p in papers:
  filename=p['filename']
  if filename in PRESERVED:
   data=(ROOT/'frontend/public'/filename).read_bytes()
   digest=hashlib.sha256(data).hexdigest()
   if digest != PRESERVED[filename]:
    raise ValueError('Published PDF changed: '+filename)
   result.append({'filename':filename,'sha256':digest,'bytes':len(data)})
  else:
   result.append(build(p))
 (ROOT/'content/sample-papers/pdf-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
 return result
if __name__=='__main__':
 print(json.dumps(build_collection()))
