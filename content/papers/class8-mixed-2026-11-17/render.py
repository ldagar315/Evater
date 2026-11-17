from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from xml.sax.saxutils import escape
from pathlib import Path
import json,sys
RELEASE="--release" in sys.argv
R=Path(__file__).parent
pdfmetrics.registerFont(TTFont('DV','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'));pdfmetrics.registerFont(TTFont('DVB','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
S=getSampleStyleSheet();S.add(ParagraphStyle(name='body',fontName='DV',fontSize=10,leading=15,spaceAfter=8));S.add(ParagraphStyle(name='small',fontName='DV',fontSize=8,leading=12,spaceAfter=6));S.add(ParagraphStyle(name='title2',fontName='DVB',fontSize=21,leading=27,spaceAfter=15,textColor=colors.HexColor('#143e38')));S.add(ParagraphStyle(name='h2b',fontName='DVB',fontSize=13,leading=18,spaceAfter=12,textColor=colors.HexColor('#143e38')))
def p(t,style='body'):return Paragraph(escape(t).replace('\n','<br/>'),S[style])
def page(c,d):
 c.saveState();
 if not RELEASE:
  c.setFillColor(colors.HexColor('#12665b'));c.setFillAlpha(.045);c.translate(295,420);c.rotate(38);c.setFont('DVB',47);c.drawCentredString(0,0,'EVATER DRAFT')
 c.restoreState();c.saveState();c.setFont('DV',8);c.setFillColor(colors.HexColor('#555555'));c.drawString(44,31,('Evater • Original mixed-chapter practice • evater.xyz' if RELEASE else 'Evater • Original mixed-chapter practice • Review draft, not published'));c.drawRightString(551,31,str(d.page));c.restoreState()
for name in ['mathematics','science']:
 data=json.load(open(R/(name+'.json')));story=[p('CLASS 8 / '+name.upper(),'small'),p(data['title'],'title2'),p('60 marks · 24 questions · 150 minutes suggested','h2b'),p('Original Evater practice. Not an official NCERT or CBSE paper. Covers selected topics across chapters, not every syllabus skill.' if RELEASE else 'Original Evater review draft. Not an official NCERT or CBSE paper. Samples topics from all catalog chapters, not every syllabus skill.','small')]+[p(t)for t in data['instructions']]
 for q in data['questions']:
  if q['number']in [1,13,19,23]:
   if q['number']!=1:story.append(PageBreak())
   story += [Spacer(1,12),p({1:'Section A / Objective questions (12 marks)',13:'Section B / Short answers (18 marks)',19:'Section C / Extended answers (20 marks)',23:'Section D / Case questions (10 marks)'}[q['number']],'h2b')]
  block=[p(str(q['number'])+'. '+q['stem']+' ['+str(q['marks'])+' mark'+('s'if q['marks']>1 else '')+']')]
  block +=[p(x['id']+'. '+x['text'],'small')for x in q.get('options',[])]
  block +=[Spacer(1,8 if q['type']=='mcq' else 30)]
  story.append(KeepTogether(block))
 story +=[PageBreak(),p('Model answers & marking guidance','title2'),p('Read after attempting the paper. Equivalent correct methods and wording earn credit. The criteria below are Evater guidance, not an official marking scheme.','small')]
 for q in data['questions']:
  a=q['answer']+(' / '+q['explanation']if 'explanation'in q else '')
  block=[p(str(q['number'])+'. '+a)]
  if'marking'in q:block +=[p('Marking: '+'; '.join(x['point']+' ('+str(x['marks'])+')'for x in q['marking']),'small')]
  story +=[KeepTogether(block),Spacer(1,8)]
 doc=SimpleDocTemplate(str(R/(name+('-practice.pdf' if RELEASE else '-review.pdf'))),pagesize=(595.28,841.89),rightMargin=44,leftMargin=44,topMargin=42,bottomMargin=58,title=data['title']+('' if RELEASE else ' / review draft'),author='Evater',invariant=1)
 doc.build(story,onFirstPage=page,onLaterPages=page)
 print(name,'rendered')
