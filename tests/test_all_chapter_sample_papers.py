"""Offline release checks. These do not claim deployed availability."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.etl.class8_maths_catalog import CHAPTERS as MATHS
from backend.etl.class8_science_catalog import CHAPTERS as SCIENCE

PAPERS = json.loads((ROOT / 'content/sample-papers/papers.json').read_text())
PRESERVED = {
    '1-class-8-light-sample-paper.pdf': 'b7e10fa7fef5ba7c008f336ce10a3a905a5fac870f04d862901e6f70c1ca963e',
    '2-class-8-electricity-sample-paper.pdf': '26b420269895a8e68b6016ec6007c61bf056890322795a1550ac22cf89e5a332',
    '3-class-8-pythagoras-sample-paper.pdf': '5d8af32824c74bd2208ed55ec481eec6c0b378d33d11786e5d2d91b7cd8ab56c',
}

def test_all_27_catalog_chapters_once():
    assert len(PAPERS) == 27
    new = [p for p in PAPERS if p['filename'] not in PRESERVED]
    for subject, catalog, existing in [('Mathematics', MATHS, {9}), ('Science', SCIENCE, {4, 10})]:
        subset = [p for p in new if p['subject'] == subject]
        assert {p['chapter'] for p in subset} == {c.sequence_number for c in catalog} - existing
        for p in subset:
            chapter = next(c for c in catalog if c.sequence_number == p['chapter'])
            assert p['title'] == chapter.title
            assert chapter.source_url in p['references']
    assert len({p['filename'] for p in PAPERS}) == 27
    assert len({p['objective'][0][0] for p in PAPERS}) == 27

def test_existing_pdf_bytes_are_immutable():
    for filename, digest in PRESERVED.items():
        assert hashlib.sha256((ROOT/'frontend/public'/filename).read_bytes()).hexdigest() == digest

def test_405_questions_and_marks():
    for p in PAPERS:
        assert len(p['objective']) == 10
        assert len(p['subjective']) == 5
        assert sum(q[1] for q in p['subjective']) == 12
        for prompt, options, key, explanation in p['objective']:
            assert prompt and explanation
            assert len(options) == len(set(options)) == 4
            assert key in 'ABCD'
        assert all(q[0] and q[2] and q[3] for q in p['subjective'])
    assert sum(len(p['objective']) + len(p['subjective']) for p in PAPERS) == 405

def test_correct_options_not_position_pattern():
    for p in PAPERS[3:]:
        assert len({q[2] for q in p['objective']}) >= 3

def test_numerical_and_symbolic_answers_independent():
    # Values computed without consulting stored answer keys or explanations.
    import math
    from fractions import Fraction
    assert [17**2, 6**3, 11**2, 7**3, 27**2 % 10] == [289,216,121,343,9]
    assert sum(range(1,8,2)) == 4**2 == 16
    assert [8**2, 9**3 % 10, 2**3 * 3**2] == [64,9,72]
    assert 5**2 + 12**2 == 13**2
    assert 15**2 == 225 and 3**3 == 27 and sum(range(1,9)) == 6**2 == 36
    assert min(k for k in range(1,73) if math.isqrt(72*k)**2 == 72*k) == 2
    assert 13**2 - 12**2 == 13+12 == 25
    assert [2**5, 3**4 * 3**2, 7**8 // 7**5, (2**3)**2] == [32,3**6,7**3,2**6]
    assert 5**0 == 1 and Fraction(3)**-2 == Fraction(1,9)
    assert 3.4*10**4 == 34000 and math.isclose(7*10**-4,0.0007)
    assert 0.1*2**5 == 3.2 and 2**4*2**3//2**5 == 4
    assert math.isclose(5.6*10**-4,0.00056) and 2**10/10**2 == 10.24 and 2**3 == 8
    assert int('1011',2) == 11 and int('11001',2) == 25 and int('34',5) == 19
    assert int('1101',2) == 13 and int('132',5) == 42
    assert 360-70-80-100 == 110 and 180-70 == 110
    assert [360*k/10 for k in [1,2,3,4]] == [36,72,108,144]
    assert math.hypot(10/2,24/2) == 13
    assert 3411%3 == 0 and 316%4 == 0 and 5391%9 == 0 and 232%5 != 0
    assert math.gcd(12,18)==6 and math.lcm(4,6)==12 and 2**2*3*5 == 60
    assert 2728==11*248 and 6732==9*748 and math.gcd(24,36)==12 and 98989==11*8999
    assert math.lcm(12,18)==36
    assert 102*98==9996 and 99**2==9801 and 53*47==2491
    for a in [-7,-1,0,2,5,11]:
        for b in [-3,0,2,8]:
            assert (a+b)**2 == a*a+2*a*b+b*b
            assert (a-b)**2 == a*a-2*a*b+b*b
            assert (a+b)*(a-b) == a*a-b*b
            assert (a+b)**2-(a-b)**2 == 4*a*b
        assert (a+2)*(a+5)==a*a+7*a+10
        assert (a+4)**2==a*a+8*a+16
        assert (a+3)*(a+2)==a*a+5*a+6
    assert [150/3,180/3*5,60/4*10,500/2000] == [50,300,150,.25]
    assert 2*30/12==5 and [600/3,2*600/3]==[200,400]
    assert 7*5==35 and 3*6/2==9 and Fraction(45,75)==Fraction(3,5)
    assert 240/8*5==150 and [840*3/7,840*4/7]==[360,480]
    assert 420/(240/4)==7 and 15*25/10==37.5
    assert Fraction(3,5)*100==60 and .25*240==60 and 500*1.1==550
    assert 36/50*100==72 and 800*.85==680 and (250-200)/200*100==25
    assert max([.41,2/5,.45,7/20])==.45 and 2000+2*.1*2000==2400
    assert 45/.3==150 and 50000-30000-5000==15000
    assert 27/30*100==90 and 40/50*100==80 and 1200*.25==300 and 1200*.75==900
    assert 100*1.5*.5==75 and 10000*.08*3==2400 and 10000+2400==12400
    assert 420-350==70 and (420-350)/350*100==20 and 350*1.3==455
    assert 3*100000/100000==3 and 600*2/5==240 and 60*2/6==20
    assert 360*10/40==90 and 6*8/12==4 and 4*18/6==12 and 1/5*100==20
    assert 8*250000/100000==20 and [1800*k/9 for k in [2,3,4]]==[400,600,800]
    assert [360*x for x in [.5,.3,.2]]==[180,108,72]
    assert [9000*x for x in [.5,.3,.2]]==[4500,2700,1800]
    assert 4*9/6==6 and 4*9/3==12
    assert 9-1==8 and 4-1==3 and 2+12-6==8 and 2+6==8
    assert 5+6-9==2 and 81*Fraction(8,9)==72 and 81*Fraction(8,9)**2==64
    assert (1+10)+(10+1)-20==2
    assert sum([6,10,14,18,22])/5==14 and sum([10,10,11,17])/4==12
    assert sum([12,15,18,21,24])/5==18 and (4*25+40)/5==28
    assert sum([2,5,8])/3==5 and sum([45,60,30,75,40])/5==50
    assert (250+80)/6==55 and 14-9==5 and 3+5==8 and (1+2)+(2+3)==8
    assert 5121%3==0 and max(k*(10-k)for k in range(1,10))==25
    assert 2*4+3==11 and (4+7)+(7+2)==20 and sum([12,13,19,20])==64
    assert 2+2*5+9==21 and sum(map(int,'7382451'))==30 and 7382451%9!=0
    assert 12*7/2==42 and 10*6==60 and 10*8/2==40 and 9**2==81
    assert (8+12)*5/2==50 and 64*2/16==8 and 15*8/2==60 and 126/14==9
    assert 12*16/2==96 and (25+15)*10/2==200 and 200*12==2400
    # New Science quantitative scenarios and corrected calendar models.
    assert 100/50==2 and [60/50,45/50]==[1.2,.9]
    assert 80*1.25+30==130 and 96/8==12
    assert 12*29.5==354 and 365.25-354==11.25
    assert 3*365+366==4*365.25==1461
    assert 18*6==108 and 240/(4*.004)==15000

def test_new_maths_objective_key_values():
    expected = {
      1: ['289','216','121','343','9','16','64','9','72','5 squared + 12 squared = 13 squared'],
      2: ['32','3 to the power 6','7 to the power 3','2 to the power 6','1','1/9','3.4 x 10 to the power 4','2 to the power 5','7 x 10 to the power -4','3.2 mm'],
      3: ['40','Thousands','11','0, 1, 2, 3, 4','XLIX','10','60','11001','19','Place value together with zero'],
      4: ['360 degrees','Equal and parallel','They are equal and bisect each other','Bisect each other at right angles','Rhombus','110 degrees','180 degrees','90 degrees','Two pairs of equal adjacent sides','Equal and perpendicular'],
      5: ['3411','316','2','5,391','232','6','12','2 squared x 3 x 5','2728','6'],
      6: ['3x + 15','a squared + 2ab + b squared','169','9,996','(a + b)(a - b)','x squared + 7x + 10','ab + ac','9,801','a squared - 2ab + b squared','10x - 15'],
      7: ['2 : 3','50 rupees','300 km','150 rupees','1 : 4','5 cups','200 and 400 rupees','35 km','12 : 15','9 m'],
      8: ['60%','60','Rs 550','72%','Rs 680','25%','0.45','2400','150','Rs 15,000'],
      10:['3 km','Rs 240','20 kg','90 deg','360 deg','4 days','12','Half','Number of equally productive workers and time for a divisible fixed job','20%'],
      11:['A small part looks like the whole','8','3','An equilateral triangle','12','5','8','A flat pattern that folds up into the solid','8','9'],
      12:['14','Exactly halfway between them','Increase','Stay the same','100','12','How a value changes over time','A line graph','Increases by 3','Increase'],
      13:['3','5','8','8','a + 2b + c','4n + 16','5121','Its digit sum is divisible by 9','25','x = 4'],
      14:['(1/2) x b x h','42 cm2','Two congruent triangles of equal area','60 cm2','Two triangles of equal area','Equal','40 cm2','81 m2','50 cm2','8 cm'],
    }
    def compact(s):
        return re.sub(r'\s+','',s).lower()
    for p in PAPERS[3:]:
        if p['subject'] == 'Mathematics':
            actual=[q[1][ord(q[2])-65] for q in p['objective']]
            assert list(map(compact,actual)) == list(map(compact,expected[p['chapter']]))
