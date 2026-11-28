import json,ast,math,re
from pathlib import Path
R=Path(__file__).parent

def calc(n):
 if isinstance(n,ast.Expression):return calc(n.body)
 if isinstance(n,ast.Constant)and isinstance(n.value,(int,float)):return n.value
 if isinstance(n,ast.UnaryOp)and isinstance(n.op,ast.USub):return -calc(n.operand)
 if isinstance(n,ast.BinOp):
  a,b=calc(n.left),calc(n.right)
  if isinstance(n.op,ast.Add):return a+b
  if isinstance(n.op,ast.Sub):return a-b
  if isinstance(n.op,ast.Mult):return a*b
  if isinstance(n.op,ast.Div):return a/b
  if isinstance(n.op,ast.Pow):return a**b
  if isinstance(n.op,ast.Mod):return a%b
 raise ValueError('unsupported arithmetic')
allstems=[]
for name,nchap in [('mathematics',14),('science',13)]:
 p=json.load(open(R/(name+'.json')));qs=p['questions'];assert p['status']=='original_draft_not_published';assert p['official_pattern']is False
 assert len(qs)==24;assert sum(q['marks']for q in qs)==60;assert[q['number']for q in qs]==list(range(1,25))
 assert set(q['chapter']for q in qs)==set(range(1,nchap+1))
 assert [sum(q['marks']for q in qs[a:b])for a,b in [(0,12),(12,18),(18,22),(22,24)]]==[12,18,20,10]
 for q in qs:
  assert len(q['stem'])>=20;assert q['answer'];allstems.append(q['stem'])
  if q['type']=='mcq':
   assert[x['id']for x in q['options']]==list('ABCD');assert q['answer']in'ABCD';assert len({x['text'].casefold()for x in q['options']})==4
  else:assert sum(x['marks']for x in q['marking'])==q['marks']
  if q.get('arithmetic_check'):
   expr,expect=q['arithmetic_check'];assert math.isclose(calc(ast.parse(expr,mode='eval')),expect,rel_tol=1e-10),(q['number'],expr)
 print(name,'24questions/60marks/sectiontotals/catalogchaptercoverage/options/marking/arithmeticPASS')
assert len(set(allstems))==48
assert 18*75+12*50==1950;assert 1950*.9==1755
assert 4**2+3**2==5**2;assert 8**2+6**2==10**2;assert 8*6/2==24
assert math.isclose((7.2-6.4)*250/1000,.2)
print('48distinctstems/casearithmeticPASS')
