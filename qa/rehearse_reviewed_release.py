"""Local socket-only test against captured production column/constraint definitions.

This program cannot accept a production DSN or connect to a remote database.
"""
from pathlib import Path
import sys,json,uuid,argparse
import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from etl.reviewed_release import publication_rows,build_transaction,build_retirement_transaction
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evidence',type=Path,required=True);args=parser.parse_args()
EVIDENCE=args.evidence
DBNAME='evater_release80_'+uuid.uuid4().hex[:12]
DIRECTORY=ROOT/'content/reviewed/class8-2026-11-14'
ROWS=publication_rows(DIRECTORY)
admin=psycopg2.connect(host='/tmp',port=55441,dbname='postgres',user='sandbox');admin.autocommit=True
with admin.cursor()as c:
 c.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(DBNAME)))
admin.close()
conn=psycopg2.connect(host='/tmp',port=55441,dbname=DBNAME,user='sandbox');conn.autocommit=True
checks=[]
with conn.cursor()as c:
 c.execute('SELECT version()');version=c.fetchone()[0]
 c.execute('CREATE SCHEMA auth; CREATE TABLE auth.users(id uuid primary key); CREATE TABLE curriculum_versions(id uuid primary key,status text); CREATE TABLE chapters(id uuid primary key,curriculum_version_id uuid,status text); CREATE TABLE concepts(id uuid primary key,chapter_id uuid,status text); CREATE TABLE ingestion_jobs(id uuid primary key); CREATE TABLE content_sources(id uuid primary key)')
 cols=json.load(open(EVIDENCE/'columns.json'));defs=json.load(open(EVIDENCE/'definitions.json'))
 fields=[]
 for col in cols:
  typ=col['data_type']
  if typ=='ARRAY':typ='text[]'
  if typ=='numeric':typ=f"numeric({col['numeric_precision']},{col['numeric_scale']})"
  fields.append('"'+col['column_name']+'" '+typ+(' NOT NULL'if col['is_nullable']=='NO'else'')+(' DEFAULT '+col['column_default']if col['column_default']else''))
 c.execute('CREATE TABLE question_bank('+','.join(fields+defs)+')')
 c.execute('ALTER TABLE question_bank ENABLE ROW LEVEL SECURITY')
 c.execute("CREATE POLICY \"Published questions are readable\" ON question_bank FOR SELECT TO public USING(status='published')")
 for v in json.load(open(EVIDENCE/'versions.json')):c.execute('INSERT INTO curriculum_versions VALUES(%s,%s)',(v['id'],v['status']))
 for h in json.load(open(EVIDENCE/'chapters.json')):c.execute('INSERT INTO chapters VALUES(%s,%s,%s)',(h['id'],h['curriculum_version_id'],h['status']))
 for p in json.load(open(EVIDENCE/'concepts.json')):c.execute('INSERT INTO concepts VALUES(%s,%s,%s)',(p['id'],p['chapter_id'],p['status']))

def snapshot():
 with conn.cursor()as c:c.execute('SELECT to_jsonb(q) FROM question_bank q ORDER BY id');return c.fetchall()
def run(commit=True):
 with conn.cursor()as c:c.execute(build_transaction(DIRECTORY,commit=commit))
def count():return len(snapshot())
def fail(label):
 before=snapshot()
 try:run()
 except psycopg2.Error as e:
  execute('ROLLBACK');assert before==snapshot(),label;checks.append(label+': failed closed, full rows unchanged ('+e.diag.message_primary+')')
 else:raise AssertionError('unexpected success '+label)
def execute(q,args=None):
 with conn.cursor()as c:c.execute(q,args)
def reset():execute('TRUNCATE question_bank')
run(False);assert count()==0;checks.append('Default ROLLBACK:80valid inserts,0persisted')
run();assert count()==80;before=snapshot();run();assert snapshot()==before;checks.append('COMMIT80published, replay80fullrowsincludingtimestampsunchanged')
execute("UPDATE question_bank SET explanation='Changed explanation for negative fixture' WHERE id=%s",(ROWS[0]['id'],));fail('Altered explanation collision')
reset();run();execute('DELETE FROM question_bank WHERE id=%s',(ROWS[0]['id'],));fail('Partial79release')
reset();execute("UPDATE concepts SET status='retired' WHERE id=%s",(ROWS[0]['concept_id'],));fail('Retired concept');execute("UPDATE concepts SET status='published' WHERE id=%s",(ROWS[0]['concept_id'],))
execute('UPDATE concepts SET chapter_id=%s WHERE id=%s',(ROWS[-1]['chapter_id'],ROWS[0]['concept_id']));fail('Concept chapter mismatch');execute('UPDATE concepts SET chapter_id=%s WHERE id=%s',(ROWS[0]['chapter_id'],ROWS[0]['concept_id']))
execute('ALTER TABLE question_bank ADD COLUMN unexpected text');fail('Schema extra column');execute('ALTER TABLE question_bank DROP COLUMN unexpected')
execute('ALTER TABLE question_bank ADD CHECK(marks<100)');fail('Schema extra constraint');execute('ALTER TABLE question_bank DROP CONSTRAINT question_bank_marks_check1')
execute("CREATE FUNCTION qa_noop_trigger() RETURNS trigger LANGUAGE plpgsql AS $$BEGIN RETURN NEW; END$$")
execute('CREATE TRIGGER qa_unexpected BEFORE INSERT ON question_bank FOR EACH ROW EXECUTE FUNCTION qa_noop_trigger()');fail('Unexpected trigger');execute('DROP TRIGGER qa_unexpected ON question_bank')
# Different ID but same chapter/hash must not be silently ignored or replaced.
r=dict(ROWS[0]);r['id']=str(uuid.uuid4());r['published_at']='2026-11-14T00:00:00Z';cols=list(r);vals=[Json(v)if isinstance(v,(dict,list))and k not in ['skill_tags','misconception_tags']else v for k,v in r.items()]
execute(sql.SQL('INSERT INTO question_bank({})VALUES({})').format(sql.SQL(',').join(map(sql.Identifier,cols)),sql.SQL(',').join(sql.Placeholder()for _ in cols)),vals);fail('Same hash different existing ID');reset()
run();assert count()==80;checks.append('Final cleanCOMMIT80published afternegativefixtures')
# Unrelated row must remain byte-for-byte unchanged across release replay and retirement.
r=dict(ROWS[0]);r['id']=str(uuid.uuid4());r['content_hash']='unrelated-existing-hash';r['source_question_id']='unrelated-existing-question';r['status']='retired';cols=list(r);vals=[Json(v)if isinstance(v,(dict,list))and k not in ['skill_tags','misconception_tags']else v for k,v in r.items()]
execute(sql.SQL('INSERT INTO question_bank({})VALUES({})').format(sql.SQL(',').join(map(sql.Identifier,cols)),sql.SQL(',').join(sql.Placeholder()for _ in cols)),vals)
with conn.cursor()as c:c.execute('SELECT to_jsonb(q) FROM question_bank q WHERE id=%s',(r['id'],));unrelated_before=c.fetchone()
run();assert count()==81
with conn.cursor()as c:c.execute(build_retirement_transaction(DIRECTORY))
assert sum(v[0]['status']=='published'for v in snapshot())==80
checks.append('DefaultretirementROLLBACKleaves80published')
with conn.cursor()as c:c.execute(build_retirement_transaction(DIRECTORY,commit=True))
assert sum(v[0]['status']=='published'for v in snapshot())==0
before=snapshot()
execute("UPDATE question_bank SET hint='Changed hint for retirement fixture' WHERE id=%s",(ROWS[0]['id'],))
changed=snapshot()
try:execute(build_retirement_transaction(DIRECTORY,commit=True))
except psycopg2.Error:execute('ROLLBACK');assert snapshot()==changed
else:raise AssertionError('changed row retired')
execute('UPDATE question_bank SET hint=%s WHERE id=%s',(ROWS[0]['hint'],ROWS[0]['id']))
assert snapshot()==before;checks.append('Retirement changed-row fixture fails closed')
with conn.cursor()as c:c.execute(build_retirement_transaction(DIRECTORY,commit=True))
assert snapshot()==before
with conn.cursor()as c:c.execute('SELECT to_jsonb(q) FROM question_bank q WHERE id=%s',(r['id'],));assert unrelated_before==c.fetchone()
checks.append('RetirementCOMMIT80andstableidempotentreplay;unrelatedrowunchanged;nodeletions')
fail('Retired release cannot be silently republished')
reset()
# Simulated post-insert failure must not leave even one release row.
try:execute(build_transaction(DIRECTORY,commit=True).replace('SELECT e.concept_id,count(*) AS exact_release_published',"DO $$BEGIN RAISE EXCEPTION 'injected postinsert failure'; END$$; SELECT e.concept_id,count(*) AS exact_release_published"))
except psycopg2.Error:execute('ROLLBACK')
else:raise AssertionError('injected failure did not fire')
assert count()==0;checks.append('PostinsertfailureROLLBACKrestores0')
report={'passed' :True,'engine':version,'schema':'Captured production38columns/25constraintdefinitions, minimal FK parent tables with real IDs/status/mappings; RLS enabled, localpublic SELECT policy, superuser execution','checks':checks,'not_claimed':['Productiondata mutation','Productionprivilegedroleauthenticationparity','ActualproductionUIavailability','Independentteacherrevieworlearnercalibration']}
(ROOT/'qa/reviewed-release-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));conn.close()
admin=psycopg2.connect(host='/tmp',port=55441,dbname='postgres',user='sandbox');admin.autocommit=True
with admin.cursor()as c:c.execute(sql.SQL('DROP DATABASE {}').format(sql.Identifier(DBNAME)))
admin.close()
