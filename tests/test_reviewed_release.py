import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from etl.reviewed_release import publication_rows,build_transaction,DIGEST
from etl.reviewed_batch import ReviewedBatchError
DIRECTORY=Path(__file__).resolve().parents[1]/'content/reviewed/class8-2026-11-14'

def test_exact_rows_have_deterministic_identity_and_honest_provenance():
    rows=publication_rows(DIRECTORY)
    assert rows==publication_rows(DIRECTORY)
    assert len(rows)==len({r['id']for r in rows})==80
    assert all(r['status']=='published' and r['quality_score']is None and r['reviewed_by']is None for r in rows)
    assert all(r['source_tier']=='manual' and r['generation_spec']['reviewed_payload_sha256']==DIGEST for r in rows)

def test_default_transaction_rolls_back_and_never_updates_existing():
    sql=build_transaction(DIRECTORY)
    assert sql.endswith('ROLLBACK;\n')
    assert 'UPDATE public.' not in sql and 'DELETE FROM' not in sql
    assert 'ON CONFLICT' not in sql
    assert 'Partial release or conflicting existing record' in sql
    assert 'Live curriculum mapping/status mismatch' in sql
    assert 'Question bank column schema changed' in sql
    assert 'Question bank constraints changed' in sql
    assert 'LOCK TABLE public.question_bank' in sql

def test_explicit_commit_changes_only_transaction_end():
    assert build_transaction(DIRECTORY,commit=True)==build_transaction(DIRECTORY).removesuffix('ROLLBACK;\n')+'COMMIT;\n'

@pytest.mark.parametrize('change',['answer','manifest_and_answer','missing','reserve'])
def test_manifest_cannot_authorize_another_payload(tmp_path,change):
    import shutil,hashlib
    for p in DIRECTORY.iterdir():shutil.copy(p,tmp_path/p.name)
    p=tmp_path/'questions.json';payload=json.loads(p.read_text())
    if change=='missing':payload['questions'].pop()
    elif change=='reserve':payload['questions'][0]['draft_id']='reserve-question'
    else:payload['questions'][0]['correct_option_id']='D'if payload['questions'][0]['correct_option_id']!='D'else'A'
    p.write_text(json.dumps(payload))
    if change=='manifest_and_answer':
        m=tmp_path/'manifest.json';v=json.loads(m.read_text());v['questions_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();m.write_text(json.dumps(v))
    with pytest.raises(ReviewedBatchError):publication_rows(tmp_path)


def test_retirement_is_exact_and_nondeleting():
    from etl.reviewed_release import build_retirement_transaction
    value=build_retirement_transaction(DIRECTORY)
    assert value.endswith('ROLLBACK;\n')
    assert "SET status='retired'" in value
    assert 'DELETE' in value.split('BEGIN;',1)[0] and 'DELETE' not in value.split('BEGIN;',1)[1]
    assert 'Release rows changed: retirement blocked' in value
