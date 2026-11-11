"""Offline reproduction/regression only. Does not connect to any database."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('current_main_question_bank', HERE.parents[2] / 'backend/app/question_bank.py')
qb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qb)

def row(name):
    return json.loads((HERE / name).read_text())

def score(q, answer):
    return qb.score_block([q], {q['id']: answer})[0]

def canonical_hash(q):
    payload = {k: q[k] for k in ['chapter_id', 'concept_id', 'question_text', 'correct_option_id']}
    payload['options'] = q['options_json']
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def test_exact_current_defect_reproduces_wrong_grading():
    q = row('2-published-row.json')
    assert not score(q, 'B')['is_correct']
    assert score(q, 'B')['marks_awarded'] == 0
    assert score(q, 'C')['is_correct']
    assert score(q, 'C')['marks_awarded'] == 1

@pytest.mark.parametrize('answer,correct,marks', [('B', True, 1), ('C', False, 0), ('A', False, 0), ('D', False, 0), ('', False, 0)])
def test_proposed_data_correction_uses_existing_scorer(answer, correct, marks):
    q = row('3-corrected-row-proposal.json')
    result = score(q, answer)
    assert result['is_correct'] == correct
    assert result['marks_awarded'] == marks
    assert result['correct_option_id'] == 'B'

def test_only_key_source_answer_and_hash_change():
    old = row('2-published-row.json')
    new = row('3-corrected-row-proposal.json')
    changed = {k for k in old if old[k] != new[k]}
    assert changed == {'correct_option_id', 'answer_spec', 'content_hash'}
    oldspec, newspec = old['answer_spec'], new['answer_spec']
    assert {k for k in oldspec if oldspec[k] != newspec[k]} == {'source_answer'}
    assert old['status'] == new['status'] == 'published'
    assert old['reviewed_at'] is None and new['reviewed_at'] is None
    assert old['generation_spec'] == new['generation_spec']

def test_hash_matches_existing_etl_convention_before_and_after():
    for filename in ['2-published-row.json', '3-corrected-row-proposal.json']:
        q = row(filename)
        assert canonical_hash(q) == q['content_hash']

def test_draft_sql_is_narrow_and_rollback_by_default():
    sql = (HERE / '1-correct-one-light-answer.DRAFT.sql').read_text()
    assert "WHERE id = '6aca65ae-e742-49a6-b542-ce8c11c02ee4'::uuid" in sql
    assert "AND content_hash = 'b28b613657717d3ac6ab2d045744ba212f5dce504b6ccbd03708e111fb58000e'" in sql
    assert "AND NOT (answer_spec ? 'correct_option_id')" in sql
    assert "IF changed <> 1 THEN RAISE EXCEPTION" in sql
    assert sql.strip().endswith('ROLLBACK;')
    assert 'UPDATE public.question_attempts' not in sql
