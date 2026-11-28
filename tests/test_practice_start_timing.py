"""Measure start phases without changing selection, writes or public contracts."""
import copy
import logging
from types import SimpleNamespace
import pytest
from fastapi import HTTPException
from app.models import QuestionBankTestRequest
from app.routers import question_bank_router as qb
from test_concept_tests import client, request, no_writes
from test_review_tests import auth, CHAPTER_A


@pytest.mark.parametrize('concept', [False, True])
@pytest.mark.parametrize('auth_ms,expected', [(125.0, '125.0'), (None, 'unknown')])
def test_success_logs_only_phase_times_and_fixed_scope(monkeypatch, caplog, concept, auth_ms, expected):
    fake = client(monkeypatch)
    ctx = auth(); ctx.validation_ms = auth_ms
    ctx.jwt = 'private-token'; ctx.tenant_id = 'private-tenant'
    clock = iter([i / 100 for i in range(10)])
    monkeypatch.setattr(qb, 'perf_counter', lambda: next(clock))
    monkeypatch.setattr(qb.secrets, 'randbelow', lambda _: 12)
    req = request() if concept else QuestionBankTestRequest(chapter_id=CHAPTER_A, question_count=7, block_size=7)
    original_rows = copy.deepcopy(fake.tables['question_bank'].rows)
    with caplog.at_level(logging.INFO, logger=qb.__name__):
        response = qb.create_concept_test(req, ctx) if concept else qb.create_question_bank_test(req, ctx)
    messages = [r.getMessage() for r in caplog.records if r.name == qb.__name__ and r.getMessage().startswith('practice_start_timing')]
    scope = 'concept' if concept else 'chapter'
    assert messages == [f'practice_start_timing auth_ms={expected} client_ms=10.0 chapter_ms=10.0 questions_ms=10.0 mastery_ms=10.0 history_ms=10.0 selection_ms=10.0 attempt_insert_ms=10.0 questions_insert_ms=10.0 response_ms=10.0 total_ms=90.0 scope={scope}']
    assert all(value not in messages[0] for value in ['private-token', 'private-tenant', str(ctx.user.id), CHAPTER_A])
    assert fake.tables['question_bank'].rows == original_rows
    assert fake.tables['test_attempts'].insert_calls == 1
    assert fake.tables['test_questions'].insert_calls == 1
    assert fake.tables['test_attempts'].rows[0]['seed'] == 13
    payload = response.model_dump(mode='json')
    assert set(payload) == {'test_id','block_number','block_size','question_count','questions','routing'}
    assert all('correct_option_id' not in q and 'answer_spec' not in q for q in payload['questions'])


def test_missing_auth_timing_is_supported_without_changing_older_callers(monkeypatch, caplog):
    client(monkeypatch)
    with caplog.at_level(logging.INFO, logger=qb.__name__): qb.create_concept_test(request(), auth())
    assert any('practice_start_timing auth_ms=unknown' in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize('failure', ['unpublished', 'insufficient'])
def test_rejected_starts_do_not_emit_success_timing_or_write(monkeypatch, caplog, failure):
    fake = client(monkeypatch, count=2 if failure == 'insufficient' else 7)
    if failure == 'unpublished': fake.tables['chapters'].rows[0]['status'] = 'retired'
    with caplog.at_level(logging.INFO, logger=qb.__name__), pytest.raises(HTTPException):
        qb.create_concept_test(request(), auth())
    assert no_writes(fake)
    assert not any('practice_start_timing' in r.getMessage() for r in caplog.records)
