"""Exact concept practice, with no production connection or data mutations."""
import copy
from uuid import UUID
from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest
from fastapi import FastAPI
from app.auth import require_user
from app.models import ConceptTestRequest, QuestionBankBlockSubmission, QuestionBankAnswer
from app.routers import question_bank_router as qb
app = FastAPI()
app.include_router(qb.router, prefix="/api/v1")
app.include_router(qb.capabilities_router, prefix="/api/v1")
from test_review_tests import FakeClient, FakeTable, auth, make_q, USER_ID, CHAPTER_A, CHAPTER_B, CONCEPT_A, CONCEPT_B


def client(monkeypatch, count=7):
    rows = [make_q(i, CONCEPT_A, CHAPTER_A) for i in range(1, count+1)]
    # Same chapter, other concept; same concept, other chapter; retired question.
    rows += [make_q(100, CONCEPT_B, CHAPTER_A), make_q(101, CONCEPT_A, CHAPTER_B)]
    retired = make_q(102, CONCEPT_A, CHAPTER_A); retired['status'] = 'retired'; rows.append(retired)
    tables = {name: FakeTable() for name in ['question_attempts','concept_mastery','test_attempts','test_questions','student_question_performance','student_enrollments']}
    tables.update(chapters=FakeTable([{'id': CHAPTER_A, 'title':'Chapter A','status':'published'}]),
                  concepts=FakeTable([{'id': CONCEPT_A,'chapter_id':CHAPTER_A,'title':'Exact concept','status':'published'}]), question_bank=FakeTable(rows))
    fake = FakeClient(tables)
    monkeypatch.setattr(qb, 'create_supabase_service_client', lambda: fake)
    return fake


def request(**overrides):
    return ConceptTestRequest(**dict({'chapter_id':CHAPTER_A,'concept_id':CONCEPT_A,'question_count':7,'block_size':7}, **overrides))


def no_writes(fake):
    return all(t.insert_calls == t.update_calls == t.upsert_calls == 0 for t in fake.tables.values())


def test_target_count_uses_both_ids_and_published_status_and_is_read_only(monkeypatch):
    fake = client(monkeypatch)
    result = qb.practice_target(UUID(CHAPTER_A), UUID(CONCEPT_A), auth())
    assert result.published_question_count == 7
    assert result.title == 'Exact concept'
    assert no_writes(fake)


@pytest.mark.parametrize('change', ['missing','wrong-chapter','retired-concept','retired-chapter'])
def test_unavailable_target_cannot_create_test(monkeypatch, change):
    fake = client(monkeypatch)
    if change == 'missing': fake.tables['concepts'].rows.clear()
    if change == 'wrong-chapter': fake.tables['concepts'].rows[0]['chapter_id'] = CHAPTER_B
    if change == 'retired-concept': fake.tables['concepts'].rows[0]['status'] = 'retired'
    if change == 'retired-chapter': fake.tables['chapters'].rows[0]['status'] = 'retired'
    with pytest.raises(HTTPException) as e: qb.create_concept_test(request(), auth())
    assert e.value.status_code == 404
    assert no_writes(fake)


@pytest.mark.parametrize('available', [0, 6])
def test_insufficient_inventory_never_substitutes_another_concept(monkeypatch, available):
    fake = client(monkeypatch, available)
    with pytest.raises(HTTPException) as e: qb.create_concept_test(request(), auth())
    assert e.value.status_code == 409
    assert no_writes(fake)


def test_complete_concept_set_and_submit_preserve_scope_and_ownership(monkeypatch):
    fake = client(monkeypatch)
    created = qb.create_concept_test(request(), auth())
    assert len(created.questions) == 7
    assert {str(q.concept_id) for q in created.questions} == {CONCEPT_A}
    row = fake.tables['test_attempts'].rows[0]
    assert row['user_id'] == USER_ID
    assert row['routing_profile']['selected_concept_id'] == CONCEPT_A
    row['status'] = 'in_progress'
    submitted = QuestionBankBlockSubmission(answers=[QuestionBankAnswer(question_id=q.id, answer='B') for q in created.questions])
    result = qb.submit_question_bank_block(row['id'], 1, submitted, auth())
    assert result.completed and result.next_block is None
    assert len(fake.tables['test_questions'].rows) == 7
    assert row['routing_profile']['selected_concept_id'] == CONCEPT_A
    question_concepts = {q['id']:q['concept_id'] for q in fake.tables['question_bank'].rows}
    assert {question_concepts[q['question_id']] for q in fake.tables['question_attempts'].rows} == {CONCEPT_A}


def test_multiblock_requests_rejected_before_any_write_for_mixed_version_safety(monkeypatch):
    fake = client(monkeypatch)
    with pytest.raises(HTTPException) as e: qb.create_concept_test(request(block_size=3), auth())
    assert e.value.status_code == 422
    assert no_writes(fake)


def test_http_contracts_require_auth_and_concept_and_do_not_leak_keys(monkeypatch):
    fake = client(monkeypatch)
    c = TestClient(app)
    assert c.post('/api/v1/concept-tests', json=request().model_dump(mode='json')).status_code == 401
    assert c.get(f'/api/v1/practice-target?chapter_id={CHAPTER_A}&concept_id={CONCEPT_A}').status_code == 401
    app.dependency_overrides[require_user] = auth
    try:
        payload = request().model_dump(mode='json'); payload.pop('concept_id')
        assert c.post('/api/v1/concept-tests', json=payload).status_code == 422
        result = c.post('/api/v1/concept-tests', json=request().model_dump(mode='json'))
        assert result.status_code == 200, result.text
        assert all('correct_option_id' not in q and 'answer_spec' not in q for q in result.json()['questions'])
    finally: app.dependency_overrides.clear()


def test_other_user_cannot_submit_a_concept_test(monkeypatch):
    from types import SimpleNamespace
    fake = client(monkeypatch)
    created = qb.create_concept_test(request(), auth())
    row = fake.tables['test_attempts'].rows[0]; row['status'] = 'in_progress'
    initial = copy.deepcopy(fake.tables['question_attempts'].rows)
    with pytest.raises(HTTPException) as e:
        qb.submit_question_bank_block(str(created.test_id), 1, QuestionBankBlockSubmission(answers=[]), SimpleNamespace(user=SimpleNamespace(id=CHAPTER_B)))
    assert e.value.status_code == 404
    assert fake.tables['question_attempts'].rows == initial


def test_public_capability_route_has_no_account_or_bank_data():
    c = TestClient(app)
    result = c.get('/api/v1/practice-capabilities')
    assert result.status_code == 200
    assert result.json() == {"concept_test_contract": 1}
    # Assert this public router is actually mounted by the real app factory.
    import ast
    from pathlib import Path
    tree = ast.parse((Path(__file__).parents[1] / 'backend/app/main.py').read_text())
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'include_router'
               and n.args and ast.unparse(n.args[0]) == 'question_bank_router.capabilities_router' for n in ast.walk(tree))
