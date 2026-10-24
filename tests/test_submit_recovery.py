"""Targeted tests for the idempotent block-submit recovery path.

A real prod incident (Oct 24): a final-block submit recorded all attempts,
then died mid-request, stranding the test - every retry 409ed. These tests
pin the recovery behaviour with a fake Supabase client.
"""
import copy
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.models import QuestionBankAnswer, QuestionBankBlockSubmission
from app.routers import question_bank_router as qbr
from app.routers import leaderboard_router as lbr


USER_ID = "11111111-1111-4111-8111-111111111111"
CHAPTER_ID = "22222222-2222-4222-8222-222222222222"
TEST_ID = "33333333-3333-4333-8333-333333333333"


def make_q(index, concept="concept-a", correct="B"):
    return {
        "id": str(uuid.UUID(int=index)),
        "chapter_id": CHAPTER_ID,
        "concept_id": concept,
        "question_type": "mcq_single",
        "question_text": f"Q{index}?",
        "options_json": [
            {"id": "A", "text": "a"}, {"id": "B", "text": "b"},
            {"id": "C", "text": "c"}, {"id": "D", "text": "d"},
        ],
        "correct_option_id": correct,
        "explanation": "why",
        "difficulty": "medium",
        "cognitive_level": "understand",
        "skill_tags": [],
        "misconception_tags": [],
        "question_style": "direct",
        "estimated_time_seconds": 60,
        "marks": 1,
        "media_json": [],
        "option_media_json": {},
        "render_config": {},
        "answer_spec": None,
    }


QUESTIONS = [make_q(1), make_q(2)]
TQ_ROWS = [
    {"id": str(uuid.UUID(int=101)), "test_attempt_id": TEST_ID, "question_id": QUESTIONS[0]["id"], "block_number": 1, "display_order": 1},
    {"id": str(uuid.UUID(int=102)), "test_attempt_id": TEST_ID, "question_id": QUESTIONS[1]["id"], "block_number": 1, "display_order": 2},
]
TEST_ROW = {
    "id": TEST_ID, "user_id": USER_ID, "chapter_id": CHAPTER_ID,
    "mode": "practice", "question_count": 2, "block_size": 2,
    "current_block": 1, "seed": 7, "routing_profile": {"mode": "practice", "difficulty": "medium"},
    "status": "in_progress", "score": None, "maximum_marks": None,
}


class FakeQuery:
    def __init__(self, table):
        self.table = table
        self.filters = []
        self.in_filters = []
        self._order = None
        self._limit = None
        self.mutation = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, col, val):
        self.filters.append((col, val))
        return self

    def in_(self, col, vals):
        self.in_filters.append((col, set(str(v) for v in vals)))
        return self

    def order(self, col, desc=False):
        self._order = (col, desc)
        return self

    def limit(self, n):
        self._limit = n
        return self

    def insert(self, rows):
        self.mutation = ("insert", rows)
        return self

    def update(self, values):
        self.mutation = ("update", values)
        return self

    def upsert(self, rows, on_conflict=None):
        self.mutation = ("upsert", rows, on_conflict)
        return self

    def _matched(self):
        rows = [r for r in self.table.rows if all(str(r.get(c)) == str(v) for c, v in self.filters)]
        for col, vals in self.in_filters:
            rows = [r for r in rows if str(r.get(col)) in vals]
        return rows

    def execute(self):
        if self.table.fail_next_execute:
            self.table.fail_next_execute = False
            raise RuntimeError("simulated mid-request failure")
        if self.mutation:
            kind = self.mutation[0]
            if kind == "insert":
                self.table.rows.extend(copy.deepcopy(self.mutation[1]))
                self.table.insert_calls += 1
            elif kind == "update":
                values = self.mutation[1]
                for r in self._matched():
                    r.update(copy.deepcopy(values))
                self.table.update_calls += 1
            elif kind == "upsert":
                rows, conflict = self.mutation[1], (self.mutation[2] or "").split(",")
                for new_row in rows:
                    for r in self.table.rows:
                        if all(str(r.get(c)) == str(new_row.get(c)) for c in conflict):
                            r.update(copy.deepcopy(new_row))
                            break
                    else:
                        self.table.rows.append(copy.deepcopy(new_row))
                self.table.upsert_calls += 1
            return SimpleNamespace(data=[])
        rows = self._matched()
        if self._order:
            col, desc = self._order
            rows = sorted(rows, key=lambda r: r.get(col), reverse=desc)
        if self._limit:
            rows = rows[: self._limit]
        return SimpleNamespace(data=copy.deepcopy(rows))


class FakeTable:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.fail_next_execute = False
        self.insert_calls = 0
        self.update_calls = 0
        self.upsert_calls = 0


class FakeClient:
    def __init__(self, tables):
        self.tables = tables

    def table(self, name):
        return FakeQuery(self.tables.setdefault(name, FakeTable()))


@pytest.fixture
def client(monkeypatch):
    tables = {
        "test_attempts": FakeTable([copy.deepcopy(TEST_ROW)]),
        "test_questions": FakeTable(copy.deepcopy(TQ_ROWS)),
        "question_bank": FakeTable(copy.deepcopy(QUESTIONS)),
        "question_attempts": FakeTable([]),
        "concept_mastery": FakeTable([]),
        "student_question_performance": FakeTable([]),
        "student_enrollments": FakeTable([]),  # no enrollment: leaderboard skipped
    }
    fake = FakeClient(tables)
    monkeypatch.setattr(qbr, "create_supabase_service_client", lambda: fake)
    return fake


def auth():
    return SimpleNamespace(user=SimpleNamespace(id=USER_ID))


def submission(option):
    return QuestionBankBlockSubmission(
        answers=[QuestionBankAnswer(question_id=q["id"], answer=option) for q in QUESTIONS]
    )


def stored_attempts(option):
    return [
        {
            "test_attempt_id": TEST_ID,
            "test_question_id": tq["id"],
            "user_id": USER_ID,
            "chapter_id": CHAPTER_ID,
            "question_id": tq["question_id"],
            "selected_option_id": None,
            "answer_json": {"value": option},
            "is_correct": option == "B",
            "marks_awarded": 1 if option == "B" else 0,
        }
        for tq in TQ_ROWS
    ]


def test_normal_submit_inserts_attempts_and_completes(client):
    response = qbr.submit_question_bank_block(uuid.UUID(TEST_ID), 1, submission("B"), auth())
    assert response.completed is True
    assert response.block_score == 2
    assert len(client.tables["question_attempts"].rows) == 2
    assert client.tables["question_attempts"].insert_calls == 1
    assert client.tables["test_attempts"].rows[0]["status"] == "completed"
    assert client.tables["concept_mastery"].upsert_calls == 1


def test_submit_recovers_from_recorded_attempts_and_ignores_retry_payload(client):
    # First submission (correct answers) was recorded, then the request died.
    client.tables["question_attempts"].rows = stored_attempts("B")
    # The retry payload disagrees (all wrong) - the recorded first write must win.
    response = qbr.submit_question_bank_block(uuid.UUID(TEST_ID), 1, submission("A"), auth())
    assert response.completed is True
    assert response.block_score == 2, "recovery must score the stored answers, not the retry payload"
    assert all(r.is_correct for r in response.results)
    # No duplicate insert on recovery; follow-ups still replay.
    assert client.tables["question_attempts"].insert_calls == 0
    assert len(client.tables["question_attempts"].rows) == 2
    assert client.tables["test_attempts"].rows[0]["status"] == "completed"
    assert client.tables["test_attempts"].rows[0]["score"] == 2
    assert client.tables["concept_mastery"].upsert_calls == 1


def test_submit_partial_attempt_set_still_409s(client):
    client.tables["question_attempts"].rows = stored_attempts("B")[:1]
    with pytest.raises(HTTPException) as excinfo:
        qbr.submit_question_bank_block(uuid.UUID(TEST_ID), 1, submission("B"), auth())
    assert excinfo.value.status_code == 409
    assert client.tables["test_attempts"].rows[0]["status"] == "in_progress"


def test_failure_after_insert_then_retry_recovers_and_completes(client):
    # Die at the first follow-up write (mastery upsert), after attempts insert.
    client.tables["concept_mastery"].fail_next_execute = True
    with pytest.raises(HTTPException) as excinfo:
        qbr.submit_question_bank_block(uuid.UUID(TEST_ID), 1, submission("B"), auth())
    assert excinfo.value.status_code == 500
    assert len(client.tables["question_attempts"].rows) == 2  # attempts survived
    assert client.tables["test_attempts"].rows[0]["status"] == "in_progress"

    # Retry succeeds through the recovery path; no duplicate attempts.
    response = qbr.submit_question_bank_block(uuid.UUID(TEST_ID), 1, submission("B"), auth())
    assert response.completed is True
    assert len(client.tables["question_attempts"].rows) == 2
    assert client.tables["question_attempts"].insert_calls == 1  # only the first call inserted
    assert client.tables["test_attempts"].rows[0]["status"] == "completed"
    assert client.tables["concept_mastery"].upsert_calls == 1  # retry replayed the upsert once (the first attempt raised before landing)


def test_leaderboard_award_is_once_only():
    tables = {
        "student_enrollments": FakeTable([{"user_id": USER_ID, "role": "student"}]),
        "leaderboard_practice_scores": FakeTable([
            {"test_attempt_id": TEST_ID, "points_awarded": 5, "raw_score": 2}
        ]),
        "question_attempts": FakeTable([]),
    }
    fake = FakeClient(tables)
    award = lbr.award_completed_practice(fake, test_id=TEST_ID, user_id=USER_ID, completed_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc))
    assert award == {"points_awarded": 5, "raw_score": 2}
    # No new leaderboard writes when the award already exists.
    assert client_insert_totals(fake) == 0


def client_insert_totals(fake):
    return sum(t.insert_calls for t in fake.tables.values())
