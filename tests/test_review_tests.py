"""Review tests: cross-chapter weak-spot practice.

Covers the pure weak-concept ranking and the router behaviour with a fake
Supabase client: review creation requires practice history, review rows keep
chapter_id NULL on test_attempts, and chapter-scoped writes in the submit
path resolve each question's own chapter.
"""
import copy
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.models import QuestionBankAnswer, QuestionBankBlockSubmission, ReviewTestRequest
from app.question_bank import pick_review_focus
from app.routers import question_bank_router as qbr


USER_ID = "11111111-1111-4111-8111-111111111111"
CHAPTER_A = "22222222-2222-4222-8222-222222222222"
CHAPTER_B = "44444444-4444-4444-8444-444444444444"
CONCEPT_A = "55555555-5555-4555-8555-555555555555"
CONCEPT_B = "66666666-6666-4666-8666-666666666666"


def mastery_row(concept, chapter, score, attempts, last="2026-11-01T10:00:00+00:00"):
    return {
        "user_id": USER_ID,
        "concept_id": concept,
        "chapter_id": chapter,
        "mastery_score": score,
        "attempt_count": attempts,
        "correct_count": attempts // 2,
        "last_difficulty": "medium",
        "last_attempted_at": last,
        "misconception_signals": {},
    }


def test_pick_review_focus_ranks_weakest_first_and_requires_signal():
    rows = [
        mastery_row("weak", CHAPTER_A, 0.2, 5),
        mastery_row("strong", CHAPTER_A, 0.95, 9),
        mastery_row("thin-signal", CHAPTER_A, 0.0, 2),  # below min_attempts
        mastery_row("mid", CHAPTER_B, 0.55, 4),
    ]
    focus = pick_review_focus(rows)
    assert [item["concept_id"] for item in focus] == ["weak", "mid", "strong"]
    assert focus[0]["chapter_id"] == CHAPTER_A
    assert focus[1]["chapter_id"] == CHAPTER_B


def test_pick_review_focus_rotates_stale_concepts_on_ties_and_caps():
    rows = [
        mastery_row(f"c{i}", CHAPTER_A, 0.5, 6, last=f"2026-10-{10 + i:02d}T10:00:00+00:00")
        for i in range(12)
    ]
    focus = pick_review_focus(rows)
    assert len(focus) == 8
    # Oldest untouched first among equal mastery.
    assert focus[0]["concept_id"] == "c0"
    assert focus[1]["concept_id"] == "c1"


def test_pick_review_focus_empty_history():
    assert pick_review_focus([]) == []
    assert pick_review_focus([mastery_row("new", CHAPTER_A, 0.1, 1)]) == []


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
        if self.mutation:
            kind = self.mutation[0]
            if kind == "insert":
                payload = self.mutation[1]
                rows = copy.deepcopy(payload if isinstance(payload, list) else [payload])
                for row in rows:
                    row.setdefault("id", str(uuid.uuid4()))
                self.table.rows.extend(copy.deepcopy(rows))
                self.table.insert_calls += 1
                return SimpleNamespace(data=rows)
            if kind == "update":
                values = self.mutation[1]
                for r in self._matched():
                    r.update(copy.deepcopy(values))
                self.table.update_calls += 1
                return SimpleNamespace(data=[])
            if kind == "upsert":
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
        self.insert_calls = 0
        self.update_calls = 0
        self.upsert_calls = 0


class FakeClient:
    def __init__(self, tables):
        self.tables = tables

    def table(self, name):
        return FakeQuery(self.tables.setdefault(name, FakeTable()))


def make_q(index, concept, chapter):
    return {
        "id": str(uuid.UUID(int=index)),
        "chapter_id": chapter,
        "concept_id": concept,
        "question_type": "mcq_single",
        "question_text": f"Q{index}?",
        "options_json": [
            {"id": "A", "text": "a"}, {"id": "B", "text": "b"},
            {"id": "C", "text": "c"}, {"id": "D", "text": "d"},
        ],
        "correct_option_id": "B",
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
        "status": "published",
    }


QUESTIONS = (
    [make_q(i, CONCEPT_A, CHAPTER_A) for i in range(1, 4)]
    + [make_q(i, CONCEPT_B, CHAPTER_B) for i in range(4, 7)]
)


def review_client(monkeypatch, mastery_rows):
    tables = {
        "concept_mastery": FakeTable(copy.deepcopy(mastery_rows)),
        "question_bank": FakeTable(copy.deepcopy(QUESTIONS)),
        "question_attempts": FakeTable([]),
        "test_attempts": FakeTable([]),
        "test_questions": FakeTable([]),
        "student_question_performance": FakeTable([]),
        "student_enrollments": FakeTable([]),
    }
    fake = FakeClient(tables)
    monkeypatch.setattr(qbr, "create_supabase_service_client", lambda: fake)
    return fake


def auth():
    return SimpleNamespace(user=SimpleNamespace(id=USER_ID))


HISTORY = [
    mastery_row(CONCEPT_A, CHAPTER_A, 0.3, 6),
    mastery_row(CONCEPT_B, CHAPTER_B, 0.4, 5),
]


def test_review_test_requires_practice_history(monkeypatch):
    review_client(monkeypatch, [])
    with pytest.raises(HTTPException) as excinfo:
        qbr.create_question_bank_review_test(ReviewTestRequest(), auth())
    assert excinfo.value.status_code == 409


def test_review_test_creation_spans_chapters(monkeypatch):
    fake = review_client(monkeypatch, HISTORY)
    response = qbr.create_question_bank_review_test(
        ReviewTestRequest(question_count=4, block_size=4), auth()
    )
    assert len(response.questions) == 4
    assert {str(value) for value in response.routing.focus_concept_ids} == {CONCEPT_A, CONCEPT_B}
    test_row = fake.tables["test_attempts"].rows[0]
    assert test_row["chapter_id"] is None
    assert test_row["mode"] == "review"
    assert set(test_row["routing_profile"]["review_concept_ids"]) == {CONCEPT_A, CONCEPT_B}
    assert fake.tables["test_questions"].insert_calls == 1


def test_review_submit_resolves_chapters_per_question(monkeypatch):
    fake = review_client(monkeypatch, HISTORY)
    created = qbr.create_question_bank_review_test(
        ReviewTestRequest(question_count=4, block_size=4), auth()
    )
    test_row = fake.tables["test_attempts"].rows[0]
    test_row.setdefault("status", "in_progress")  # DB column default
    answers = QuestionBankBlockSubmission(
        answers=[
            QuestionBankAnswer(question_id=question.id, answer="B")
            for question in created.questions
        ]
    )
    result = qbr.submit_question_bank_block(
        test_row["id"], 1, answers, auth()
    )
    assert result.completed is True

    question_chapter = {str(q["id"]): q["chapter_id"] for q in QUESTIONS}
    attempts = fake.tables["question_attempts"].rows
    assert len(attempts) == 4
    for attempt in attempts:
        assert attempt["chapter_id"] == question_chapter[str(attempt["question_id"])]

    concept_chapter = {CONCEPT_A: CHAPTER_A, CONCEPT_B: CHAPTER_B}
    mastery_rows = fake.tables["concept_mastery"].rows
    assert len(mastery_rows) == 2
    for row in mastery_rows:
        assert row["chapter_id"] == concept_chapter[str(row["concept_id"])]

    performance_rows = fake.tables["student_question_performance"].rows
    assert len(performance_rows) == 4
    for row in performance_rows:
        assert row["chapter_id"] == question_chapter[str(row["question_id"])]


def test_review_tests_skip_leaderboard_award(monkeypatch):
    fake = review_client(monkeypatch, HISTORY)
    created = qbr.create_question_bank_review_test(
        ReviewTestRequest(question_count=4, block_size=4), auth()
    )
    test_row = fake.tables["test_attempts"].rows[0]
    test_row.setdefault("status", "in_progress")  # DB column default
    answers = QuestionBankBlockSubmission(
        answers=[
            QuestionBankAnswer(question_id=question.id, answer="B")
            for question in created.questions
        ]
    )
    result = qbr.submit_question_bank_block(test_row["id"], 1, answers, auth())
    assert result.leaderboard_points_awarded is None
