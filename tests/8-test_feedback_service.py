"""Mocked-LLM tests for the core feedback path (feedback_service)."""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import feedback_service
from app.models import (Answer, AnswerInput, DirectFeedbackRequest, Feedback,
                         PublicQuestion, Question)

Q1 = {"question_text": "q1?", "question_type": "short_answer", "question_number": 1,
      "maximum_marks": 2, "difficulty": "Easy", "contains_math_expression": False,
      "options": None, "correct_answer": "kept-extra"}
Q2 = {"question_text": "q2?", "question_type": "long_answer", "question_number": 2,
      "maximum_marks": 3, "difficulty": "Medium", "contains_math_expression": True,
      "options": None}


def _fb(n):
    return types.SimpleNamespace(feedback=Feedback(
        question_number=n, max_scored=2, error_type="No mistake",
        explanation=f"expl{n}", next_step=f"next{n}"))


def test_merged_feedback_orders_and_preserves_extras():
    calls = []

    def feedback_fn(*, question, answer):
        calls.append((question.question_number, answer))
        return _fb(question.question_number)

    separated = types.SimpleNamespace(answers=[
        Answer(question_number=2, answer="a2"),
        Answer(question_number=1, answer="a1"),
        Answer(question_number=99, answer="no-matching-question"),
    ])
    merged = feedback_service.generate_merged_feedback(
        ["https://x/sheet.png"], {"questions": [Q1, Q2]},
        ocr_fn=lambda urls: "ocr text",
        separate_fn=lambda *, answer_sheet_text: separated,
        feedback_fn=feedback_fn,
    )
    # feedback called once per matched answer, in answer order
    assert calls == [(2, "a2"), (1, "a1")]
    # merged follows client question order, extras preserved
    assert [m["question_number"] for m in merged] == [1, 2]
    assert merged[0]["correct_answer"] == "kept-extra"
    assert merged[0]["answer"]["answer"] == "a1"
    assert merged[0]["feedback"]["explanation"] == "expl1"
    assert merged[1]["answer"]["answer"] == "a2"
    # the unmatched answer (99) produced no feedback call and is not a question row


def test_build_direct_answers_text_image_combination():
    req = DirectFeedbackRequest(
        questions=[PublicQuestion.model_validate(Q1), PublicQuestion.model_validate(Q2)],
        answers=[
            AnswerInput(question_number=1, answer_text="typed", image_url="https://x/img.png"),
            AnswerInput(question_number=2, image_url="https://x/img2.png"),
            AnswerInput(question_number=99, answer_text="orphan"),
        ],
    )
    pairs = feedback_service.build_direct_answers(req, ocr_fn=lambda urls: f"OCR[{urls[0]}]")
    assert [p[1].question_number for p in pairs] == [1, 2]
    assert pairs[0][1].answer == "typed\n\n[Image Content]: OCR[https://x/img.png]"
    assert pairs[1][1].answer == "OCR[https://x/img2.png]"


def test_build_direct_answers_empty_answer_placeholder():
    req = DirectFeedbackRequest(
        questions=[PublicQuestion.model_validate(Q1)],
        answers=[AnswerInput(question_number=1)],
    )
    pairs = feedback_service.build_direct_answers(req, ocr_fn=lambda urls: "unused")
    assert pairs[0][1].answer == "No answer provided."


def test_generate_direct_feedback_maps_each_pair():
    req = DirectFeedbackRequest(
        questions=[PublicQuestion.model_validate(Q1), PublicQuestion.model_validate(Q2)],
        answers=[AnswerInput(question_number=1, answer_text="a1"),
                 AnswerInput(question_number=2, answer_text="a2")],
    )
    out = feedback_service.generate_direct_feedback(
        req,
        ocr_fn=lambda urls: "unused",
        feedback_fn=lambda *, question, answer: _fb(question.question_number),
    )
    assert [f["question_number"] for f in out] == [1, 2]
    assert out[0]["error_type"] == "No mistake"


def test_gen_feedback_direct_route_through_response_model(monkeypatch):
    """Route-level: gen_feedback_direct validated by DirectFeedbackResponse."""
    from datetime import datetime, timezone
    from fastapi.testclient import TestClient
    from supabase_auth.types import User

    import app.dspy_modules as dspy_modules_module
    from app.auth import AuthContext, require_user
    from app.main import app

    app.dependency_overrides[require_user] = lambda: AuthContext(
        jwt="j", tenant_id="t", user=User(
            id="u1", app_metadata={}, user_metadata={}, aud="authenticated",
            email="u@e.com", created_at=datetime.now(timezone.utc)))
    monkeypatch.setattr(dspy_modules_module, "feedback_generation",
                        lambda *, question, answer: _fb(question.question_number))
    try:
        client = TestClient(app)
        resp = client.post("/api/gen_feedback_direct", json={
            "questions": [Q1],
            "answers": [{"question_number": 1, "answer_text": "a1"}],
        })
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"feedback": [{
            "question_number": 1, "explanation": "expl1", "max_scored": 2,
            "error_type": "No mistake", "next_step": "next1"}]}
    finally:
        app.dependency_overrides.clear()


def test_response_models_wired_in_router_metadata():
    from app.models import (DirectFeedbackResponse, MergedFeedbackResponse,
                            PublicQuestionsResponse)
    from app.routers import feedback_router, test_router

    fb = {r.path: r for r in feedback_router.router.routes}
    tg = {r.path: r for r in test_router.router.routes}
    assert fb["/gen_answer"].response_model is MergedFeedbackResponse
    assert fb["/gen_feedback_direct"].response_model is DirectFeedbackResponse
    assert tg["/gen_question"].response_model is PublicQuestionsResponse


def test_invalid_llm_payload_rejected_by_wired_response_model(monkeypatch):
    """Negative fixture: an LLM feedback payload violating the Feedback model
    (error_type outside the Literal) must be rejected by the wired response
    model - a 500, not a 200 passing bad data to the client."""
    from datetime import datetime, timezone
    from fastapi.testclient import TestClient
    from supabase_auth.types import User

    import app.dspy_modules as dspy_modules_module
    from app.auth import AuthContext, require_user
    from app.main import app

    app.dependency_overrides[require_user] = lambda: AuthContext(
        jwt="j", tenant_id="t", user=User(
            id="u1", app_metadata={}, user_metadata={}, aud="authenticated",
            email="u@e.com", created_at=datetime.now(timezone.utc)))

    def bad_feedback(*, question, answer):
        # Bypasses the Feedback model's own validation to simulate a raw LLM
        # payload with an out-of-vocabulary error_type reaching the response.
        bad = {"question_number": question.question_number, "explanation": "x",
               "max_scored": 1, "error_type": "none", "next_step": "y"}
        return types.SimpleNamespace(feedback=types.SimpleNamespace(model_dump=lambda: bad))

    monkeypatch.setattr(dspy_modules_module, "feedback_generation", bad_feedback)
    try:
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/gen_feedback_direct", json={
            "questions": [Q1],
            "answers": [{"question_number": 1, "answer_text": "a1"}],
        })
        assert resp.status_code == 500, resp.text
        assert resp.json() == {"detail": "Internal server error."}
    finally:
        app.dependency_overrides.clear()


def test_merged_item_missing_required_field_rejected(monkeypatch):
    """Negative fixture: a merged row missing the required question_text must
    fail MergedFeedbackResponse validation rather than reach the client."""
    from datetime import datetime, timezone
    from fastapi.testclient import TestClient
    from supabase_auth.types import User

    import app.dspy_modules as dspy_modules_module
    import app.services as services_module
    from app.auth import AuthContext, require_user
    from app.main import app

    app.dependency_overrides[require_user] = lambda: AuthContext(
        jwt="j", tenant_id="t", user=User(
            id="u1", app_metadata={}, user_metadata={}, aud="authenticated",
            email="u@e.com", created_at=datetime.now(timezone.utc)))

    monkeypatch.setattr(services_module, "ocr_answer_images", lambda urls: "ocr text")
    monkeypatch.setattr(dspy_modules_module, "answer_seperation",
                        lambda *, answer_sheet_text: types.SimpleNamespace(
                            answers=[Answer(question_number=1, answer="a1")]))
    monkeypatch.setattr(dspy_modules_module, "feedback_generation",
                        lambda *, question, answer: _fb(1))
    # Question payload missing question_text - passes Question.model_validate?
    # (question_text is required, so the service itself rejects it; either way
    # the client must not receive a malformed merged row.)
    q_bad = {k: v for k, v in Q1.items() if k != "question_text"}
    try:
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/gen_answer", json={
            "image_url": ["https://x/a.png"], "questions": {"questions": [q_bad]}})
        assert resp.status_code == 500, resp.text
    finally:
        app.dependency_overrides.clear()


Q_MCQ_SANITIZED = {
    "question_text": "Pick the even number.",
    "question_type": "mcq_single",
    "question_number": 3,
    "difficulty": "Easy",
    "contains_math_expression": False,
    "maximum_marks": 1,
    "options": [{"text": "3"}, {"text": "4"}],  # no is_correct: sanitized wire
}

Q_MCQ_KEYED = {
    "question_text": "Pick the even number.",
    "question_type": "mcq_single",
    "question_number": 3,
    "difficulty": "Easy",
    "contains_math_expression": False,
    "maximum_marks": 1,
    "options": [{"text": "3", "is_correct": False}, {"text": "4", "is_correct": True}],
}


def _capture_feedback(captured):
    def capture_feedback(*, question, answer):
        captured["question"] = question
        captured["answer"] = answer
        return types.SimpleNamespace(feedback=types.SimpleNamespace(model_dump=lambda: {
            "question_number": 3, "explanation": "x", "max_scored": 1,
            "error_type": "none", "next_step": "y"}))
    return capture_feedback


def test_mcq_without_test_id_fails_closed():
    """Objective questions are never graded without server-held keys: no
    silent LLM re-derivation."""
    import pytest
    from app import feedback_service
    from app.models import DirectFeedbackRequest

    req = DirectFeedbackRequest(
        questions=[Q_MCQ_SANITIZED],
        answers=[{"question_number": 3, "answer_text": "4"}],
    )
    with pytest.raises(feedback_service.AnswerKeysUnavailable):
        feedback_service.generate_direct_feedback(req)


def test_written_answer_without_test_id_still_works():
    """Non-keyed types (short/long answer) grade key-free as before."""
    from app import feedback_service
    from app.models import DirectFeedbackRequest

    captured = {}
    req = DirectFeedbackRequest(
        questions=[Q1],
        answers=[{"question_number": 1, "answer_text": "a1"}],
    )
    out = feedback_service.generate_direct_feedback(
        req, feedback_fn=_capture_feedback(captured))
    assert len(out) == 1
    assert captured["question"].question_text == "q1?"


def test_test_id_hydrates_keys_from_store_and_ignores_client_questions():
    """With test_id, grading uses server-stored keyed questions; the
    client-sent payload (even a substituted question text) is ignored."""
    from app import feedback_service
    from app.models import DirectFeedbackRequest

    captured = {}
    bogus = dict(Q_MCQ_SANITIZED)
    bogus["question_text"] = "SUBSTITUTED BY CLIENT"
    req = DirectFeedbackRequest(
        questions=[bogus],
        answers=[{"question_number": 3, "answer_text": "4"}],
        test_id=42,
    )
    out = feedback_service.generate_direct_feedback(
        req, jwt="j", user_id="u1",
        feedback_fn=_capture_feedback(captured),
        test_loader=lambda tid, jwt: {"id": 42, "created_by": "u1",
                                      "test": [Q_MCQ_SANITIZED], "key_id": "key-1"},
        key_loader=lambda kid: {"id": "key-1", "user_id": "u1",
                                "questions": [Q_MCQ_KEYED]},
    )
    assert len(out) == 1
    graded = captured["question"]
    assert graded.question_text == "Pick the even number."  # server copy won
    assert any(o.is_correct for o in graded.options)  # keys hydrated


def test_test_id_legacy_row_keys_used_when_no_key_store_link():
    """Pre-migration rows keep key-grounded grading: keys come from the
    row's own stored test json (server-side), never from the client."""
    from app import feedback_service
    from app.models import DirectFeedbackRequest

    captured = {}
    req = DirectFeedbackRequest(
        questions=[Q_MCQ_SANITIZED],
        answers=[{"question_number": 3, "answer_text": "4"}],
        test_id=7,
    )
    feedback_service.generate_direct_feedback(
        req, jwt="j", user_id="u1",
        feedback_fn=_capture_feedback(captured),
        test_loader=lambda tid, jwt: {"id": 7, "created_by": "u1",
                                      "test": [Q_MCQ_KEYED], "key_id": None},
        key_loader=lambda kid: None,
    )
    assert any(o.is_correct for o in captured["question"].options)


def test_test_id_other_users_test_rejected():
    """Ownership: someone else's test_id is indistinguishable from missing."""
    import pytest
    from app import feedback_service
    from app.models import DirectFeedbackRequest

    req = DirectFeedbackRequest(
        questions=[Q_MCQ_SANITIZED],
        answers=[{"question_number": 3, "answer_text": "4"}],
        test_id=99,
    )
    with pytest.raises(feedback_service.TestNotFound):
        feedback_service.generate_direct_feedback(
            req, jwt="j", user_id="u1",
            test_loader=lambda tid, jwt: {"id": 99, "created_by": "OTHER",
                                          "test": [Q_MCQ_KEYED], "key_id": None},
            key_loader=lambda kid: None,
        )


def test_test_id_mismatched_key_row_rejected():
    """Ownership chain: a key-store row owned by someone else fails closed."""
    import pytest
    from app import feedback_service
    from app.models import DirectFeedbackRequest

    req = DirectFeedbackRequest(
        questions=[Q_MCQ_SANITIZED],
        answers=[{"question_number": 3, "answer_text": "4"}],
        test_id=42,
    )
    with pytest.raises(feedback_service.AnswerKeysUnavailable):
        feedback_service.generate_direct_feedback(
            req, jwt="j", user_id="u1",
            test_loader=lambda tid, jwt: {"id": 42, "created_by": "u1",
                                          "test": [Q_MCQ_SANITIZED], "key_id": "key-1"},
            key_loader=lambda kid: {"id": "key-1", "user_id": "OTHER",
                                    "questions": [Q_MCQ_KEYED]},
        )


def test_test_id_sanitized_row_without_keys_fails_closed_for_mcq():
    import pytest
    from app import feedback_service
    from app.models import DirectFeedbackRequest

    req = DirectFeedbackRequest(
        questions=[Q_MCQ_SANITIZED],
        answers=[{"question_number": 3, "answer_text": "4"}],
        test_id=55,
    )
    with pytest.raises(feedback_service.AnswerKeysUnavailable):
        feedback_service.generate_direct_feedback(
            req, jwt="j", user_id="u1",
            test_loader=lambda tid, jwt: {"id": 55, "created_by": "u1",
                                          "test": [Q_MCQ_SANITIZED], "key_id": None},
            key_loader=lambda kid: None,
        )
