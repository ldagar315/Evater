"""Mocked-LLM tests for the core feedback path (feedback_service)."""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import feedback_service
from app.models import Answer, AnswerInput, DirectFeedbackRequest, Feedback, Question

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
        questions=[Question.model_validate(Q1), Question.model_validate(Q2)],
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
        questions=[Question.model_validate(Q1)],
        answers=[AnswerInput(question_number=1)],
    )
    pairs = feedback_service.build_direct_answers(req, ocr_fn=lambda urls: "unused")
    assert pairs[0][1].answer == "No answer provided."


def test_generate_direct_feedback_maps_each_pair():
    req = DirectFeedbackRequest(
        questions=[Question.model_validate(Q1), Question.model_validate(Q2)],
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
    from app.models import DirectFeedbackResponse, MergedFeedbackResponse, QuestionsResponse
    from app.routers import feedback_router, test_router

    fb = {r.path: r for r in feedback_router.router.routes}
    tg = {r.path: r for r in test_router.router.routes}
    assert fb["/gen_answer"].response_model is MergedFeedbackResponse
    assert fb["/gen_feedback_direct"].response_model is DirectFeedbackResponse
    assert tg["/gen_question"].response_model is QuestionsResponse


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
