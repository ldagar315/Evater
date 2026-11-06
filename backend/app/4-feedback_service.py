"""Core feedback-generation flow, extracted from the feedback router.

Pure Python - no FastAPI imports. LLM/OCR callables are injectable so the
core path is unit-testable with mocked dspy modules. None means "use the
real default", resolved at call time so tests can monkeypatch the module
attributes.
"""
from typing import Any, Dict, List, Optional, Tuple

from . import dspy_modules, services
from .models import Answer, DirectFeedbackRequest, PublicQuestion, Question

# Question types whose grading is meaningless without the generated answer key.
KEYED_QUESTION_TYPES = {"mcq_single", "mcq_multi", "true_false"}


class TestNotFound(Exception):
    """The referenced test does not exist or does not belong to the caller."""


class AnswerKeysUnavailable(Exception):
    """Objective questions need server-held keys and none exist for this test."""


def _default_test_loader(test_id: int, jwt: str) -> Optional[Dict[str, Any]]:
    from .supabase_client import create_supabase_client

    resp = (
        create_supabase_client(jwt)
        .table("Questions_Created")
        .select("id,created_by,test,key_id")
        .eq("id", test_id)
        .single()
        .execute()
    )
    return resp.data


def _default_key_loader(key_id: str) -> Optional[Dict[str, Any]]:
    from .supabase_client import create_supabase_service_client

    resp = (
        create_supabase_service_client()
        .table("generated_test_keys")
        .select("id,user_id,questions")
        .eq("id", key_id)
        .single()
        .execute()
    )
    return resp.data


def _has_keys(questions: List[Dict[str, Any]]) -> bool:
    return any(
        opt.get("is_correct") is not None
        for q in questions
        for opt in (q.get("options") or [])
    )


def _load_server_side_questions(
    test_id: int,
    *,
    jwt: str,
    user_id: str,
    answered_numbers: set,
    test_loader,
    key_loader,
) -> List[Dict[str, Any]]:
    """Load the test's questions with answer keys from server-side storage.

    Clients never supply the graded questions: everything comes from storage,
    with the ownership chain checked at every hop, so a submitted test_id or
    key_id cannot be used to read or grade against someone else's keys."""
    row = test_loader(test_id, jwt)
    if not row or str(row.get("created_by")) != str(user_id):
        raise TestNotFound()

    key_id = row.get("key_id")
    if key_id:
        key_row = key_loader(str(key_id))
        if not key_row or str(key_row.get("user_id")) != str(user_id):
            raise AnswerKeysUnavailable()
        return key_row["questions"]

    stored = row.get("test") or []
    if _has_keys(stored):
        # Legacy rows predate the key store; their keys already live in the
        # row, readable server-side (never returned to the client).
        return stored

    # Sanitized row without a key-store entry: objective questions cannot be
    # graded reliably. Fail closed instead of guessing. Types come from the
    # stored row, never the client payload.
    answered_types = {
        q.get("question_type") for q in stored if q.get("question_number") in answered_numbers
    }
    if answered_types & KEYED_QUESTION_TYPES:
        raise AnswerKeysUnavailable()
    return stored


def generate_merged_feedback(
    image_urls: List[str],
    questions_payload: Dict[str, Any],
    *,
    ocr_fn=None,
    separate_fn=None,
    feedback_fn=None,
) -> List[Dict[str, Any]]:
    """gen_answer flow: OCR the answer sheet, split per-question answers,
    generate feedback for each, and merge questions+answers+feedback."""
    ocr_fn = ocr_fn or services.ocr_answer_images
    separate_fn = separate_fn or dspy_modules.answer_seperation
    feedback_fn = feedback_fn or dspy_modules.feedback_generation

    text = ocr_fn(image_urls)
    separated = separate_fn(answer_sheet_text=text)

    question_map = {
        q.question_number: q
        for q in (Question.model_validate(q) for q in questions_payload["questions"])
    }

    feedback_items = []
    for answer in separated.answers:
        question = question_map.get(answer.question_number)
        if question:
            feedback = feedback_fn(question=question, answer=answer.answer)
            feedback_items.append(feedback.feedback.model_dump())

    return services.merge_qaf(
        questions_list=questions_payload["questions"],
        answer_list=[a.model_dump() for a in separated.answers],
        feedback_list_new=feedback_items,
    )


def build_direct_answers(
    request: DirectFeedbackRequest,
    *,
    ocr_fn=None,
    questions: Optional[List[Any]] = None,
) -> List[Tuple[Any, Answer]]:
    """Pair each direct-feedback answer with its question, assembling the
    final answer text (typed text and/or OCR of an uploaded image).

    `questions` overrides the client-sent payload (server-loaded rows)."""
    ocr_fn = ocr_fn or services.ocr_answer_images

    source = questions if questions is not None else request.questions
    question_map = {q.question_number: q for q in source}
    pairs: List[Tuple[Any, Answer]] = []
    for ans_input in request.answers:
        question = question_map.get(ans_input.question_number)
        if not question:
            continue

        final_answer_text = ans_input.answer_text or ""
        if ans_input.image_url:
            ocr_result = ocr_fn([ans_input.image_url])
            if final_answer_text:
                final_answer_text += f"\n\n[Image Content]: {ocr_result}"
            else:
                final_answer_text = ocr_result
        if not final_answer_text:
            final_answer_text = "No answer provided."

        pairs.append((question, Answer(
            question_number=ans_input.question_number,
            answer=final_answer_text,
        )))
    return pairs


def generate_direct_feedback(
    request: DirectFeedbackRequest,
    *,
    jwt: Optional[str] = None,
    user_id: Optional[str] = None,
    ocr_fn=None,
    feedback_fn=None,
    test_loader=None,
    key_loader=None,
) -> List[Dict[str, Any]]:
    """gen_feedback_direct flow: one feedback item per answered question.

    With test_id, questions and answer keys come from server-side storage
    (ownership-checked) and grading is anchored to the generated key. Without
    test_id, only non-keyed question types may be graded - objective questions
    fail closed rather than being re-derived by the LLM."""
    feedback_fn = feedback_fn or dspy_modules.feedback_generation
    answered_numbers = {a.question_number for a in request.answers}

    questions: Optional[List[Any]] = None
    if request.test_id is not None:
        if not jwt or not user_id:
            raise TestNotFound()
        rows = _load_server_side_questions(
            request.test_id,
            jwt=jwt,
            user_id=user_id,
            answered_numbers=answered_numbers,
            test_loader=test_loader or _default_test_loader,
            key_loader=key_loader or _default_key_loader,
        )
        questions = [Question.model_validate(r) for r in rows]
    else:
        answered_types = {
            q.question_type
            for q in request.questions
            if q.question_number in answered_numbers
        }
        if answered_types & KEYED_QUESTION_TYPES:
            raise AnswerKeysUnavailable()

    return [
        feedback_fn(question=question, answer=answer.answer).feedback.model_dump()
        for question, answer in build_direct_answers(request, ocr_fn=ocr_fn, questions=questions)
    ]
