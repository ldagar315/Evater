"""Core feedback-generation flow, extracted from the feedback router.

Pure Python - no FastAPI imports. LLM/OCR callables are injectable so the
core path is unit-testable with mocked dspy modules. None means "use the
real default", resolved at call time so tests can monkeypatch the module
attributes.
"""
from typing import Any, Dict, List, Tuple

from . import dspy_modules, services
from .models import Answer, DirectFeedbackRequest, Question


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
) -> List[Tuple[Question, Answer]]:
    """Pair each direct-feedback answer with its question, assembling the
    final answer text (typed text and/or OCR of an uploaded image)."""
    ocr_fn = ocr_fn or services.ocr_answer_images

    question_map = {q.question_number: q for q in request.questions}
    pairs: List[Tuple[Question, Answer]] = []
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
    ocr_fn=None,
    feedback_fn=None,
) -> List[Dict[str, Any]]:
    """gen_feedback_direct flow: one feedback item per answered question."""
    feedback_fn = feedback_fn or dspy_modules.feedback_generation
    return [
        feedback_fn(question=question, answer=answer.answer).feedback.model_dump()
        for question, answer in build_direct_answers(request, ocr_fn=ocr_fn)
    ]
