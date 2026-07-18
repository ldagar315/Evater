from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from pydantic import ValidationError

from .models import QuestionCandidate, QuestionItemCandidate


class QuestionPackError(ValueError):
    pass


def load_candidates(path: str | Path) -> List[QuestionCandidate]:
    payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_questions = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(raw_questions, list):
        raise QuestionPackError("Question file must contain a list or a questions array.")

    candidates: List[QuestionCandidate] = []
    errors: List[str] = []
    for index, raw_question in enumerate(raw_questions, start=1):
        try:
            candidates.append(QuestionCandidate.model_validate(raw_question))
        except ValidationError as exc:
            errors.append(f"question {index}: {exc.errors()[0].get('msg', 'invalid question')}")
    if errors:
        raise QuestionPackError("\n".join(errors))
    return candidates


def load_question_items(path: str | Path) -> List[QuestionItemCandidate]:
    payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_items = payload.get("items", []) if isinstance(payload, dict) else []
    if not isinstance(raw_items, list):
        raise QuestionPackError("Question file items must be a list.")

    items: List[QuestionItemCandidate] = []
    errors: List[str] = []
    for index, raw_item in enumerate(raw_items, start=1):
        try:
            items.append(QuestionItemCandidate.model_validate(raw_item))
        except ValidationError as exc:
            errors.append(f"item {index}: {exc.errors()[0].get('msg', 'invalid question item')}")
    if errors:
        raise QuestionPackError("\n".join(errors))
    return items


def validate_question_items(items: List[QuestionItemCandidate], minimum_count: int = 1) -> Dict[str, Any]:
    if len(items) < minimum_count:
        raise QuestionPackError(f"Expected at least {minimum_count} extracted question items; found {len(items)}.")
    hashes = [item.content_hash for item in items]
    if len(set(hashes)) != len(hashes):
        raise QuestionPackError("Extracted question items contain duplicate content hashes after deduplication.")
    return {
        "count": len(items),
        "question_types": dict(Counter(item.question_type for item in items)),
        "review_required": any(item.review_status != "approved" for item in items),
    }


def distribution(candidates: Iterable[QuestionCandidate]) -> Dict[str, Dict[str, int]]:
    questions = list(candidates)
    return {
        "difficulty": dict(Counter(question.difficulty for question in questions)),
        "cognitive_level": dict(Counter(question.cognitive_level for question in questions)),
        "question_style": dict(Counter(question.question_style for question in questions)),
        "review_status": dict(Counter(question.review_status for question in questions)),
        "concepts": dict(Counter(str(question.concept_id) for question in questions)),
    }


def validate_publishable_pack(
    candidates: List[QuestionCandidate],
    expected_count: int = 100,
) -> Dict[str, Any]:
    if len(candidates) != expected_count:
        raise QuestionPackError(
            f"Expected exactly {expected_count} candidates for publishing, found {len(candidates)}."
        )
    if any(question.review_status != "approved" for question in candidates):
        raise QuestionPackError("Every question must be approved before publishing.")
    hashes = [question.content_hash for question in candidates]
    if len(set(hashes)) != len(hashes):
        raise QuestionPackError("Question pack contains duplicate content hashes.")

    difficulty_counts = Counter(question.difficulty for question in candidates)
    expected_difficulty = {"easy": 40, "medium": 40, "hard": 20}
    if difficulty_counts != expected_difficulty:
        raise QuestionPackError(
            f"Difficulty distribution must be {expected_difficulty}; got {dict(difficulty_counts)}."
        )

    cognitive_counts = Counter(question.cognitive_level for question in candidates)
    expected_cognitive = {"recall": 30, "understand": 35, "apply": 25, "analyze": 10}
    if cognitive_counts != expected_cognitive:
        raise QuestionPackError(
            f"Cognitive distribution must be {expected_cognitive}; got {dict(cognitive_counts)}."
        )

    return {
        "count": len(candidates),
        "distribution": distribution(candidates),
        "publishable": True,
    }


def validate_scraped_pack(
    candidates: List[QuestionCandidate],
    minimum_count: int = 10,
) -> Dict[str, Any]:
    """Validate a source scrape without pretending it passed editorial review."""

    if len(candidates) < minimum_count:
        raise QuestionPackError(
            f"A scraped pack needs at least {minimum_count} candidates; found {len(candidates)}."
        )
    hashes = [question.content_hash for question in candidates]
    if len(set(hashes)) != len(hashes):
        raise QuestionPackError("Scraped pack contains duplicate content hashes after deduplication.")
    if any(question.review_status not in {"draft", "review"} for question in candidates):
        raise QuestionPackError("Scraped packs must remain draft/review until editorial approval.")
    if any(not question.source_question_id for question in candidates):
        raise QuestionPackError("Every scraped candidate needs its original source question id.")
    return {
        "count": len(candidates),
        "distribution": distribution(candidates),
        "publishable": False,
        "review_required": True,
    }
