"""Deterministic question-bank selection, scoring, and routing primitives."""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


DIFFICULTIES = ("easy", "medium", "hard")
DIFFICULTY_INDEX = {value: index for index, value in enumerate(DIFFICULTIES)}
QUESTION_TYPES = (
    "mcq_single",
    "mcq_multi",
    "assertion_reason",
    "true_false",
    "fill_blank",
    "numerical",
    "case_study",
    "diagram_based",
    "matching",
)


class QuestionBankError(ValueError):
    """A content or request error that should be returned as HTTP 400."""


def _normalize_media(raw_media: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_media, list):
        return []

    return [
        media
        for media in raw_media
        if isinstance(media, dict) and (media.get("url") or media.get("src") or media.get("storage_path"))
    ]


def normalize_options(
    raw_options: Any,
    option_media: Any = None,
    *,
    exact_count: Optional[int] = 4,
) -> List[Dict[str, Any]]:
    if not isinstance(raw_options, list):
        raise QuestionBankError("Question options must be a list.")

    options: List[Dict[str, Any]] = []
    seen_ids = set()
    for raw_option in raw_options:
        if not isinstance(raw_option, dict):
            raise QuestionBankError("Each question option must be an object.")
        option_id = str(raw_option.get("id", "")).strip()
        option_text = str(raw_option.get("text", "")).strip()
        if not option_id or not option_text:
            raise QuestionBankError("Each option needs a non-empty id and text.")
        if option_id in seen_ids:
            raise QuestionBankError("Question options must have unique ids.")
        seen_ids.add(option_id)
        media_by_option = option_media if isinstance(option_media, dict) else {}
        options.append(
            {
                "id": option_id,
                "text": option_text,
                "media": _normalize_media(media_by_option.get(option_id, raw_option.get("media", []))),
            }
        )

    if exact_count is not None and len(options) != exact_count:
        raise QuestionBankError(f"Expected exactly {exact_count} question options.")
    return options


def normalize_question(row: Mapping[str, Any]) -> Dict[str, Any]:
    question = dict(row)
    question["id"] = str(question["id"])
    question["concept_id"] = str(question["concept_id"])
    question["question_type"] = str(question.get("question_type") or "mcq_single")
    if question["question_type"] not in QUESTION_TYPES:
        raise QuestionBankError(f"Unsupported question type: {question['question_type']}.")
    question["media"] = _normalize_media(question.get("media_json", question.get("media")))
    question["options"] = normalize_options(
        question.get("options_json", question.get("options")),
        question.get("option_media_json", question.get("option_media")),
        exact_count=4 if question["question_type"] == "mcq_single" else None,
    )
    if question["question_type"] == "mcq_multi" and len(question["options"]) < 2:
        raise QuestionBankError("Multiple-select questions need at least two options.")
    question["render_config"] = question.get("render_config") or {}
    question["difficulty"] = str(question["difficulty"]).lower()
    if question["difficulty"] not in DIFFICULTIES:
        raise QuestionBankError("Question difficulty must be easy, medium, or hard.")
    return question


def to_public_question(row: Mapping[str, Any]) -> Dict[str, Any]:
    question = normalize_question(row)
    return {
        "id": question["id"],
        "concept_id": question["concept_id"],
        "question_type": question["question_type"],
        "question_text": question["question_text"],
        "options": question["options"],
        "media": question["media"],
        "render_config": question["render_config"],
        "explanation": question.get("explanation"),
        "hint": question.get("hint"),
        "difficulty": question["difficulty"],
        "cognitive_level": question["cognitive_level"],
        "skill_tags": question.get("skill_tags") or [],
        "misconception_tags": question.get("misconception_tags") or [],
        "question_style": question.get("question_style") or "direct",
        "estimated_time_seconds": question.get("estimated_time_seconds") or 60,
        "maximum_marks": question.get("marks") or 1,
    }


def _target_difficulty(
    mode: str,
    block_number: int,
    routing: Optional[Mapping[str, Any]],
) -> str:
    if routing and routing.get("difficulty") in DIFFICULTIES:
        return str(routing["difficulty"])
    if mode == "remedial":
        return "easy"
    if mode == "challenge":
        return "hard"
    if mode == "diagnostic" and block_number == 1:
        return "medium"
    return "medium"


def select_question_block(
    questions: Sequence[Mapping[str, Any]],
    mastery: Mapping[str, float],
    seen_question_ids: Iterable[str],
    seed: int,
    count: int,
    mode: str = "practice",
    block_number: int = 1,
    routing: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Select a deterministic, lightly adaptive block from published questions."""

    if count < 1:
        raise QuestionBankError("Question block size must be positive.")

    normalized = [normalize_question(question) for question in questions]
    if len(normalized) < count:
        raise QuestionBankError(
            f"Not enough published questions for this block: requested {count}, found {len(normalized)}."
        )

    seen_ids = {str(question_id) for question_id in seen_question_ids}
    unseen = [question for question in normalized if question["id"] not in seen_ids]
    pool = unseen if len(unseen) >= count else normalized
    target = _target_difficulty(mode, block_number, routing)
    focus_ids = {str(value) for value in (routing or {}).get("focus_concept_ids", [])}

    rng = random.Random(seed + block_number * 1009)
    rng.shuffle(pool)
    concept_counts: Dict[str, int] = defaultdict(int)

    def rank(question: Mapping[str, Any]) -> tuple:
        concept_id = str(question["concept_id"])
        difficulty_distance = abs(DIFFICULTY_INDEX[question["difficulty"]] - DIFFICULTY_INDEX[target])
        concept_mastery = float(mastery.get(concept_id, 0))
        weak_concept_rank = -1 if concept_id in focus_ids else 0
        return (
            weak_concept_rank,
            difficulty_distance,
            concept_counts[concept_id],
            concept_mastery,
        )

    ranked = sorted(pool, key=rank)
    selected: List[Dict[str, Any]] = []
    for question in ranked:
        if len(selected) >= count:
            break
        selected.append(question)
        concept_counts[str(question["concept_id"])] += 1

    if len(selected) != count:
        raise QuestionBankError("Unable to build a complete question block.")

    return {
        "questions": selected,
        "difficulty": target,
        "focus_concept_ids": sorted(focus_ids),
        "selection_reason": f"{mode}:{target}:block-{block_number}",
    }


def _clean_answer(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def _answer_is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _option_text(question: Mapping[str, Any], option_id: Any) -> str | None:
    option_key = str(option_id).strip()
    for option in question.get("options", []):
        if str(option.get("id")) == option_key:
            return str(option.get("text") or option_key)
    return None


def _format_answer(question: Mapping[str, Any], answer: Any) -> str:
    if _answer_is_empty(answer):
        return "Not answered"
    if isinstance(answer, dict):
        return "; ".join(f"{key}: {value}" for key, value in answer.items())
    if isinstance(answer, list):
        return ", ".join(_format_answer(question, value) for value in answer)
    option_text = _option_text(question, answer)
    return f"{answer}. {option_text}" if option_text and str(answer) in {"A", "B", "C", "D"} else str(answer)


def _correct_summary(question: Mapping[str, Any], answer_spec: Mapping[str, Any]) -> str:
    correct_option_ids = answer_spec.get("correct_option_ids")
    if isinstance(correct_option_ids, list):
        return ", ".join(_format_answer(question, value) for value in correct_option_ids)
    correct_option_id = answer_spec.get("correct_option_id") or question.get("correct_option_id")
    if correct_option_id:
        return _format_answer(question, correct_option_id)
    accepted = answer_spec.get("accepted_answers") or answer_spec.get("accepted_values")
    if isinstance(accepted, list):
        return " or ".join(str(value) for value in accepted)
    if "expected_value" in answer_spec:
        return str(answer_spec["expected_value"])
    pairs = answer_spec.get("pairs")
    if isinstance(pairs, dict):
        return "; ".join(f"{key}: {value}" for key, value in pairs.items())
    return "Answer key unavailable"


def _score_deterministic_question(question: Mapping[str, Any], submitted: Any) -> Dict[str, Any]:
    question_type = str(question["question_type"])
    answer_spec = question.get("answer_spec") or {}
    maximum_marks = int(question.get("marks") or 1)
    correct_option_id = answer_spec.get("correct_option_id") or question.get("correct_option_id")
    correct_option_ids = answer_spec.get("correct_option_ids")
    is_correct = False
    marks_awarded = 0

    if isinstance(correct_option_ids, list) and correct_option_ids:
        submitted_ids = submitted if isinstance(submitted, list) else []
        is_correct = {
            _clean_answer(value) for value in submitted_ids
        } == {
            _clean_answer(value) for value in correct_option_ids
        }
        marks_awarded = maximum_marks if is_correct else 0
    elif correct_option_id:
        is_correct = not _answer_is_empty(submitted) and _clean_answer(submitted) == _clean_answer(correct_option_id)
        marks_awarded = maximum_marks if is_correct else 0
    elif question_type == "numerical" and "expected_value" in answer_spec:
        try:
            actual = float(submitted)
            expected = float(answer_spec["expected_value"])
            tolerance = float(answer_spec.get("tolerance", 0))
            relative_tolerance = float(answer_spec.get("relative_tolerance", 0))
            allowed = max(tolerance, abs(expected) * relative_tolerance)
            is_correct = math.isfinite(actual) and abs(actual - expected) <= allowed
        except (TypeError, ValueError):
            is_correct = False
        marks_awarded = maximum_marks if is_correct else 0
    elif question_type in {"true_false", "fill_blank"}:
        accepted = answer_spec.get("accepted_answers") or answer_spec.get("accepted_values")
        if accepted is None and "expected_value" in answer_spec:
            accepted = [answer_spec["expected_value"]]
        accepted_values = accepted if isinstance(accepted, list) else [accepted]
        is_correct = any(_clean_answer(submitted) == _clean_answer(value) for value in accepted_values if value is not None)
        marks_awarded = maximum_marks if is_correct else 0
    elif question_type == "matching":
        expected_pairs = answer_spec.get("pairs")
        actual_pairs = submitted if isinstance(submitted, dict) else {}
        if isinstance(expected_pairs, dict) and expected_pairs:
            correct_pairs = sum(
                1
                for key, value in expected_pairs.items()
                if _clean_answer(actual_pairs.get(key)) == _clean_answer(value)
            )
            is_correct = correct_pairs == len(expected_pairs)
            marks_awarded = round(maximum_marks * correct_pairs / len(expected_pairs))

    return {
        "is_correct": is_correct,
        "marks_awarded": marks_awarded,
        "maximum_marks": maximum_marks,
        "answer_summary": _format_answer(question, submitted),
        "correct_answer_summary": _correct_summary(question, answer_spec),
        "selected_option_id": submitted if isinstance(submitted, str) and submitted in {"A", "B", "C", "D"} else None,
        "correct_option_id": correct_option_id if correct_option_id in {"A", "B", "C", "D"} else None,
        "selected_option_ids": [str(value) for value in submitted] if isinstance(submitted, list) else [],
        "correct_option_ids": [str(value) for value in correct_option_ids] if isinstance(correct_option_ids, list) else [],
    }


def score_block(
    questions: Sequence[Mapping[str, Any]],
    submitted_answers: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Score all supported learner question types with stored deterministic rules."""

    results: List[Dict[str, Any]] = []
    for question in questions:
        normalized = normalize_question(question)
        question_id = normalized["id"]
        scored = _score_deterministic_question(normalized, submitted_answers.get(question_id))
        results.append(
            {
                "question_id": question_id,
                "concept_id": normalized["concept_id"],
                **scored,
                "explanation": normalized.get("explanation"),
                "difficulty": normalized["difficulty"],
            }
        )
    return results


def build_routing_decision(
    results: Sequence[Mapping[str, Any]],
    previous_difficulty: Optional[str] = None,
) -> Dict[str, Any]:
    if not results:
        return {"difficulty": previous_difficulty or "medium", "focus_concept_ids": [], "status": "on_track"}

    total_marks = sum(int(result.get("maximum_marks") or 1) for result in results)
    score = sum(int(result.get("marks_awarded") or 0) for result in results)
    percentage = score / total_marks if total_marks else 0

    by_concept: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        by_concept[str(result["concept_id"])].append(result)

    weak_concepts = []
    for concept_id, concept_results in by_concept.items():
        concept_total = len(concept_results)
        concept_correct = sum(1 for result in concept_results if result.get("is_correct"))
        concept_accuracy = concept_correct / concept_total
        if concept_accuracy < 0.6:
            weak_concepts.append((concept_accuracy, concept_id))
    weak_concepts.sort(key=lambda item: (item[0], item[1]))

    previous = previous_difficulty if previous_difficulty in DIFFICULTIES else "medium"
    if percentage < 0.40:
        difficulty = "easy"
        status = "needs_review"
    elif percentage > 0.75:
        difficulty = DIFFICULTIES[min(DIFFICULTY_INDEX[previous] + 1, len(DIFFICULTIES) - 1)]
        status = "challenge_next"
    else:
        difficulty = previous
        status = "on_track"

    return {
        "difficulty": difficulty,
        "focus_concept_ids": [concept_id for _, concept_id in weak_concepts[:3]],
        "status": status,
    }


def update_mastery(
    results: Sequence[Mapping[str, Any]],
    existing_mastery: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result["concept_id"])].append(result)

    updates: List[Dict[str, Any]] = []
    for concept_id, concept_results in grouped.items():
        previous = existing_mastery.get(concept_id, {})
        previous_mastery = float(previous.get("mastery_score") or 0)
        correct_count = sum(1 for result in concept_results if result.get("is_correct"))
        accuracy = correct_count / len(concept_results)
        mastery_score = round((0.70 * previous_mastery) + (0.30 * accuracy * 100), 2)
        updates.append(
            {
                "concept_id": concept_id,
                "mastery_score": mastery_score,
                "attempt_count": int(previous.get("attempt_count") or 0) + len(concept_results),
                "correct_count": int(previous.get("correct_count") or 0) + correct_count,
                "last_difficulty": concept_results[-1].get("difficulty"),
                "accuracy": round(accuracy * 100, 2),
                "status": (
                    "needs_review" if mastery_score < 45
                    else "challenge_next" if mastery_score >= 75
                    else "on_track"
                ),
            }
        )
    return updates
