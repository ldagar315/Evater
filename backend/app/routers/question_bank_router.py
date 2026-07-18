from __future__ import annotations

import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from ..auth import AuthContext, require_user
from ..models import (
    QuestionBankAnswer,
    QuestionBankBlockResponse,
    QuestionBankBlockSubmission,
    QuestionFlagResponse,
    QuestionBankTestRequest,
    QuestionBankTestResponse,
)
from ..question_bank import (
    QuestionBankError,
    build_routing_decision,
    score_block,
    select_question_block,
    to_public_question,
    update_mastery,
)
from ..supabase_client import create_supabase_client, create_supabase_service_client

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(require_user)])

PUBLIC_QUESTION_COLUMNS = (
    "id,chapter_id,concept_id,question_type,question_text,options_json,explanation,hint,"
    "difficulty,cognitive_level,skill_tags,misconception_tags,question_style,"
    "estimated_time_seconds,marks,media_json,option_media_json,render_config"
)


def _user_id(auth: AuthContext) -> str:
    return str(auth.user.id)


def _one(data: Any) -> Dict[str, Any] | None:
    if isinstance(data, list):
        return data[0] if data else None
    return data if isinstance(data, dict) else None


def _mastery_map(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row["concept_id"]): row for row in rows}


def _submitted_value(answer: QuestionBankAnswer) -> Any:
    return answer.answer if answer.answer is not None else answer.selected_option_id


def _submitted_json(answer: QuestionBankAnswer) -> Dict[str, Any]:
    value = _submitted_value(answer)
    return value if isinstance(value, dict) else {"value": value}


def _question_response(
    test_id: str,
    block_number: int,
    block_size: int,
    question_count: int,
    questions: List[Dict[str, Any]],
    routing: Dict[str, Any],
) -> QuestionBankTestResponse:
    return QuestionBankTestResponse(
        test_id=test_id,
        block_number=block_number,
        block_size=block_size,
        question_count=question_count,
        questions=[to_public_question(question) for question in questions],
        routing=routing,
    )


def _load_published_questions(client: Any, chapter_id: str) -> List[Dict[str, Any]]:
    response = (
        client.table("question_bank")
        .select(PUBLIC_QUESTION_COLUMNS)
        .eq("chapter_id", chapter_id)
        .eq("status", "published")
        .execute()
    )
    return response.data or []


def _load_mastery(client: Any, user_id: str, chapter_id: str) -> List[Dict[str, Any]]:
    response = (
        client.table("concept_mastery")
        .select("concept_id,mastery_score,attempt_count,correct_count,last_difficulty")
        .eq("user_id", user_id)
        .eq("chapter_id", chapter_id)
        .execute()
    )
    return response.data or []


@router.get("/chapters")
def list_question_bank_chapters(
    auth: AuthContext = Depends(require_user),
) -> Dict[str, Any]:
    """Return the published Grade 8 Science chapters available for practice."""
    client = create_supabase_client(auth.jwt)
    curriculum_response = (
        client.table("curriculum_versions")
        .select("id,grade,subject,version_label")
        .eq("grade", 8)
        .eq("subject", "Science")
        .eq("status", "published")
        .execute()
    )
    curriculum_ids = [str(row["id"]) for row in (curriculum_response.data or [])]
    if not curriculum_ids:
        return {"chapters": []}

    chapter_response = (
        client.table("chapters")
        .select("id,sequence_number,title,description,curriculum_version_id")
        .in_("curriculum_version_id", curriculum_ids)
        .eq("status", "published")
        .order("sequence_number")
        .execute()
    )
    return {"chapters": chapter_response.data or []}


@router.post("/questions/{question_id}/flag", response_model=QuestionFlagResponse)
def flag_question_bank_question(
    question_id: UUID,
    auth: AuthContext = Depends(require_user),
) -> QuestionFlagResponse:
    """Record a student's review flag without asking them to classify the issue."""
    client = create_supabase_client(auth.jwt)
    question_response = (
        client.table("question_bank")
        .select("id")
        .eq("id", str(question_id))
        .eq("status", "published")
        .execute()
    )
    if not _one(question_response.data):
        raise HTTPException(status_code=404, detail="Published question not found.")

    client.table("question_review_flags").upsert(
        [{"user_id": _user_id(auth), "question_id": str(question_id)}],
        on_conflict="user_id,question_id",
        ignore_duplicates=True,
    ).execute()
    return QuestionFlagResponse(question_id=question_id, flagged=True)


def _next_review_at(status: str) -> str:
    days = {"needs_review": 1, "on_track": 3, "challenge_next": 7}.get(status, 3)
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


def _update_question_performance(
    client: Any,
    user_id: str,
    test: Dict[str, Any],
    submitted_by_id: Dict[str, QuestionBankAnswer],
    results: List[Dict[str, Any]],
) -> None:
    """Maintain student-by-question counters for future adaptive selection.

    The raw question_attempts rows remain the source of truth. This compact
    cache makes repeated misses and consecutive wrong answers cheap to query
    without changing what the student sees during the current test.
    """
    question_ids = [str(result["question_id"]) for result in results]
    existing_rows = (
        client.table("student_question_performance")
        .select("*")
        .eq("user_id", user_id)
        .in_("question_id", question_ids)
        .execute()
    ).data or []
    existing_by_question_id = {str(row["question_id"]): row for row in existing_rows}
    now = datetime.now(timezone.utc).isoformat()
    rows = []

    for result in results:
        question_id = str(result["question_id"])
        submitted = submitted_by_id[question_id]
        previous = existing_by_question_id.get(question_id, {})
        is_correct = bool(result["is_correct"])
        attempt_count = int(previous.get("attempt_count") or 0) + 1
        correct_count = int(previous.get("correct_count") or 0) + int(is_correct)
        wrong_count = int(previous.get("wrong_count") or 0) + int(not is_correct)
        consecutive_wrong_count = 0 if is_correct else int(previous.get("consecutive_wrong_count") or 0) + 1

        rows.append(
            {
                "user_id": user_id,
                "question_id": question_id,
                "chapter_id": str(test["chapter_id"]),
                "attempt_count": attempt_count,
                "correct_count": correct_count,
                "wrong_count": wrong_count,
                "consecutive_wrong_count": consecutive_wrong_count,
                "last_answer_correct": is_correct,
                "last_selected_option_id": submitted.selected_option_id,
                "last_answer_json": _submitted_json(submitted),
                "last_response_time_ms": submitted.response_time_ms,
                "first_attempted_at": previous.get("first_attempted_at") or now,
                "last_attempted_at": now,
                "last_correct_at": now if is_correct else previous.get("last_correct_at"),
                "last_wrong_at": now if not is_correct else previous.get("last_wrong_at"),
                "next_review_at": _next_review_at("on_track" if is_correct else "needs_review"),
                "updated_at": now,
            }
        )

    client.table("student_question_performance").upsert(
        rows,
        on_conflict="user_id,question_id",
    ).execute()


@router.post("/tests", response_model=QuestionBankTestResponse)
def create_question_bank_test(
    request: QuestionBankTestRequest,
    auth: AuthContext = Depends(require_user),
) -> QuestionBankTestResponse:
    """Create a deterministic/adaptive test from the published question bank."""
    user_id = _user_id(auth)
    client = create_supabase_client(auth.jwt)

    try:
        chapter_response = (
            client.table("chapters")
            .select("id,title,curriculum_version_id")
            .eq("id", str(request.chapter_id))
            .eq("status", "published")
            .execute()
        )
        if not _one(chapter_response.data):
            raise HTTPException(status_code=404, detail="Published chapter not found.")

        questions = _load_published_questions(client, str(request.chapter_id))
        if len(questions) < request.question_count:
            raise HTTPException(
                status_code=409,
                detail=f"This chapter has {len(questions)} published questions; {request.question_count} are required.",
            )

        mastery_rows = _load_mastery(client, user_id, str(request.chapter_id))
        previous_questions = (
            client.table("question_attempts")
            .select("question_id")
            .eq("user_id", user_id)
            .eq("chapter_id", str(request.chapter_id))
            .order("created_at", desc=True)
            .limit(20)
            .execute()
        ).data or []
        seen_ids = [str(row["question_id"]) for row in previous_questions]
        seed = secrets.randbelow(2**62 - 1) + 1
        first_block_size = min(request.block_size, request.question_count)
        selection = select_question_block(
            questions=questions,
            mastery={key: float(value.get("mastery_score") or 0) for key, value in _mastery_map(mastery_rows).items()},
            seen_question_ids=seen_ids,
            seed=seed,
            count=first_block_size,
            mode=request.mode,
            block_number=1,
        )

        test_insert = (
            client.table("test_attempts")
            .insert(
                {
                    "user_id": user_id,
                    "chapter_id": str(request.chapter_id),
                    "mode": request.mode,
                    "question_count": request.question_count,
                    "block_size": request.block_size,
                    "current_block": 1,
                    "seed": seed,
                    "routing_profile": {
                        "difficulty": selection["difficulty"],
                        "focus_concept_ids": selection["focus_concept_ids"],
                        "mode": request.mode,
                    },
                }
            )
            .execute()
        )
        test_row = _one(test_insert.data)
        if not test_row:
            raise RuntimeError("Supabase did not return the created test attempt.")

        test_id = str(test_row["id"])
        client.table("test_questions").insert(
            [
                {
                    "test_attempt_id": test_id,
                    "question_id": question["id"],
                    "block_number": 1,
                    "display_order": index + 1,
                    "selection_reason": selection["selection_reason"],
                }
                for index, question in enumerate(selection["questions"])
            ]
        ).execute()

        routing = {
            "difficulty": selection["difficulty"],
            "focus_concept_ids": selection["focus_concept_ids"],
            "status": "on_track",
        }
        return _question_response(
            test_id=test_id,
            block_number=1,
            block_size=request.block_size,
            question_count=request.question_count,
            questions=selection["questions"],
            routing=routing,
        )
    except HTTPException:
        raise
    except QuestionBankError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("Question-bank configuration error: %s", exc)
        raise HTTPException(status_code=500, detail="Question-bank service is not configured.") from exc
    except Exception as exc:
        logger.exception("Failed to create question-bank test")
        raise HTTPException(status_code=500, detail="Could not create test.") from exc


@router.post("/tests/{test_id}/blocks/{block_number}/submit", response_model=QuestionBankBlockResponse)
def submit_question_bank_block(
    test_id: UUID,
    block_number: int,
    request: QuestionBankBlockSubmission,
    auth: AuthContext = Depends(require_user),
) -> QuestionBankBlockResponse:
    """Score one block, update mastery, and route the next block."""
    user_id = _user_id(auth)
    client = create_supabase_client(auth.jwt)
    test_id_text = str(test_id)

    try:
        test_response = (
            client.table("test_attempts")
            .select("*")
            .eq("id", test_id_text)
            .eq("user_id", user_id)
            .execute()
        )
        test = _one(test_response.data)
        if not test:
            raise HTTPException(status_code=404, detail="Test attempt not found.")
        if test["status"] != "in_progress":
            raise HTTPException(status_code=409, detail="This test is no longer accepting answers.")
        if int(test["current_block"]) != block_number:
            raise HTTPException(status_code=409, detail="This test block is not the current block.")

        test_questions = (
            client.table("test_questions")
            .select("id,question_id,display_order")
            .eq("test_attempt_id", test_id_text)
            .eq("block_number", block_number)
            .order("display_order")
            .execute()
        ).data or []
        expected_question_ids = [str(row["question_id"]) for row in test_questions]
        submitted_by_id = {str(answer.question_id): answer for answer in request.answers}
        if len(submitted_by_id) != len(request.answers) or set(submitted_by_id) != set(expected_question_ids):
            raise HTTPException(status_code=400, detail="Submit exactly one answer for each question in this block.")

        existing_attempts = (
            client.table("question_attempts")
            .select("test_question_id")
            .in_("test_question_id", [str(row["id"]) for row in test_questions])
            .execute()
        ).data or []
        if existing_attempts:
            raise HTTPException(status_code=409, detail="This test block was already submitted.")

        service_client = create_supabase_service_client()
        answer_rows = (
            service_client.table("question_bank")
            .select(
                "id,concept_id,question_text,options_json,correct_option_id,explanation,"
                "question_type,difficulty,cognitive_level,skill_tags,misconception_tags,question_style,"
                "estimated_time_seconds,marks,media_json,option_media_json,render_config,answer_spec"
            )
            .in_("id", expected_question_ids)
            .execute()
        ).data or []
        answer_rows_by_id = {str(row["id"]): row for row in answer_rows}
        if set(answer_rows_by_id) != set(expected_question_ids):
            raise HTTPException(status_code=409, detail="One or more published questions are no longer available.")

        answers = {
            question_id: _submitted_value(submitted_by_id[question_id])
            for question_id in expected_question_ids
        }
        results = score_block(
            [answer_rows_by_id[question_id] for question_id in expected_question_ids],
            answers,
        )
        result_by_id = {str(result["question_id"]): result for result in results}

        client.table("question_attempts").insert(
            [
                {
                    "test_attempt_id": test_id_text,
                    "test_question_id": test_question["id"],
                    "user_id": user_id,
                    "chapter_id": test["chapter_id"],
                    "question_id": test_question["question_id"],
                    "selected_option_id": submitted_by_id[str(test_question["question_id"])].selected_option_id,
                    "answer_json": _submitted_json(submitted_by_id[str(test_question["question_id"])]),
                    "is_correct": result_by_id[str(test_question["question_id"])]["is_correct"],
                    "marks_awarded": result_by_id[str(test_question["question_id"])]["marks_awarded"],
                    "response_time_ms": submitted_by_id[str(test_question["question_id"])].response_time_ms,
                }
                for test_question in test_questions
            ]
        ).execute()

        try:
            _update_question_performance(
                client=client,
                user_id=user_id,
                test=test,
                submitted_by_id=submitted_by_id,
                results=results,
            )
        except Exception:
            # Raw question_attempts is authoritative; a cache failure should
            # never make a completed student's score disappear.
            logger.exception("Failed to update student question performance cache")

        mastery_rows = _load_mastery(client, user_id, str(test["chapter_id"]))
        mastery_updates = update_mastery(results, _mastery_map(mastery_rows))
        for mastery_update in mastery_updates:
            existing = _mastery_map(mastery_rows).get(str(mastery_update["concept_id"]), {})
            client.table("concept_mastery").upsert(
                {
                    "user_id": user_id,
                    "chapter_id": test["chapter_id"],
                    "concept_id": mastery_update["concept_id"],
                    "mastery_score": mastery_update["mastery_score"],
                    "attempt_count": mastery_update["attempt_count"],
                    "correct_count": mastery_update["correct_count"],
                    "last_difficulty": mastery_update["last_difficulty"],
                    "misconception_signals": existing.get("misconception_signals") or {},
                    "last_attempted_at": datetime.now(timezone.utc).isoformat(),
                    "next_review_at": _next_review_at(mastery_update["status"]),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                on_conflict="user_id,chapter_id,concept_id",
            ).execute()

        previous_routing = test.get("routing_profile") or {}
        routing = build_routing_decision(
            results,
            previous_difficulty=previous_routing.get("difficulty"),
        )
        block_score = sum(int(result["marks_awarded"]) for result in results)
        block_total = sum(int(result["maximum_marks"]) for result in results)
        block_percentage = round((block_score / block_total) * 100, 2) if block_total else 0
        current_block = int(test["current_block"])
        answered_count = current_block * int(test["block_size"])
        remaining = max(int(test["question_count"]) - answered_count, 0)

        next_block = None
        if remaining:
            all_selected = (
                client.table("test_questions")
                .select("question_id")
                .eq("test_attempt_id", test_id_text)
                .execute()
            ).data or []
            selected_ids = [str(row["question_id"]) for row in all_selected]
            questions = _load_published_questions(client, str(test["chapter_id"]))
            updated_mastery = _mastery_map(mastery_rows)
            updated_mastery.update({str(update["concept_id"]): update for update in mastery_updates})
            next_selection = select_question_block(
                questions=questions,
                mastery={
                    key: float(value.get("mastery_score") or 0)
                    for key, value in updated_mastery.items()
                },
                seen_question_ids=selected_ids,
                seed=int(test["seed"]),
                count=min(int(test["block_size"]), remaining),
                mode=test["mode"],
                block_number=current_block + 1,
                routing=routing,
            )
            next_block_number = current_block + 1
            display_offset = len(all_selected)
            client.table("test_questions").insert(
                [
                    {
                        "test_attempt_id": test_id_text,
                        "question_id": question["id"],
                        "block_number": next_block_number,
                        "display_order": display_offset + index + 1,
                        "selection_reason": next_selection["selection_reason"],
                    }
                    for index, question in enumerate(next_selection["questions"])
                ]
            ).execute()
            client.table("test_attempts").update(
                {"current_block": next_block_number, "routing_profile": routing}
            ).eq("id", test_id_text).eq("user_id", user_id).execute()
            next_block = _question_response(
                test_id=test_id_text,
                block_number=next_block_number,
                block_size=int(test["block_size"]),
                question_count=int(test["question_count"]),
                questions=next_selection["questions"],
                routing=routing,
            )
        else:
            all_attempts = (
                client.table("question_attempts")
                .select("marks_awarded")
                .eq("test_attempt_id", test_id_text)
                .execute()
            ).data or []
            all_test_questions = (
                client.table("test_questions")
                .select("question_id")
                .eq("test_attempt_id", test_id_text)
                .execute()
            ).data or []
            all_question_ids = [str(row["question_id"]) for row in all_test_questions]
            all_question_rows = (
                service_client.table("question_bank")
                .select("id,marks")
                .in_("id", all_question_ids)
                .execute()
            ).data or []
            maximum_marks = sum(int(row.get("marks") or 1) for row in all_question_rows)
            final_score = sum(int(attempt["marks_awarded"]) for attempt in all_attempts)
            client.table("test_attempts").update(
                {
                    "status": "completed",
                    "score": final_score,
                    "maximum_marks": maximum_marks,
                    "routing_profile": routing,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
            ).eq("id", test_id_text).eq("user_id", user_id).execute()

        return QuestionBankBlockResponse(
            test_id=test_id,
            block_number=block_number,
            block_score=block_score,
            block_total=block_total,
            percentage=block_percentage,
            results=[
                {
                    "question_id": result["question_id"],
                    "is_correct": result["is_correct"],
                    "marks_awarded": result["marks_awarded"],
                    "maximum_marks": result["maximum_marks"],
                    "explanation": result.get("explanation"),
                    "selected_option_id": result.get("selected_option_id"),
                    "correct_option_id": result.get("correct_option_id"),
                    "selected_option_ids": result.get("selected_option_ids") or [],
                    "correct_option_ids": result.get("correct_option_ids") or [],
                    "answer_summary": result.get("answer_summary"),
                    "correct_answer_summary": result.get("correct_answer_summary"),
                }
                for result in results
            ],
            mastery_updates=mastery_updates,
            next_block=next_block,
            completed=next_block is None,
        )
    except HTTPException:
        raise
    except QuestionBankError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("Question-bank configuration error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Question-bank scoring is not configured with a service-role key.",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to submit question-bank block")
        raise HTTPException(status_code=500, detail="Could not submit test block.") from exc


@router.get("/mastery")
def get_question_bank_mastery(
    chapter_id: UUID,
    auth: AuthContext = Depends(require_user),
) -> Dict[str, Any]:
    client = create_supabase_client(auth.jwt)
    rows = _load_mastery(client, _user_id(auth), str(chapter_id))
    return {"chapter_id": chapter_id, "concepts": rows}
