from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from ..auth import AuthContext, require_user
from ..models import LeaderboardEntry, LeaderboardResponse
from ..supabase_client import create_supabase_service_client

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(require_user)])

LeaderboardScope = Literal["classroom", "school"]
LeaderboardPeriod = Literal["weekly", "all_time"]


def _one(rows: Any) -> Dict[str, Any] | None:
    if isinstance(rows, list):
        return rows[0] if rows else None
    return rows if isinstance(rows, dict) else None


def _display_name(row: Dict[str, Any]) -> str:
    raw_name = str(row.get("name") or row.get("user_name") or "Student").strip()
    parts = raw_name.split()
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]} {parts[-1][0]}."


def build_leaderboard_entries(
    members: Iterable[Dict[str, Any]],
    completed_tests: Iterable[Dict[str, Any]],
    question_attempts: Iterable[Dict[str, Any]],
    current_user_id: str,
) -> List[LeaderboardEntry]:
    """Build the explainable MVP score from completed practice activity.

    A completed practice test is worth 25 points and each correct answer is
    worth 10 points. The server owns the source rows and tie-breaks are stable.
    """
    scores: Dict[str, Dict[str, Any]] = {}
    for member in members:
        user_id = str(member["user_id"])
        scores[user_id] = {
            "display_name": _display_name(member),
            "score": 0,
            "correct_answers": 0,
            "completed_tests": 0,
        }

    test_users: Dict[str, str] = {}
    for test in completed_tests:
        user_id = str(test.get("user_id"))
        if user_id not in scores:
            continue
        test_id = str(test.get("id"))
        test_users[test_id] = user_id
        scores[user_id]["completed_tests"] += 1
        scores[user_id]["score"] += 25

    for attempt in question_attempts:
        user_id = str(attempt.get("user_id") or test_users.get(str(attempt.get("test_attempt_id"))))
        if user_id not in scores or not attempt.get("is_correct"):
            continue
        scores[user_id]["correct_answers"] += 1
        scores[user_id]["score"] += 10

    ordered = sorted(
        ((user_id, values) for user_id, values in scores.items()),
        key=lambda item: (
            -item[1]["score"],
            -item[1]["correct_answers"],
            -item[1]["completed_tests"],
            item[1]["display_name"].lower(),
            item[0],
        ),
    )

    return [
        LeaderboardEntry(
            rank=index,
            display_name=values["display_name"],
            score=values["score"],
            correct_answers=values["correct_answers"],
            completed_tests=values["completed_tests"],
            is_current_user=user_id == current_user_id,
        )
        for index, (user_id, values) in enumerate(ordered, start=1)
    ]


def _unavailable_response(
    scope: LeaderboardScope,
    period: LeaderboardPeriod,
    message: str,
) -> LeaderboardResponse:
    return LeaderboardResponse(
        scope=scope,
        scope_label="Your learning community",
        period=period,
        period_label="Last 7 days" if period == "weekly" else "All time",
        scope_available=False,
        membership_message=message,
    )


@router.get("/leaderboard", response_model=LeaderboardResponse)
def get_leaderboard(
    scope: LeaderboardScope = Query(default="classroom"),
    period: LeaderboardPeriod = Query(default="weekly"),
    auth: AuthContext = Depends(require_user),
) -> LeaderboardResponse:
    """Return a privacy-scoped ranking for the authenticated learner.

    The endpoint uses the service client only after authenticating the caller
    and resolving their explicit enrollment. Client-provided school or grade
    strings never determine the result set.
    """
    current_user_id = str(auth.user.id)

    try:
        client = create_supabase_service_client()
        enrollment = _one(
            client.table("student_enrollments")
            .select("user_id,school_id,classroom_id")
            .eq("user_id", current_user_id)
            .limit(1)
            .execute()
            .data
        )
        if not enrollment:
            return _unavailable_response(
                scope,
                period,
                "Your account is not assigned to a classroom yet.",
            )

        school = _one(
            client.table("schools")
            .select("id,name")
            .eq("id", enrollment["school_id"])
            .limit(1)
            .execute()
            .data
        )
        classroom = _one(
            client.table("classrooms")
            .select("id,name,school_id")
            .eq("id", enrollment["classroom_id"])
            .limit(1)
            .execute()
            .data
        )
        if not school or not classroom or classroom["school_id"] != school["id"]:
            return _unavailable_response(
                scope,
                period,
                "Your classroom membership needs to be updated.",
            )

        member_query = client.table("student_enrollments").select("user_id")
        if scope == "classroom":
            member_query = member_query.eq("classroom_id", enrollment["classroom_id"])
            scope_label = f'{school["name"]} · {classroom["name"]}'
        else:
            member_query = member_query.eq("school_id", enrollment["school_id"])
            scope_label = str(school["name"])

        members = member_query.execute().data or []
        member_ids = [str(member["user_id"]) for member in members]
        if not member_ids:
            return LeaderboardResponse(
                scope=scope,
                scope_label=scope_label,
                period=period,
                period_label="Last 7 days" if period == "weekly" else "All time",
                scope_available=True,
                membership_message="No learners have joined this group yet.",
            )

        profiles = (
            client.table("Users")
            .select("created_by,name,user_name")
            .in_("created_by", member_ids)
            .execute()
            .data
            or []
        )
        profile_by_user_id = {str(profile["created_by"]): profile for profile in profiles}
        members_for_scoring = [
            {
                "user_id": user_id,
                **profile_by_user_id.get(user_id, {"name": "Student"}),
            }
            for user_id in member_ids
        ]

        tests_query = (
            client.table("test_attempts")
            .select("id,user_id,status,mode,completed_at")
            .in_("user_id", member_ids)
            .eq("status", "completed")
            .eq("mode", "practice")
        )
        if period == "weekly":
            start_at = datetime.now(timezone.utc) - timedelta(days=7)
            tests_query = tests_query.gte("completed_at", start_at.isoformat())
        completed_tests = tests_query.execute().data or []
        test_ids = [str(test["id"]) for test in completed_tests]

        question_attempts = []
        if test_ids:
            question_attempts = (
                client.table("question_attempts")
                .select("user_id,test_attempt_id,is_correct,marks_awarded")
                .in_("test_attempt_id", test_ids)
                .execute()
                .data
                or []
            )

        entries = build_leaderboard_entries(
            members_for_scoring,
            completed_tests,
            question_attempts,
            current_user_id,
        )
        current_user_rank = next(
            (entry.rank for entry in entries if entry.is_current_user),
            None,
        )
        return LeaderboardResponse(
            scope=scope,
            scope_label=scope_label,
            period=period,
            period_label="Last 7 days" if period == "weekly" else "All time",
            scope_available=True,
            entries=entries,
            current_user_rank=current_user_rank,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to load scoped leaderboard")
        raise HTTPException(status_code=500, detail="Could not load leaderboard.") from exc
