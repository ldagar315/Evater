from __future__ import annotations

import logging
from datetime import datetime, timezone
from math import floor
from typing import Any, Dict, Iterable, List, Literal, Mapping, Sequence

from fastapi import APIRouter, Depends, HTTPException, Query

from ..auth import AuthContext, require_user
from ..leaderboard import (
    LEAGUE_TIERS,
    apply_league_multiplier,
    apply_season_inactivity_penalty,
    calculate_practice_score,
    league_label,
    league_sort_key,
    promotion_threshold,
    question_score,
    resolve_season_movements,
    scoring_sort_key,
    season_window,
)
from ..models import LeaderboardEntry, LeaderboardResponse
from ..supabase_client import create_supabase_service_client

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(require_user)])

# Keep accepting the previous UI's query values while the hosting provider's
# frontend cache rolls forward. All supported values intentionally resolve to
# the one current product mode: the global seasonal league.
LeaderboardScope = Literal["global", "classroom", "school"]
LeaderboardPeriod = Literal["season", "weekly", "all_time"]


def _one(rows: Any) -> Dict[str, Any] | None:
    if isinstance(rows, list):
        return rows[0] if rows else None
    return rows if isinstance(rows, dict) else None


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _display_name(row: Mapping[str, Any]) -> str:
    raw_name = str(row.get("name") or row.get("user_name") or "Student").strip()
    parts = raw_name.split()
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]} {parts[-1][0]}."


def _unavailable_response(message: str) -> LeaderboardResponse:
    return LeaderboardResponse(
        scope="global",
        scope_label="Global student league",
        period="season",
        period_label="Current season",
        scope_available=False,
        membership_message=message,
    )


def _finalize_expired_seasons(client: Any, now: datetime) -> None:
    expired = (
        client.table("leaderboard_seasons")
        .select("*")
        .eq("status", "active")
        .lt("ends_at", _iso(now))
        .execute()
        .data
        or []
    )
    for season in expired:
        players = (
            client.table("leaderboard_players")
            .select("season_id,user_id,league_tier,points,correct_answers,completed_practices,inactive_seasons,inactivity_baseline_points,inactivity_floor_reached")
            .eq("season_id", season["id"])
            .execute()
            .data
            or []
        )
        movements = resolve_season_movements(players)
        for player in players:
            movement = movements.get(str(player["user_id"]), {
                "final_tier": player.get("league_tier") or "bronze_3",
                "movement": "held",
            })
            inactivity_state = _finalized_inactivity_state(player)
            (
                client.table("leaderboard_players")
                .update({
                    **inactivity_state,
                    "final_tier": movement["final_tier"],
                    "movement": movement["movement"],
                    "updated_at": _iso(now),
                })
                .eq("season_id", season["id"])
                .eq("user_id", player["user_id"])
                .execute()
            )
        (
            client.table("leaderboard_seasons")
            .update({"status": "completed", "completed_at": _iso(now)})
            .eq("id", season["id"])
            .eq("status", "active")
            .execute()
        )


def _ensure_current_season(client: Any, now: datetime) -> Dict[str, Any]:
    _finalize_expired_seasons(client, now)
    window = season_window(now)
    current = _one(
        client.table("leaderboard_seasons")
        .select("*")
        .eq("season_key", window["season_key"])
        .limit(1)
        .execute()
        .data
    )
    if current:
        return current

    client.table("leaderboard_seasons").upsert(
        {
            "season_key": window["season_key"],
            "season_number": window["season_number"],
            "starts_at": _iso(window["starts_at"]),
            "ends_at": _iso(window["ends_at"]),
            "status": "active",
        },
        on_conflict="season_key",
    ).execute()
    current = _one(
        client.table("leaderboard_seasons")
        .select("*")
        .eq("season_key", window["season_key"])
        .limit(1)
        .execute()
        .data
    )
    if not current:
        raise RuntimeError("Supabase did not return the active leaderboard season.")
    return current


def _previous_player(client: Any, user_id: str) -> Dict[str, Any] | None:
    return _one(
        client.table("leaderboard_players")
        .select("league_tier,final_tier,points,completed_practices,inactive_seasons,inactivity_baseline_points,inactivity_floor_reached")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
        .data
    )


def _finalized_inactivity_state(player: Mapping[str, Any]) -> Dict[str, Any]:
    if int(player.get("completed_practices") or 0) > 0:
        return {
            "inactive_seasons": 0,
            "inactivity_baseline_points": None,
            "inactivity_floor_reached": False,
        }

    inactive_seasons = int(player.get("inactive_seasons") or 0) + 1
    baseline_points = player.get("inactivity_baseline_points")
    if baseline_points is None:
        baseline_points = int(player.get("points") or 0)
    baseline_points = max(0, int(baseline_points))
    floor_points = floor(baseline_points * 50 / 100)
    return {
        "inactive_seasons": inactive_seasons,
        "inactivity_baseline_points": baseline_points,
        "inactivity_floor_reached": inactive_seasons >= 2 and int(player.get("points") or 0) <= floor_points,
    }


def _next_player_state(previous: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not previous:
        return {
            "league_tier": "bronze_3",
            "points": 0,
            "inactive_seasons": 0,
            "inactivity_baseline_points": None,
            "inactivity_floor_reached": False,
        }

    previous_points = int(previous.get("points") or 0)
    completed_practices = int(previous.get("completed_practices") or 0)
    if completed_practices > 0:
        return {
            "league_tier": str(previous.get("final_tier") or previous.get("league_tier") or "bronze_3"),
            "points": previous_points,
            "inactive_seasons": 0,
            "inactivity_baseline_points": None,
            "inactivity_floor_reached": False,
        }

    inactive_seasons = int(previous.get("inactive_seasons") or 0)
    baseline_points = previous.get("inactivity_baseline_points")
    if baseline_points is None:
        baseline_points = previous_points
    points, floor_reached = apply_season_inactivity_penalty(
        previous_points,
        inactive_seasons,
        int(baseline_points),
    )
    return {
        "league_tier": str(previous.get("final_tier") or previous.get("league_tier") or "bronze_3"),
        "points": points,
        "inactive_seasons": inactive_seasons,
        "inactivity_baseline_points": int(baseline_points),
        "inactivity_floor_reached": floor_reached,
    }


def _ensure_players(client: Any, season_id: str, user_ids: Sequence[str], now: datetime) -> None:
    if not user_ids:
        return
    existing = (
        client.table("leaderboard_players")
        .select("user_id")
        .eq("season_id", season_id)
        .in_("user_id", list(user_ids))
        .execute()
        .data
        or []
    )
    existing_ids = {str(row["user_id"]) for row in existing}
    missing = [user_id for user_id in user_ids if user_id not in existing_ids]
    if not missing:
        return
    rows = []
    for user_id in missing:
        previous = _previous_player(client, user_id)
        state = _next_player_state(previous)
        rows.append({
            "season_id": season_id,
            "user_id": user_id,
            **state,
            "last_decay_at": None,
            "updated_at": _iso(now),
        })
    client.table("leaderboard_players").insert(rows).execute()


def _is_empty_answer(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _attempt_is_answered(attempt: Mapping[str, Any]) -> bool:
    if attempt.get("selected_option_id"):
        return True
    answer_json = attempt.get("answer_json")
    if isinstance(answer_json, dict):
        if "value" in answer_json:
            return not _is_empty_answer(answer_json.get("value"))
        return bool(answer_json)
    return not _is_empty_answer(answer_json)


def award_completed_practice(
    client: Any,
    *,
    test_id: str,
    user_id: str,
    completed_at: datetime,
) -> Dict[str, int] | None:
    """Award one completed practice exactly once for the current season."""
    enrollment = _one(
        client.table("student_enrollments")
        .select("user_id")
        .eq("user_id", user_id)
        .eq("role", "student")
        .limit(1)
        .execute()
        .data
    )
    if not enrollment:
        return None

    existing_practice = _one(
        client.table("leaderboard_practice_scores")
        .select("points_awarded,raw_score")
        .eq("test_attempt_id", test_id)
        .limit(1)
        .execute()
        .data
    )
    if existing_practice:
        return {
            "raw_score": int(existing_practice.get("raw_score") or 0),
            "points_awarded": int(existing_practice.get("points_awarded") or 0),
        }

    now = completed_at if completed_at.tzinfo else completed_at.replace(tzinfo=timezone.utc)
    season = _ensure_current_season(client, now)
    season_id = str(season["id"])
    _ensure_players(client, season_id, [user_id], now)
    player = _one(
        client.table("leaderboard_players")
        .select("*")
        .eq("season_id", season_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
        .data
    )
    if not player:
        raise RuntimeError("Supabase did not return the leaderboard player.")

    attempts = (
        client.table("question_attempts")
        .select("question_id,is_correct,selected_option_id,answer_json")
        .eq("test_attempt_id", test_id)
        .execute()
        .data
        or []
    )
    question_ids = [str(attempt["question_id"]) for attempt in attempts]
    question_rows = (
        client.table("question_bank")
        .select("id,difficulty")
        .in_("id", question_ids)
        .execute()
        .data
        or []
    ) if question_ids else []
    difficulty_by_id = {str(row["id"]): row.get("difficulty") or "easy" for row in question_rows}

    already_scored = (
        client.table("leaderboard_question_scores")
        .select("question_id")
        .eq("season_id", season_id)
        .eq("user_id", user_id)
        .in_("question_id", question_ids)
        .execute()
        .data
        or []
    ) if question_ids else []
    already_scored_ids = {str(row["question_id"]) for row in already_scored}

    score_inputs = []
    question_score_rows = []
    for attempt in attempts:
        question_id = str(attempt["question_id"])
        if question_id in already_scored_ids:
            continue
        score_inputs.append({
            "difficulty": difficulty_by_id.get(question_id, "easy"),
            "is_correct": bool(attempt.get("is_correct")),
            "is_answered": _attempt_is_answered(attempt),
        })

    practice_score = calculate_practice_score(score_inputs)
    raw_score = int(practice_score["raw_score"])
    points_awarded = apply_league_multiplier(raw_score, str(player.get("league_tier") or "bronze_3"))

    for attempt, score_input in zip(
        [attempt for attempt in attempts if str(attempt["question_id"]) not in already_scored_ids],
        score_inputs,
    ):
        question_score_rows.append({
            "season_id": season_id,
            "user_id": user_id,
            "question_id": str(attempt["question_id"]),
            "test_attempt_id": test_id,
            "points_awarded": question_score(
                difficulty=score_input["difficulty"],
                is_correct=score_input["is_correct"],
                is_answered=score_input["is_answered"],
            ),
        })

    if question_score_rows:
        client.table("leaderboard_question_scores").insert(question_score_rows).execute()

    updated_points = int(player.get("points") or 0) + points_awarded
    client.table("leaderboard_practice_scores").insert({
        "test_attempt_id": test_id,
        "season_id": season_id,
        "user_id": user_id,
        "raw_score": raw_score,
        "points_awarded": points_awarded,
        "scored_question_count": len(score_inputs),
        "correct_answers": int(practice_score["correct_answers"]),
        "completed_at": _iso(now),
    }).execute()
    client.table("leaderboard_players").update({
        "points": updated_points,
        "completed_practices": int(player.get("completed_practices") or 0) + 1,
        "correct_answers": int(player.get("correct_answers") or 0) + int(practice_score["correct_answers"]),
        "last_activity_at": _iso(now),
        "last_decay_at": None,
        "inactive_seasons": 0,
        "inactivity_baseline_points": None,
        "inactivity_floor_reached": False,
        "updated_at": _iso(now),
    }).eq("season_id", season_id).eq("user_id", user_id).execute()
    return {"raw_score": raw_score, "points_awarded": points_awarded}


def _build_entries(
    players: Iterable[Dict[str, Any]],
    current_user_id: str,
) -> List[LeaderboardEntry]:
    ordered = sorted(players, key=scoring_sort_key)
    league_ranks: Dict[str, int] = {}
    for tier in LEAGUE_TIERS:
        tier_players = sorted(
            [player for player in ordered if player.get("league_tier") == tier],
            key=league_sort_key,
        )
        league_ranks.update({str(player["user_id"]): index for index, player in enumerate(tier_players, start=1)})

    return [
        LeaderboardEntry(
            rank=index,
            display_name=str(player.get("display_name") or "Student"),
            score=int(player.get("points") or 0),
            correct_answers=int(player.get("correct_answers") or 0),
            completed_tests=int(player.get("completed_practices") or 0),
            league_tier=str(player.get("league_tier") or "bronze_3"),
            league_label=league_label(str(player.get("league_tier") or "bronze_3")),
            league_rank=league_ranks.get(str(player["user_id"]), 1),
            is_current_user=str(player["user_id"]) == current_user_id,
        )
        for index, player in enumerate(ordered, start=1)
    ]


@router.get("/leaderboard", response_model=LeaderboardResponse)
def get_leaderboard(
    scope: LeaderboardScope = Query(default="global"),
    period: LeaderboardPeriod = Query(default="season"),
    auth: AuthContext = Depends(require_user),
) -> LeaderboardResponse:
    """Return the current global seasonal league for verified students."""
    del scope, period
    current_user_id = str(auth.user.id)
    now = datetime.now(timezone.utc)

    try:
        client = create_supabase_service_client()
        verified_members = (
            client.table("student_enrollments")
            .select("user_id")
            .eq("role", "student")
            .execute()
            .data
            or []
        )
        member_ids = [str(member["user_id"]) for member in verified_members]
        if current_user_id not in member_ids:
            return _unavailable_response("Only verified student accounts appear in the global league.")

        season = _ensure_current_season(client, now)
        season_id = str(season["id"])
        _ensure_players(client, season_id, member_ids, now)
        players = (
            client.table("leaderboard_players")
            .select("*")
            .eq("season_id", season_id)
            .in_("user_id", member_ids)
            .execute()
            .data
            or []
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
        players_for_display = [
            {
                **player,
                "display_name": _display_name(profile_by_user_id.get(str(player["user_id"]), {})),
            }
            for player in players
        ]
        entries = _build_entries(players_for_display, current_user_id)
        current_entry = next((entry for entry in entries if entry.is_current_user), None)
        current_player = next(
            (player for player in players_for_display if str(player["user_id"]) == current_user_id),
            None,
        )
        current_tier = str((current_player or {}).get("league_tier") or "bronze_3")
        threshold = promotion_threshold(current_tier)
        topper = entries[0] if entries else None
        season_number = int(season["season_number"])
        return LeaderboardResponse(
            scope="global",
            scope_label="Global student league",
            period="season",
            period_label=f"Season {season_number}",
            scope_available=True,
            entries=entries,
            current_user_rank=current_entry.rank if current_entry else None,
            current_user_league=current_tier,
            current_user_league_label=league_label(current_tier),
            current_user_league_rank=current_entry.league_rank if current_entry else None,
            promotion_threshold=threshold,
            promotion_points_remaining=max(threshold - int((current_player or {}).get("points") or 0), 0) if threshold is not None else None,
            season_number=season_number,
            season_starts_at=str(season["starts_at"]),
            season_ends_at=str(season["ends_at"]),
            topper_name=topper.display_name if topper else None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to load global seasonal leaderboard")
        raise HTTPException(status_code=500, detail="Could not load leaderboard.") from exc
