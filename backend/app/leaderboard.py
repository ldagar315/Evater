"""Pure rules for Evater's simple seasonal global league."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import floor
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


QUESTION_VALUES = {"easy": 4, "medium": 8, "hard": 12}
LEAGUE_TIERS = (
    "bronze_3",
    "bronze_2",
    "bronze_1",
    "silver_3",
    "silver_2",
    "silver_1",
    "gold_3",
    "gold_2",
    "gold_1",
    "platinum_3",
    "platinum_2",
    "platinum_1",
    "diamond_3",
    "diamond_2",
    "diamond_1",
)
LEAGUE_LABELS = {
    "bronze_3": "Bronze III",
    "bronze_2": "Bronze II",
    "bronze_1": "Bronze I",
    "silver_3": "Silver III",
    "silver_2": "Silver II",
    "silver_1": "Silver I",
    "gold_3": "Gold III",
    "gold_2": "Gold II",
    "gold_1": "Gold I",
    "platinum_3": "Platinum III",
    "platinum_2": "Platinum II",
    "platinum_1": "Platinum I",
    "diamond_3": "Diamond III",
    "diamond_2": "Diamond II",
    "diamond_1": "Diamond I",
}

# Thresholds are deliberately configuration, not magic inside the request
# handler. They can be tuned after real traffic without changing the scoring
# algorithm or the meaning of historical points.
PROMOTION_THRESHOLDS = {
    "bronze_3": 100,
    "bronze_2": 150,
    "bronze_1": 200,
    "silver_3": 275,
    "silver_2": 350,
    "silver_1": 450,
    "gold_3": 575,
    "gold_2": 725,
    "gold_1": 900,
    "platinum_3": 1100,
    "platinum_2": 1350,
    "platinum_1": 1650,
    "diamond_3": 2000,
    "diamond_2": 2400,
    "diamond_1": None,
}

# A higher league earns fewer effective points. Flooring keeps every awarded
# positive score an integer and the minimum positive award equal to one.
LEAGUE_MULTIPLIERS = {
    "bronze": 100,
    "silver": 90,
    "gold": 80,
    "platinum": 70,
    "diamond": 60,
}

SEASON_LENGTH = timedelta(days=14)
INACTIVITY_REMAINING_PERCENT = 90
INACTIVITY_FLOOR_PERCENT = 50
# Monday 2026-01-05 gives stable UTC two-week windows without needing a
# scheduler. The next request after a boundary creates/finalizes state.
SEASON_EPOCH = datetime(2026, 1, 5, tzinfo=timezone.utc)


def league_label(tier: str) -> str:
    return LEAGUE_LABELS.get(tier, "Bronze III")


def league_index(tier: str) -> int:
    try:
        return LEAGUE_TIERS.index(tier)
    except ValueError:
        return 0


def league_multiplier(tier: str) -> int:
    return LEAGUE_MULTIPLIERS.get(tier.split("_", 1)[0], 100)


def promotion_threshold(tier: str) -> int | None:
    return PROMOTION_THRESHOLDS.get(tier)


def question_score(*, difficulty: str, is_correct: bool, is_answered: bool) -> int:
    """Return integer raw points for one completed-practice question.

    Easy/medium/hard are worth 4/8/12. Wrong answers lose 25% and
    unanswered questions lose 50%. The values are scaled by four so no
    question-level decimal is ever created.
    """
    value = QUESTION_VALUES.get(str(difficulty).lower(), QUESTION_VALUES["easy"])
    if is_correct:
        return value
    if not is_answered:
        return -(value // 2)
    return -(value // 4)


def calculate_practice_score(questions: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    raw_score = 0
    correct_answers = 0
    for question in questions:
        is_correct = bool(question.get("is_correct"))
        raw_score += question_score(
            difficulty=str(question.get("difficulty") or "easy"),
            is_correct=is_correct,
            is_answered=bool(question.get("is_answered")),
        )
        correct_answers += int(is_correct)

    return {
        "raw_score": max(0, raw_score),
        "correct_answers": correct_answers,
    }


def apply_league_multiplier(raw_score: int, tier: str) -> int:
    if raw_score <= 0:
        return 0
    scaled = floor(raw_score * league_multiplier(tier) / 100)
    return max(1, scaled)


def season_window(now: datetime | None = None) -> Dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    elapsed_seconds = (current - SEASON_EPOCH).total_seconds()
    season_offset = floor(elapsed_seconds / SEASON_LENGTH.total_seconds())
    starts_at = SEASON_EPOCH + season_offset * SEASON_LENGTH
    ends_at = starts_at + SEASON_LENGTH
    season_number = season_offset + 1
    return {
        "season_key": f"season-{season_number:04d}",
        "season_number": season_number,
        "starts_at": starts_at,
        "ends_at": ends_at,
    }


def apply_season_inactivity_penalty(
    points: int,
    inactive_seasons: int,
    inactivity_baseline_points: int | None,
) -> Tuple[int, bool]:
    """Apply one seasonal trophy penalty after the first inactive season.

    ``inactive_seasons`` is the number of consecutive inactive seasons already
    completed before the current season. A value of zero therefore represents
    the first inactive season and keeps the immunity intact. Starting with the
    second consecutive inactive season, trophies lose 10% once per season, but
    never fall below 50% of the balance held when the inactive streak began.
    """
    current_points = max(0, int(points))
    if inactive_seasons <= 0:
        return current_points, False

    baseline = max(
        0,
        int(inactivity_baseline_points if inactivity_baseline_points is not None else current_points),
    )
    floor_points = floor(baseline * INACTIVITY_FLOOR_PERCENT / 100)
    if current_points <= floor_points:
        return floor_points, True

    remaining = max(floor(current_points * INACTIVITY_REMAINING_PERCENT / 100), floor_points)
    return remaining, remaining <= floor_points


def scoring_sort_key(player: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        -league_index(str(player.get("league_tier") or "bronze_3")),
        -int(player.get("points") or 0),
        -int(player.get("correct_answers") or 0),
        -int(player.get("completed_practices") or 0),
        str(player.get("display_name") or "Student").lower(),
        str(player.get("user_id") or ""),
    )


def league_sort_key(player: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        -int(player.get("points") or 0),
        -int(player.get("correct_answers") or 0),
        -int(player.get("completed_practices") or 0),
        str(player.get("display_name") or "Student").lower(),
        str(player.get("user_id") or ""),
    )


def _can_promote(player: Mapping[str, Any]) -> bool:
    return int(player.get("completed_practices") or 0) > 0


def _protected_from_demotion(player: Mapping[str, Any]) -> bool:
    inactive_seasons = int(player.get("inactive_seasons") or 0)
    current_is_inactive = int(player.get("completed_practices") or 0) == 0
    if not current_is_inactive:
        return False
    effective_inactive_seasons = inactive_seasons + (1 if current_is_inactive else 0)
    if effective_inactive_seasons == 1:
        return True
    if effective_inactive_seasons < 2:
        return False
    if bool(player.get("inactivity_floor_reached")):
        return True

    baseline = player.get("inactivity_baseline_points")
    if baseline is None:
        return False
    floor_points = floor(max(0, int(baseline)) * INACTIVITY_FLOOR_PERCENT / 100)
    return int(player.get("points") or 0) <= floor_points


def resolve_season_movements(players: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, str]]:
    """Return one end-of-season movement decision per player."""
    by_tier: Dict[str, List[Mapping[str, Any]]] = {tier: [] for tier in LEAGUE_TIERS}
    for player in players:
        tier = str(player.get("league_tier") or "bronze_3")
        by_tier.setdefault(tier, []).append(player)

    movements: Dict[str, Dict[str, str]] = {}
    for tier in LEAGUE_TIERS:
        group = sorted(by_tier.get(tier, []), key=league_sort_key)
        if not group:
            continue

        for player in group:
            user_id = str(player["user_id"])
            movements[user_id] = {"final_tier": tier, "movement": "held"}

        threshold = promotion_threshold(tier)
        promotion_candidates = [player for player in group if _can_promote(player)]
        if threshold is not None and tier != LEAGUE_TIERS[-1] and promotion_candidates:
            winner = sorted(promotion_candidates, key=league_sort_key)[0]
            if int(winner.get("points") or 0) >= threshold:
                movements[str(winner["user_id"])] = {
                    "final_tier": LEAGUE_TIERS[league_index(tier) + 1],
                    "movement": "promoted",
                }

        if len(group) >= 3 and tier != LEAGUE_TIERS[0]:
            demotion_candidates = [player for player in group if not _protected_from_demotion(player)]
            if demotion_candidates:
                demoted = sorted(demotion_candidates, key=league_sort_key)[-1]
                movements[str(demoted["user_id"])] = {
                    "final_tier": LEAGUE_TIERS[league_index(tier) - 1],
                    "movement": "demoted",
                }

    return movements
