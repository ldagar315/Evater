from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.leaderboard import (
    apply_league_multiplier,
    apply_season_inactivity_penalty,
    calculate_practice_score,
    question_score,
    resolve_season_movements,
    season_window,
)
from app.routers.leaderboard_router import _finalized_inactivity_state, _next_player_state
from app.main import app


def test_leaderboard_requires_authentication():
    response = TestClient(app).get('/api/v1/leaderboard')
    assert response.status_code == 401


def test_integer_question_values_use_25_and_50_percent_penalties():
    assert question_score(difficulty='easy', is_correct=True, is_answered=True) == 4
    assert question_score(difficulty='easy', is_correct=False, is_answered=True) == -1
    assert question_score(difficulty='easy', is_correct=False, is_answered=False) == -2
    assert question_score(difficulty='medium', is_correct=True, is_answered=True) == 8
    assert question_score(difficulty='hard', is_correct=False, is_answered=True) == -3


def test_practice_score_has_zero_floor_and_counts_correct_answers():
    result = calculate_practice_score([
        {'difficulty': 'easy', 'is_correct': True, 'is_answered': True},
        {'difficulty': 'medium', 'is_correct': False, 'is_answered': True},
        {'difficulty': 'hard', 'is_correct': False, 'is_answered': False},
    ])

    assert result == {'raw_score': 0, 'correct_answers': 1}


def test_league_multiplier_floors_and_keeps_positive_scores_at_one():
    assert apply_league_multiplier(4, 'silver_3') == 3
    assert apply_league_multiplier(4, 'gold_3') == 3
    assert apply_league_multiplier(1, 'diamond_1') == 1
    assert apply_league_multiplier(0, 'diamond_1') == 0


def test_season_window_is_two_weeks():
    start = datetime(2026, 7, 20, tzinfo=timezone.utc)
    window = season_window(start)
    assert window['starts_at'] == start
    assert window['ends_at'] == datetime(2026, 8, 3, tzinfo=timezone.utc)


def test_first_inactive_season_has_immunity():
    assert apply_season_inactivity_penalty(101, 0, 101) == (101, False)


def test_inactivity_penalty_starts_in_second_consecutive_season():
    assert apply_season_inactivity_penalty(101, 1, 101) == (90, False)
    assert apply_season_inactivity_penalty(90, 2, 101) == (81, False)


def test_inactivity_penalty_stops_at_half_of_starting_balance():
    points, floor_reached = apply_season_inactivity_penalty(51, 12, 101)

    assert points == 50
    assert floor_reached is True


def test_trophies_carry_into_the_next_season_for_active_students():
    state = _next_player_state({
        'final_tier': 'silver_3',
        'points': 240,
        'completed_practices': 3,
        'inactive_seasons': 0,
    })

    assert state == {
        'league_tier': 'silver_3',
        'points': 240,
        'inactive_seasons': 0,
        'inactivity_baseline_points': None,
        'inactivity_floor_reached': False,
    }


def test_inactive_seasons_carry_trophies_then_apply_the_penalty():
    first_inactive = _next_player_state({
        'final_tier': 'silver_3',
        'points': 100,
        'completed_practices': 3,
        'inactive_seasons': 0,
    })
    second_inactive = _next_player_state({
        'final_tier': 'silver_3',
        'points': first_inactive['points'],
        'completed_practices': 0,
        'inactive_seasons': 1,
        'inactivity_baseline_points': 100,
    })

    assert first_inactive['points'] == 100
    assert first_inactive['inactive_seasons'] == 0
    assert second_inactive['points'] == 90
    assert second_inactive['inactive_seasons'] == 1
    assert second_inactive['inactivity_baseline_points'] == 100


def test_end_of_season_records_the_current_inactive_streak():
    state = _finalized_inactivity_state({
        'points': 100,
        'completed_practices': 0,
        'inactive_seasons': 0,
    })

    assert state == {
        'inactive_seasons': 1,
        'inactivity_baseline_points': 100,
        'inactivity_floor_reached': False,
    }


def test_season_promotes_one_eligible_student_and_demotes_bottom_with_three_players():
    movements = resolve_season_movements([
        {'user_id': 'bronze-a', 'league_tier': 'bronze_3', 'points': 110, 'correct_answers': 4, 'completed_practices': 2},
        {'user_id': 'bronze-b', 'league_tier': 'bronze_3', 'points': 80, 'correct_answers': 3, 'completed_practices': 2},
        {'user_id': 'bronze-c', 'league_tier': 'bronze_3', 'points': 20, 'correct_answers': 1, 'completed_practices': 1},
        {'user_id': 'silver-a', 'league_tier': 'silver_3', 'points': 300, 'correct_answers': 10, 'completed_practices': 4},
        {'user_id': 'silver-b', 'league_tier': 'silver_3', 'points': 200, 'correct_answers': 8, 'completed_practices': 3},
        {'user_id': 'silver-c', 'league_tier': 'silver_3', 'points': 100, 'correct_answers': 4, 'completed_practices': 2},
    ])

    assert movements['bronze-a'] == {'final_tier': 'bronze_2', 'movement': 'promoted'}
    assert movements['bronze-c'] == {'final_tier': 'bronze_3', 'movement': 'held'}
    assert movements['silver-a'] == {'final_tier': 'silver_2', 'movement': 'promoted'}
    assert movements['silver-c'] == {'final_tier': 'bronze_1', 'movement': 'demoted'}


def test_small_leagues_do_not_demote():
    movements = resolve_season_movements([
        {'user_id': 'gold-a', 'league_tier': 'gold_3', 'points': 10, 'correct_answers': 1, 'completed_practices': 1},
        {'user_id': 'gold-b', 'league_tier': 'gold_3', 'points': 0, 'correct_answers': 0, 'completed_practices': 0},
    ])

    assert movements['gold-b'] == {'final_tier': 'gold_3', 'movement': 'held'}


def test_first_inactive_season_is_protected_from_demotion():
    movements = resolve_season_movements([
        {'user_id': 'silver-a', 'league_tier': 'silver_3', 'points': 200, 'correct_answers': 4, 'completed_practices': 2},
        {'user_id': 'silver-b', 'league_tier': 'silver_3', 'points': 150, 'correct_answers': 3, 'completed_practices': 1},
        {'user_id': 'silver-c', 'league_tier': 'silver_3', 'points': 0, 'correct_answers': 0, 'completed_practices': 0, 'inactive_seasons': 0},
    ])

    assert movements['silver-c'] == {'final_tier': 'silver_3', 'movement': 'held'}
    assert movements['silver-b'] == {'final_tier': 'bronze_1', 'movement': 'demoted'}


def test_inactive_student_at_fifty_percent_floor_is_not_demoted():
    movements = resolve_season_movements([
        {'user_id': 'gold-a', 'league_tier': 'gold_3', 'points': 300, 'correct_answers': 4, 'completed_practices': 2},
        {'user_id': 'gold-b', 'league_tier': 'gold_3', 'points': 100, 'correct_answers': 2, 'completed_practices': 1},
        {
            'user_id': 'gold-c',
            'league_tier': 'gold_3',
            'points': 50,
            'correct_answers': 0,
            'completed_practices': 0,
            'inactive_seasons': 4,
            'inactivity_baseline_points': 100,
            'inactivity_floor_reached': True,
        },
    ])

    assert movements['gold-c'] == {'final_tier': 'gold_3', 'movement': 'held'}
    assert movements['gold-b'] == {'final_tier': 'silver_1', 'movement': 'demoted'}
