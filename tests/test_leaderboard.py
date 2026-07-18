from fastapi.testclient import TestClient

from app.main import app
from app.routers.leaderboard_router import build_leaderboard_entries


def test_leaderboard_requires_authentication():
    response = TestClient(app).get('/api/v1/leaderboard')
    assert response.status_code == 401


def test_leaderboard_scores_completed_practice_and_keeps_zero_score_members():
    entries = build_leaderboard_entries(
        members=[
            {'user_id': 'user-a', 'name': 'Asha Rao'},
            {'user_id': 'user-b', 'name': 'Bharat Sen'},
            {'user_id': 'user-c', 'name': 'Chirag'},
        ],
        completed_tests=[
            {'id': 'test-a', 'user_id': 'user-a'},
            {'id': 'test-b', 'user_id': 'user-b'},
        ],
        question_attempts=[
            {'user_id': 'user-a', 'test_attempt_id': 'test-a', 'is_correct': True},
            {'user_id': 'user-a', 'test_attempt_id': 'test-a', 'is_correct': True},
            {'user_id': 'user-b', 'test_attempt_id': 'test-b', 'is_correct': True},
            {'user_id': 'user-b', 'test_attempt_id': 'test-b', 'is_correct': False},
        ],
        current_user_id='user-b',
    )

    assert [(entry.rank, entry.display_name, entry.score) for entry in entries] == [
        (1, 'Asha R.', 45),
        (2, 'Bharat S.', 35),
        (3, 'Chirag', 0),
    ]
    assert entries[1].is_current_user is True


def test_leaderboard_ties_use_correct_answers_then_name():
    entries = build_leaderboard_entries(
        members=[
            {'user_id': 'user-z', 'name': 'Zoya'},
            {'user_id': 'user-a', 'name': 'Aarav'},
        ],
        completed_tests=[
            {'id': 'test-z', 'user_id': 'user-z'},
            {'id': 'test-a', 'user_id': 'user-a'},
        ],
        question_attempts=[
            {'user_id': 'user-z', 'test_attempt_id': 'test-z', 'is_correct': True},
            {'user_id': 'user-a', 'test_attempt_id': 'test-a', 'is_correct': True},
        ],
        current_user_id='user-a',
    )

    assert [entry.display_name for entry in entries] == ['Aarav', 'Zoya']
