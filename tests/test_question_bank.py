import pytest

from app.question_bank import (
    QuestionBankError,
    build_routing_decision,
    normalize_options,
    score_block,
    select_question_block,
    to_public_question,
    update_mastery,
)
from app.models import QuestionBankTestResponse


def make_question(index: int, concept_id: str, difficulty: str = "medium"):
    return {
        "id": f"00000000-0000-0000-0000-{index:012d}",
        "concept_id": concept_id,
        "question_text": f"Question {index} asks something useful?",
        "options_json": [
            {"id": "A", "text": "Option A"},
            {"id": "B", "text": "Option B"},
            {"id": "C", "text": "Option C"},
            {"id": "D", "text": "Option D"},
        ],
        "correct_option_id": "B",
        "explanation": "Because option B is correct.",
        "difficulty": difficulty,
        "cognitive_level": "understand",
        "skill_tags": ["classification"],
        "misconception_tags": [],
        "question_style": "direct",
        "estimated_time_seconds": 60,
        "marks": 1,
    }


def test_published_question_options_require_exactly_four_choices():
    with pytest.raises(QuestionBankError):
        normalize_options([{"id": "A", "text": "Only one"}])


def test_selection_is_deterministic_and_avoids_recent_questions():
    questions = [
        make_question(1, "weak", "easy"),
        make_question(2, "weak", "medium"),
        make_question(3, "strong", "hard"),
        make_question(4, "strong", "medium"),
    ]

    first = select_question_block(
        questions=questions,
        mastery={"weak": 20, "strong": 90},
        seen_question_ids=[],
        seed=42,
        count=2,
        routing={"difficulty": "easy", "focus_concept_ids": ["weak"]},
    )
    second = select_question_block(
        questions=questions,
        mastery={"weak": 20, "strong": 90},
        seen_question_ids=[],
        seed=42,
        count=2,
        routing={"difficulty": "easy", "focus_concept_ids": ["weak"]},
    )

    assert [question["id"] for question in first["questions"]] == [
        question["id"] for question in second["questions"]
    ]
    assert all(question["concept_id"] == "weak" for question in first["questions"])

    next_block = select_question_block(
        questions=questions,
        mastery={"weak": 20, "strong": 90},
        seen_question_ids=[question["id"] for question in first["questions"]],
        seed=42,
        count=2,
        routing={"difficulty": "hard", "focus_concept_ids": []},
        block_number=2,
    )
    assert not ({question["id"] for question in first["questions"]} & {
        question["id"] for question in next_block["questions"]
    })


def test_scoring_is_server_side_and_unanswered_is_incorrect():
    questions = [make_question(1, "concept-a"), make_question(2, "concept-b")]
    results = score_block(
        questions,
        {
            questions[0]["id"]: "B",
            questions[1]["id"]: None,
        },
    )

    assert results[0]["is_correct"] is True
    assert results[0]["marks_awarded"] == 1
    assert results[1]["is_correct"] is False
    assert results[1]["marks_awarded"] == 0


def test_all_supported_non_freeform_question_types_have_deterministic_scoring():
    questions = [
        {
            **make_question(10, "concept-a"),
            "question_type": "mcq_multi",
            "answer_spec": {"correct_option_ids": ["A", "C"]},
        },
        {
            **make_question(11, "concept-b"),
            "question_type": "assertion_reason",
            "answer_spec": {"correct_option_id": "B"},
        },
        {
            **make_question(12, "concept-c"),
            "question_type": "true_false",
            "options_json": [{"id": "A", "text": "True"}, {"id": "B", "text": "False"}],
            "correct_option_id": "A",
        },
        {
            **make_question(13, "concept-d"),
            "question_type": "fill_blank",
            "options_json": [],
            "correct_option_id": None,
            "answer_spec": {"accepted_answers": ["controlled", "control"]},
        },
        {
            **make_question(14, "concept-e"),
            "question_type": "numerical",
            "options_json": [],
            "correct_option_id": None,
            "answer_spec": {"expected_value": 15, "tolerance": 0.1},
        },
        {
            **make_question(15, "concept-f"),
            "question_type": "case_study",
            "answer_spec": {"correct_option_id": "C"},
            "correct_option_id": None,
        },
        {
            **make_question(16, "concept-g"),
            "question_type": "diagram_based",
            "answer_spec": {"correct_option_id": "D"},
            "correct_option_id": None,
        },
        {
            **make_question(17, "concept-h"),
            "question_type": "matching",
            "options_json": [],
            "correct_option_id": None,
            "answer_spec": {"pairs": {"one": "a", "two": "b"}},
            "marks": 2,
        },
    ]
    answers = {
        questions[0]["id"]: ["C", "A"],
        questions[1]["id"]: "B",
        questions[2]["id"]: "A",
        questions[3]["id"]: "controlled",
        questions[4]["id"]: "15.05",
        questions[5]["id"]: "C",
        questions[6]["id"]: "D",
        questions[7]["id"]: {"one": "a", "two": "b"},
    }

    results = score_block(questions, answers)

    assert all(result["is_correct"] for result in results)
    assert results[0]["correct_option_ids"] == ["A", "C"]
    assert results[-1]["marks_awarded"] == 2


def test_matching_awards_partial_marks_deterministically():
    question = {
        **make_question(18, "concept-a"),
        "question_type": "matching",
        "options_json": [],
        "correct_option_id": None,
        "answer_spec": {"pairs": {"one": "a", "two": "b", "three": "c"}},
        "marks": 3,
    }

    result = score_block([question], {question["id"]: {"one": "a", "two": "wrong", "three": "c"}})[0]

    assert result["is_correct"] is False
    assert result["marks_awarded"] == 2


def test_routing_focuses_weak_concepts_and_promotes_high_performance():
    results = [
        {"question_id": "q1", "concept_id": "weak", "is_correct": False, "marks_awarded": 0, "maximum_marks": 1},
        {"question_id": "q2", "concept_id": "weak", "is_correct": False, "marks_awarded": 0, "maximum_marks": 1},
        {"question_id": "q3", "concept_id": "strong", "is_correct": True, "marks_awarded": 1, "maximum_marks": 1},
        {"question_id": "q4", "concept_id": "strong", "is_correct": True, "marks_awarded": 1, "maximum_marks": 1},
    ]

    routing = build_routing_decision(results, previous_difficulty="medium")

    assert routing["status"] == "on_track"
    assert routing["focus_concept_ids"] == ["weak"]
    assert routing["difficulty"] == "medium"

    challenge = build_routing_decision(
        [
            {"question_id": "q1", "concept_id": "strong", "is_correct": True, "marks_awarded": 1, "maximum_marks": 1},
            {"question_id": "q2", "concept_id": "strong", "is_correct": True, "marks_awarded": 1, "maximum_marks": 1},
        ],
        previous_difficulty="medium",
    )
    assert challenge["status"] == "challenge_next"
    assert challenge["difficulty"] == "hard"


def test_mastery_update_is_bounded_and_accumulates_attempts():
    updates = update_mastery(
        [
            {"concept_id": "concept-a", "is_correct": True, "difficulty": "medium"},
            {"concept_id": "concept-a", "is_correct": False, "difficulty": "medium"},
        ],
        {
            "concept-a": {
                "mastery_score": 40,
                "attempt_count": 2,
                "correct_count": 1,
            }
        },
    )

    assert updates[0]["concept_id"] == "concept-a"
    assert updates[0]["mastery_score"] == 43
    assert updates[0]["attempt_count"] == 4
    assert updates[0]["correct_count"] == 2


def test_public_question_response_shape_excludes_the_answer_key():
    question = make_question(1, "concept-a")
    question["concept_id"] = "33333333-3333-4333-8333-333333333301"
    public = {
        "id": question["id"],
        "concept_id": question["concept_id"],
        "question_type": "mcq_single",
        "question_text": question["question_text"],
        "options": question["options_json"],
        "difficulty": question["difficulty"],
        "cognitive_level": question["cognitive_level"],
        "skill_tags": question["skill_tags"],
        "misconception_tags": question["misconception_tags"],
        "question_style": question["question_style"],
        "estimated_time_seconds": question["estimated_time_seconds"],
        "maximum_marks": question["marks"],
    }
    response = QuestionBankTestResponse.model_validate(
        {
            "test_id": "10000000-0000-4000-8000-000000000001",
            "block_number": 1,
            "block_size": 5,
            "question_count": 10,
            "questions": [public],
            "routing": {"difficulty": "medium", "focus_concept_ids": [], "status": "on_track"},
        }
    )

    assert response.questions[0].options[0].id == "A"
    assert "correct_option_id" not in response.questions[0].model_dump()


def test_public_question_preserves_question_and_option_media():
    question = make_question(1, "concept-a")
    question["media_json"] = [
        {"type": "image", "url": "/question.svg", "alt": "Question diagram"}
    ]
    question["option_media_json"] = {
        "B": [{"type": "image", "url": "/option.svg", "alt": "Option diagram"}]
    }

    public = to_public_question(question)

    assert public["media"][0]["url"] == "/question.svg"
    assert public["options"][1]["media"][0]["url"] == "/option.svg"
