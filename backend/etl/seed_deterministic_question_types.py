"""Seed local-stage examples for every deterministic learner question type.

These fixtures are intentionally separate from scraped content. They make the
renderer and scorer testable immediately while source answer keys for the
archive are still being normalized. They are not a production content pack.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from .class8_science_catalog import CHAPTERS, CURRICULUM_VERSION_ID


CHAPTER_ID = str(CHAPTERS[0].id)
CONCEPT_IDS = {
    "questions": "33333333-3333-4333-8333-333333333301",
    "variables": "33333333-3333-4333-8333-333333333302",
    "observation": "33333333-3333-4333-8333-333333333303",
    "measurement": "33333333-3333-4333-8333-333333333304",
}


def _fixture(
    *,
    question_type: str,
    concept_key: str,
    question_text: str,
    options: list[dict[str, str]],
    answer_spec: dict[str, Any],
    explanation: str,
    marks: int = 1,
    render_config: dict[str, Any] | None = None,
    media: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    content_hash = hashlib.sha256(f"local-stage:{question_type}:{question_text}".encode()).hexdigest()
    now = datetime.now(timezone.utc).isoformat()
    return {
        "curriculum_version_id": str(CURRICULUM_VERSION_ID),
        "chapter_id": CHAPTER_ID,
        "concept_id": CONCEPT_IDS[concept_key],
        "question_type": question_type,
        "question_text": question_text,
        "options_json": options,
        "correct_option_id": answer_spec.get("correct_option_id"),
        "explanation": explanation,
        "hint": None,
        "difficulty": "medium",
        "cognitive_level": "understand",
        "skill_tags": ["local_stage_fixture", question_type],
        "misconception_tags": [],
        "question_style": "scenario" if question_type == "case_study" else "diagram" if question_type == "diagram_based" else "direct",
        "estimated_time_seconds": 120 if question_type in {"case_study", "diagram_based", "matching"} else 60,
        "marks": marks,
        "quality_score": 1.0,
        "content_hash": content_hash,
        "status": "published",
        "published_at": now,
        "source_url": "manual://local-stage/deterministic-question-types",
        "source_question_id": f"local-stage:{question_type}",
        "source_locator": "Local stage deterministic renderer fixture",
        "question_family_key": f"local-stage:{question_type}",
        "variant_key": "deterministic-v1",
        "media_json": media or [],
        "option_media_json": {},
        "render_config": render_config or {},
        "answer_spec": {"type": question_type, **answer_spec},
        "generation_spec": {"generator": "local-stage-deterministic-types-v1"},
        "reviewed_at": now,
    }


def deterministic_fixtures() -> list[dict[str, Any]]:
    return [
        _fixture(
            question_type="mcq_multi",
            concept_key="questions",
            question_text="Which two actions help make an experiment a fair test?",
            options=[
                {"id": "A", "text": "Change one variable at a time."},
                {"id": "B", "text": "Record observations carefully."},
                {"id": "C", "text": "Change every condition together."},
                {"id": "D", "text": "Ignore unexpected results."},
            ],
            answer_spec={"correct_option_ids": ["A", "B"]},
            explanation="A fair test changes one variable at a time and records the evidence carefully.",
        ),
        _fixture(
            question_type="assertion_reason",
            concept_key="observation",
            question_text="Choose the correct relationship. Assertion: Repeating an observation improves confidence in a result. Reason: Repeated observations help reveal whether a result is consistent.",
            options=[
                {"id": "A", "text": "Both are true, and the reason explains the assertion."},
                {"id": "B", "text": "Both are true, but the reason does not explain the assertion."},
                {"id": "C", "text": "The assertion is true, but the reason is false."},
                {"id": "D", "text": "The assertion is false, but the reason is true."},
            ],
            answer_spec={"correct_option_id": "A"},
            render_config={
                "assertion": "Repeating an observation improves confidence in a result.",
                "reason": "Repeated observations help reveal whether a result is consistent.",
            },
            explanation="Repeated observations help us check repeatability, so the reason explains the assertion.",
        ),
        _fixture(
            question_type="true_false",
            concept_key="variables",
            question_text="A controlled variable is kept the same during a fair test.",
            options=[{"id": "A", "text": "True"}, {"id": "B", "text": "False"}],
            answer_spec={"correct_option_id": "A"},
            explanation="Keeping controlled variables the same makes it easier to link a result to the variable being tested.",
        ),
        _fixture(
            question_type="fill_blank",
            concept_key="variables",
            question_text="In a fair test, the factor kept unchanged is called a ______ variable.",
            options=[],
            answer_spec={"accepted_answers": ["controlled", "control"]},
            explanation="A controlled variable is deliberately kept unchanged while another variable is tested.",
        ),
        _fixture(
            question_type="numerical",
            concept_key="measurement",
            question_text="A learner records lengths of 12 cm, 15 cm, and 18 cm. What is the mean length?",
            options=[],
            answer_spec={"expected_value": 15, "tolerance": 0.01, "unit": "cm"},
            explanation="The mean is (12 + 15 + 18) ÷ 3 = 15 cm.",
            render_config={"unit": "cm"},
        ),
        _fixture(
            question_type="case_study",
            concept_key="variables",
            question_text="Which conclusion is best supported by this investigation?",
            options=[
                {"id": "A", "text": "The changed variable is the independent variable."},
                {"id": "B", "text": "Every observation is automatically a conclusion."},
                {"id": "C", "text": "A fair test changes all conditions together."},
                {"id": "D", "text": "A conclusion does not need evidence."},
            ],
            answer_spec={"correct_option_id": "A"},
            explanation="The independent variable is the condition deliberately changed by the investigator.",
            render_config={"passage": "A group changes only the amount of sunlight given to identical seedlings. It measures their height after one week and keeps the soil, water, and pot size the same."},
        ),
        _fixture(
            question_type="diagram_based",
            concept_key="questions",
            question_text="Which label represents the variable that the investigator deliberately changes in this experiment diagram?",
            options=[
                {"id": "A", "text": "The independent variable"},
                {"id": "B", "text": "The dependent variable"},
                {"id": "C", "text": "A controlled variable"},
                {"id": "D", "text": "The conclusion"},
            ],
            answer_spec={"correct_option_id": "A"},
            explanation="The independent variable is the condition the investigator deliberately changes.",
            render_config={"diagram_type": "investigation_flow", "labels": ["Change", "Measure", "Keep same"]},
        ),
        _fixture(
            question_type="matching",
            concept_key="variables",
            question_text="Match each investigation term with its role.",
            options=[],
            answer_spec={"pairs": {"independent": "changed", "dependent": "measured", "controlled": "kept_same"}},
            explanation="The independent variable is changed, the dependent variable is measured, and controlled variables are kept the same.",
            marks=3,
            render_config={
                "prompts": [
                    {"id": "independent", "text": "Independent variable"},
                    {"id": "dependent", "text": "Dependent variable"},
                    {"id": "controlled", "text": "Controlled variable"},
                ],
                "choices": [
                    {"id": "changed", "text": "Changed"},
                    {"id": "measured", "text": "Measured"},
                    {"id": "kept_same", "text": "Kept the same"},
                ],
            },
        ),
    ]


def main() -> int:
    from app.supabase_client import create_supabase_service_client

    client = create_supabase_service_client()
    rows = deterministic_fixtures()
    client.table("question_bank").upsert(rows, on_conflict="chapter_id,content_hash").execute()
    print(f"Published {len(rows)} deterministic question-type fixtures for local stage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
