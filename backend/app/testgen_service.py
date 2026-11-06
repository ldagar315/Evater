"""Test-generation flow, extracted from the test router.

Pure Python - no FastAPI imports. LLM/DB callables are injectable so the
core path is unit-testable with mocked dspy modules and a stubbed Supabase
client. None means "use the real default", resolved at call time.
"""
from typing import Any, Dict, List, Optional

from . import dspy_modules, services, supabase_client as supabase_client_module
from .models import InputDataQuestion

_DISTRIBUTION_BY_TEST_TYPE = {
    "objective": "result_distribution_mcq",
    "subjective": "result_distribution_subjective",
}
_DEFAULT_DISTRIBUTION = "result_distribution"


def generate_questions(
    request: InputDataQuestion,
    *,
    jwt: Optional[str] = None,
    distribution_fns: Optional[Dict[str, Any]] = None,
    chapter_summary_fn=None,
    test_generation_fn=None,
    client_factory=None,
    maximum_marks_fn=None,
) -> List[Dict[str, Any]]:
    """gen_question flow: 2 LLM calls (structure, then test) + 1 DB read."""
    distribution_fns = distribution_fns or {}
    chapter_summary_fn = chapter_summary_fn or services.get_chapter_summary
    test_generation_fn = test_generation_fn or dspy_modules.test_generation
    client_factory = client_factory or supabase_client_module.create_supabase_client
    maximum_marks_fn = maximum_marks_fn or services.maximum_marks

    dist_name = _DISTRIBUTION_BY_TEST_TYPE.get(request.test_type, _DEFAULT_DISTRIBUTION)
    distribution_fn = distribution_fns.get(dist_name) or getattr(dspy_modules, dist_name)
    result = distribution_fn(
        difficulty_level=request.difficulty_level,
        subject=request.subject,
        length=request.length,
        special_instructions=request.special_instructions,
    )

    client = client_factory(jwt)
    summary_of_key_points = chapter_summary_fn(
        chapter_name=request.topic,
        grade=request.grade,
        subject=request.subject,
        supabase_client=client,
    )

    generated_test = test_generation_fn(
        topic=request.topic,
        topic_covered=summary_of_key_points,
        subject=request.subject,
        grade=request.grade,
        test_structure=result.test_structure,
        difficulty=request.difficulty_level,
    )

    questions = [q.model_dump() for q in generated_test.test]
    for q in questions:
        q["maximum_marks"] = maximum_marks_fn(q["question_type"])
    return [_strip_answer_keys(q) for q in questions]


def _strip_answer_keys(question: Dict[str, Any]) -> Dict[str, Any]:
    """Never send answer keys to clients: drop is_correct from MCQ options.

    The question-bank v1 flow makes the same guarantee via PublicQuestionOption;
    this keeps gen-flow options key-free at the service boundary, with the
    router's response_model as a second layer."""
    options = question.get("options")
    if options:
        question["options"] = [
            {k: v for k, v in opt.items() if k != "is_correct"} for opt in options
        ]
    return question
