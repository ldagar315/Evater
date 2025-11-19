from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from ..models import InputDataQuestion, ErrorResponse
from ..dspy_modules import (
    result_distribution, result_distribution_mcq, result_distribution_subjective, 
    test_generation
)
from ..services import get_chapter_summary, maximum_marks

router = APIRouter()

@router.post("/gen_question", response_model=Dict[str, Any], responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
def generate_questions(InputDataQuestion: InputDataQuestion):
    """ LLM calls X 2
    Database Call X 1 
    Outputs -> A list of questions objects """
    
    # Pydantic handles validation of required fields automatically via InputDataQuestion model
    
    if "only mcq" in InputDataQuestion.special_instructions:
        result = result_distribution_mcq(
            difficulty_level= InputDataQuestion.difficulty_level,
            subject = InputDataQuestion.subject,
            length = InputDataQuestion.length, 
            special_instructions = InputDataQuestion.special_instructions
        )
    elif "only subjective" in InputDataQuestion.special_instructions:
        result = result_distribution_subjective(
            difficulty_level= InputDataQuestion.difficulty_level,
            subject = InputDataQuestion.subject,
            length = InputDataQuestion.length, 
            special_instructions = InputDataQuestion.special_instructions
        )
    else:
        result = result_distribution(
            difficulty_level= InputDataQuestion.difficulty_level,
            subject = InputDataQuestion.subject,
            length = InputDataQuestion.length, 
            special_instructions = InputDataQuestion.special_instructions
        )
    
    summary_of_key_points = get_chapter_summary(
        chapter_name =  InputDataQuestion.topic,
        grade = InputDataQuestion.grade,
        subject = InputDataQuestion.subject
    )
    
    generated_test = test_generation(
        topic=InputDataQuestion.topic,
        topic_covered = summary_of_key_points,
        subject = InputDataQuestion.subject,
        grade = InputDataQuestion.grade, 
        test_structure= result.test_structure, # Access the output field correctly
        difficulty = InputDataQuestion.difficulty_level
    )

    questions =  [q.model_dump() for q in generated_test.test]
    for i in questions:
        i["maximum_marks"] = maximum_marks(i["question_type"])
    return {"questions": questions}
