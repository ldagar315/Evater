from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from ..models import InputDataAnswer, ErrorResponse, Question
from ..dspy_modules import answer_seperation, feedback_generation
from ..services import answer_ocr_extraction, merge_qaf

router = APIRouter()

@router.post("/gen_answer", response_model=Dict[str, Any], responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
def generate_feedback(InputDataAnswer: InputDataAnswer):
    # Pydantic handles validation
    
    image_url = InputDataAnswer.image_url
    questions = InputDataAnswer.questions

    text = answer_ocr_extraction(image_url)

    seperated_answers = answer_seperation(
        answer_sheet_text = text
    )

    feedback_list = []
    # Validate and parse questions
    question_object_list = [Question.model_validate(q) for q in questions['questions']]
    question_map = {q.question_number: q for q in question_object_list}

    for answer in seperated_answers.answers:
        question_object_for_the_answer = question_map.get(answer.question_number)

        if question_object_for_the_answer:
            feedback = feedback_generation(question=question_object_for_the_answer, answer=answer.answer)
            feedback_list.append(feedback)
            
    feedback_list_new = [feedback.feedback.model_dump() for feedback in feedback_list]
    answer_list = [answer.model_dump() for answer in seperated_answers.answers]
    questions_list = questions['questions']

    merged = merge_qaf(
        questions_list = questions_list,
        answer_list = answer_list,
        feedback_list_new = feedback_list_new
    )
    return {"merged" : merged }
