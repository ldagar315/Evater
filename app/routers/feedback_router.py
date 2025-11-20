from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from ..models import InputDataAnswer, ErrorResponse, Question, DirectFeedbackRequest, Answer
from ..dspy_modules import answer_seperation, feedback_generation, ocr_text
from ..services import answer_ocr_extraction, merge_qaf, ocr_text_single_image

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

@router.post("/gen_feedback_direct", response_model=Dict[str, Any], responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
def generate_feedback_direct(request: DirectFeedbackRequest):
    """
    Directly generate feedback for questions and answers (text or image).
    Bypasses the answer separation logic used for bulk uploads.
    """
    feedback_list = []
    
    # Create a map of questions for easy lookup
    question_map = {q.question_number: q for q in request.questions}
    
    for ans_input in request.answers:
        question = question_map.get(ans_input.question_number)
        if not question:
            continue
            
        final_answer_text = ""
        
        # If text answer is provided, use it
        if ans_input.answer_text:
            final_answer_text = ans_input.answer_text
            
        # If image is provided, run OCR and append/replace
        if ans_input.image_url:
            ocr_result = ocr_text_single_image(ans_input.image_url)
            if final_answer_text:
                final_answer_text += f"\n\n[Image Content]: {ocr_result}"
            else:
                final_answer_text = ocr_result
        
        if not final_answer_text:
            final_answer_text = "No answer provided."

        # Create Answer object
        answer_obj = Answer(
            question_number=ans_input.question_number,
            answer=final_answer_text
        )
        
        # Generate Feedback
        feedback = feedback_generation(question=question, answer=answer_obj.answer)
        feedback_list.append(feedback.feedback.model_dump())

    return {"feedback": feedback_list}
