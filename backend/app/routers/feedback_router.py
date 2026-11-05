from fastapi import APIRouter, Depends
from ..models import InputDataAnswer, ErrorResponse, DirectFeedbackRequest, MergedFeedbackResponse, DirectFeedbackResponse
from ..auth import require_user
from .. import feedback_service

router = APIRouter(dependencies=[Depends(require_user)])

@router.post("/gen_answer", response_model=MergedFeedbackResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
def generate_feedback(InputDataAnswer: InputDataAnswer):
    # Pydantic handles validation
    merged = feedback_service.generate_merged_feedback(
        InputDataAnswer.image_url,
        InputDataAnswer.questions,
    )
    return {"merged": merged}

@router.post("/gen_feedback_direct", response_model=DirectFeedbackResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
def generate_feedback_direct(request: DirectFeedbackRequest):
    """
    Directly generate feedback for questions and answers (text or image).
    Bypasses the answer separation logic used for bulk uploads.
    """
    return {"feedback": feedback_service.generate_direct_feedback(request)}
