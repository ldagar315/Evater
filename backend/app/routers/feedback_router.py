from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from ..models import InputDataAnswer, ErrorResponse, DirectFeedbackRequest, MergedFeedbackResponse, DirectFeedbackResponse
from ..auth import AuthContext, require_user
from .. import feedback_service
from ..client_version import require_supported_client

router = APIRouter(dependencies=[Depends(require_user)])

@router.post("/gen_answer", response_model=MergedFeedbackResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
def generate_feedback(InputDataAnswer: InputDataAnswer, auth: AuthContext = Depends(require_user), x_evater_client: Optional[str] = Header(None)):
    require_supported_client(x_evater_client)
    try:
        merged = feedback_service.generate_merged_feedback(
            InputDataAnswer.image_url,
            InputDataAnswer.questions,
            test_id=InputDataAnswer.test_id,
            jwt=auth.jwt,
            user_id=auth.user.id,
        )
    except feedback_service.TestNotFound:
        raise HTTPException(status_code=404, detail="Test not found.")
    except feedback_service.AnswerKeysUnavailable:
        raise HTTPException(status_code=409, detail="Answer keys are unavailable for this test, so its objective questions cannot be graded.")
    return {"merged": merged}

@router.post("/gen_feedback_direct", response_model=DirectFeedbackResponse, responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
def generate_feedback_direct(request: DirectFeedbackRequest, auth: AuthContext = Depends(require_user), x_evater_client: Optional[str] = Header(None)):
    require_supported_client(x_evater_client)
    """
    Directly generate feedback for questions and answers (text or image).
    Bypasses the answer separation logic used for bulk uploads.
    """
    try:
        feedback = feedback_service.generate_direct_feedback(
            request, jwt=auth.jwt, user_id=auth.user.id
        )
    except feedback_service.TestNotFound:
        raise HTTPException(status_code=404, detail="Test not found.")
    except feedback_service.AnswerKeysUnavailable:
        raise HTTPException(status_code=409, detail="Answer keys are unavailable for this test, so its objective questions cannot be graded.")
    return {"feedback": feedback}
