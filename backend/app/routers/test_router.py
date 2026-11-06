from fastapi import APIRouter, Depends
from ..models import InputDataQuestion, ErrorResponse, QuestionsResponse, PublicQuestionsResponse
from ..auth import AuthContext, require_user
from .. import testgen_service

router = APIRouter(dependencies=[Depends(require_user)])

@router.post("/gen_question", response_model=PublicQuestionsResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
def generate_questions(InputDataQuestion: InputDataQuestion, auth: AuthContext = Depends(require_user)):
    """ LLM calls X 2
    Database Call X 1
    Outputs -> A list of questions objects """
    questions = testgen_service.generate_questions(InputDataQuestion, jwt=auth.jwt)
    return {"questions": questions}
