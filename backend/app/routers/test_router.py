from fastapi import APIRouter, Depends, HTTPException
from ..models import InputDataQuestion, ErrorResponse, QuestionsResponse, PublicQuestionsResponse
from ..auth import AuthContext, require_user
from .. import testgen_service
from ..supabase_client import create_supabase_service_client

router = APIRouter(dependencies=[Depends(require_user)])

@router.post("/gen_question", response_model=PublicQuestionsResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
def generate_questions(InputDataQuestion: InputDataQuestion, auth: AuthContext = Depends(require_user)):
    """ LLM calls X 2
    Database Call X 1
    Outputs -> A list of questions objects """
    questions = testgen_service.generate_questions(InputDataQuestion, jwt=auth.jwt)

    # Persist answer keys server-side before anything leaves: feedback grading
    # hydrates from this store, and clients only ever receive the sanitized
    # questions plus this opaque reference. Fail closed: a test whose keys were
    # not persisted could never be graded objectively.
    try:
        inserted = (
            create_supabase_service_client()
            .table("generated_test_keys")
            .insert({"user_id": auth.user.id, "questions": questions})
            .execute()
        )
        key_id = inserted.data[0]["id"]
    except Exception:
        raise HTTPException(status_code=500, detail="Could not securely store the test's answer keys. Please try again.")

    return {"questions": testgen_service.sanitize_questions(questions), "key_id": key_id}
