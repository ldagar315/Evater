from fastapi import APIRouter, Depends

from ..auth import AuthContext, require_user

router = APIRouter(dependencies=[Depends(require_user)])


@router.get("/me")
def me(auth: AuthContext = Depends(require_user)):
    return {
        "user_id": auth.user.id,
        "email": auth.user.email,
        "tenant_id": auth.tenant_id,
    }

