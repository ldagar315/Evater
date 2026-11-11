"""Unregistered mixed-paper contract for non-production review.

No main-router registration or production data loader exists. A reviewed loader
must provide provenance-bearing content; the application fails closed by default.
"""
from typing import List, Literal, Callable
from fastapi import APIRouter, HTTPException
from pydantic import Field, StrictInt, StrictStr, model_validator
from .sample_paper_models import PublicModel

class ObjectiveOption(PublicModel):
    label: StrictStr = Field(min_length=1)
    text: StrictStr = Field(min_length=1)

class ObjectiveQuestion(PublicModel):
    position: StrictInt = Field(ge=1,le=10)
    bank_id: StrictStr = Field(min_length=1)
    content_hash: StrictStr = Field(min_length=1)
    question_type: Literal['objective'] = 'objective'
    prompt: StrictStr = Field(min_length=1)
    options: List[ObjectiveOption]
    answer: StrictStr = Field(min_length=1)

class SubjectiveQuestion(PublicModel):
    position: StrictInt = Field(ge=11,le=15)
    question_type: Literal['subjective'] = 'subjective'
    prompt: StrictStr = Field(min_length=1)
    marks: StrictInt = Field(ge=1,le=10)
    model_answer: StrictStr = Field(min_length=1)
    marking_criteria: List[StrictStr] = Field(min_length=1)
    source_url: StrictStr = Field(pattern=r'^https://')
    source_page: StrictInt = Field(ge=1)

class MixedSamplePaper(PublicModel):
    format: Literal['10-objective-5-subjective-v1']
    status: Literal['review-only']
    grade: StrictInt = Field(ge=6,le=10)
    title: StrictStr = Field(min_length=1)
    objective: List[ObjectiveQuestion] = Field(min_length=10,max_length=10)
    subjective: List[SubjectiveQuestion] = Field(min_length=5,max_length=5)
    @model_validator(mode='after')
    def positions(self):
        if [q.position for q in self.objective]!=list(range(1,11)) or [q.position for q in self.subjective]!=list(range(11,16)):
            raise ValueError('Sections must contain ordered positions1-10 and11-15')
        if len({q.bank_id for q in self.objective})!=10:
            raise ValueError('Objective bank IDs must be unique')
        return self

def review_router(loader: Callable[[str],dict] | None=None) -> APIRouter:
    """Explicit caller registration only. Never imported by app.main."""
    router=APIRouter()
    @router.get('/review/mixed-sample-papers/{paper_id}', response_model=MixedSamplePaper)
    def get_paper(paper_id: str):
        if loader is None:
            raise HTTPException(503, 'Mixed paper review loader is not configured.')
        try:
            return loader(paper_id)
        except KeyError:
            raise HTTPException(404, 'Review paper not found.')
    return router
