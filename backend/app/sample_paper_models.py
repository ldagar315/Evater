"""Strict anonymous response contracts. No database or grading changes."""
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

class PublicModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

class ChapterSummary(PublicModel):
    sequence_number: StrictInt = Field(ge=1)
    title: StrictStr = Field(min_length=1)
    slug: StrictStr = Field(min_length=1)
    published_count: StrictInt = Field(ge=0)

class SubjectIndex(PublicModel):
    subject: StrictStr = Field(min_length=1)
    subject_name: StrictStr = Field(min_length=1)
    chapters: List[ChapterSummary]

class SamplePapersIndex(PublicModel):
    grade: StrictInt = Field(ge=6, le=10)
    subjects: List[SubjectIndex]

class PaperChapter(PublicModel):
    sequence_number: StrictInt = Field(ge=1)
    title: StrictStr = Field(min_length=1)
    slug: StrictStr = Field(min_length=1)
    description: Optional[StrictStr] = None

class PaperOption(PublicModel):
    id: StrictStr = Field(min_length=1)
    text: StrictStr = Field(min_length=1)
    # Existing public-media payload is retained; its internal schema is a separate audit.
    media: List[Dict[str, Any]] = Field(default_factory=list)

class PaperQuestion(PublicModel):
    position: StrictInt = Field(ge=1,le=15)
    id: StrictStr = Field(min_length=1)
    question_type: Literal['mcq_single','mcq_multi','true_false','numerical','fill_blank','assertion_reason','case_study','diagram_based','matching']
    question_text: StrictStr = Field(min_length=1)
    options: List[PaperOption]
    answer_summary: StrictStr = Field(min_length=1)

class SamplePaper(PublicModel):
    grade: StrictInt = Field(ge=6,le=10)
    subject: StrictStr = Field(min_length=1)
    subject_name: StrictStr = Field(min_length=1)
    chapter: PaperChapter
    question_count: Literal[15]
    questions: List[PaperQuestion] = Field(min_length=15,max_length=15)

    @model_validator(mode='after')
    def complete_positions(self):
        if [q.position for q in self.questions] != list(range(1,16)):
            raise ValueError('Paper must have ordered positions1 through15')
        if len({q.id for q in self.questions}) != 15:
            raise ValueError('Paper question IDs must be unique')
        return self
