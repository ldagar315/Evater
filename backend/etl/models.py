from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


class CandidateOption(BaseModel):
    id: Literal["A", "B", "C", "D"]
    text: str = Field(min_length=1, max_length=500)


class QuestionCandidate(BaseModel):
    chapter_id: UUID
    concept_id: UUID
    question_text: str = Field(min_length=10, max_length=1000)
    options: List[CandidateOption]
    correct_option_id: Literal["A", "B", "C", "D"]
    explanation: str = Field(min_length=5, max_length=2000)
    hint: Optional[str] = Field(default=None, max_length=500)
    difficulty: Literal["easy", "medium", "hard"]
    cognitive_level: Literal["recall", "understand", "apply", "analyze"]
    skill_tags: List[str] = Field(default_factory=list)
    misconception_tags: List[str] = Field(default_factory=list)
    question_style: Literal["direct", "scenario", "experiment", "data", "diagram"] = "direct"
    estimated_time_seconds: int = Field(default=60, ge=10, le=600)
    marks: int = Field(default=1, ge=1, le=10)
    source_url: str
    source_locator: Optional[str] = None
    source_question_id: Optional[str] = None
    question_family_key: Optional[str] = None
    variant_key: Optional[str] = None
    media: List[Dict[str, Any]] = Field(default_factory=list)
    option_media: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    render_config: Dict[str, Any] = Field(default_factory=dict)
    answer_spec: Dict[str, Any] = Field(default_factory=dict)
    generation_spec: Dict[str, Any] = Field(default_factory=dict)
    license_status: str = "review_required"
    review_status: Literal["draft", "review", "approved", "rejected"] = "draft"
    content_hash: Optional[str] = None

    @field_validator("question_text", "explanation")
    @classmethod
    def trim_text(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("skill_tags", "misconception_tags")
    @classmethod
    def normalize_tags(cls, values: List[str]) -> List[str]:
        return sorted({value.strip().lower() for value in values if value.strip()})

    @model_validator(mode="after")
    def validate_question(self) -> "QuestionCandidate":
        ids = [option.id for option in self.options]
        texts = [option.text.strip().casefold() for option in self.options]
        if ids != ["A", "B", "C", "D"] or len(set(ids)) != 4:
            raise ValueError("Options must contain exactly A, B, C, and D in order.")
        if len(set(texts)) != 4:
            raise ValueError("Options must have unique text.")
        if self.correct_option_id not in ids:
            raise ValueError("correct_option_id must match one of the options.")
        if not self.source_url.startswith(("https://", "manual://")):
            raise ValueError("source_url must use https:// or manual://.")
        if not self.content_hash:
            canonical = {
                "chapter_id": str(self.chapter_id),
                "concept_id": str(self.concept_id),
                "question_text": self.question_text,
                "options": [option.model_dump() for option in self.options],
                "correct_option_id": self.correct_option_id,
            }
            encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
            self.content_hash = hashlib.sha256(encoded).hexdigest()
        return self

    def to_db_row(
        self,
        curriculum_version_id: UUID,
        source_id: Optional[UUID] = None,
        ingestion_job_id: Optional[UUID] = None,
        status_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "curriculum_version_id": str(curriculum_version_id),
            "chapter_id": str(self.chapter_id),
            "concept_id": str(self.concept_id),
            "source_id": str(source_id) if source_id else None,
            "ingestion_job_id": str(ingestion_job_id) if ingestion_job_id else None,
            "source_url": self.source_url,
            "question_type": "mcq_single",
            "question_text": self.question_text,
            "options_json": [option.model_dump() for option in self.options],
            "correct_option_id": self.correct_option_id,
            "explanation": self.explanation,
            "hint": self.hint,
            "difficulty": self.difficulty,
            "cognitive_level": self.cognitive_level,
            "skill_tags": self.skill_tags,
            "misconception_tags": self.misconception_tags,
            "question_style": self.question_style,
            "estimated_time_seconds": self.estimated_time_seconds,
            "marks": self.marks,
            "question_family_key": self.question_family_key,
            "variant_key": self.variant_key,
            "media_json": self.media,
            "option_media_json": self.option_media,
            "render_config": self.render_config,
            "answer_spec": self.answer_spec,
            "generation_spec": self.generation_spec,
            "source_question_id": self.source_question_id,
            "source_locator": self.source_locator,
            "content_hash": self.content_hash,
            "quality_score": 1.0 if self.review_status == "approved" else None,
            "status": status_override or ("published" if self.review_status == "approved" else "review"),
        }


QuestionItemType = Literal[
    "mcq_single",
    "mcq_multi",
    "assertion_reason",
    "true_false",
    "fill_blank",
    "short_answer",
    "long_answer",
    "numerical",
    "case_study",
    "diagram_based",
    "matching",
    "other",
]


class QuestionItemCandidate(BaseModel):
    """Flexible source item retained before a renderer/evaluator exists."""

    chapter_id: UUID
    concept_id: UUID
    question_type: QuestionItemType
    question_text: str = Field(min_length=10, max_length=5000)
    options: List[Dict[str, Any]] = Field(default_factory=list)
    explanation: Optional[str] = Field(default=None, max_length=5000)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    cognitive_level: Literal["recall", "understand", "apply", "analyze"] = "understand"
    skill_tags: List[str] = Field(default_factory=list)
    question_style: Literal["direct", "scenario", "experiment", "data", "diagram"] = "direct"
    estimated_time_seconds: int = Field(default=60, ge=10, le=1200)
    marks: int = Field(default=1, ge=1, le=20)
    source_url: str
    source_locator: Optional[str] = None
    source_question_id: Optional[str] = None
    media: List[Dict[str, Any]] = Field(default_factory=list)
    option_media: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    render_config: Dict[str, Any] = Field(default_factory=dict)
    answer_spec: Dict[str, Any] = Field(default_factory=dict)
    generation_spec: Dict[str, Any] = Field(default_factory=dict)
    license_status: str = "review_required"
    review_status: Literal["draft", "review", "approved", "rejected"] = "review"
    content_hash: Optional[str] = None

    @field_validator("question_text")
    @classmethod
    def trim_question_text(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("explanation")
    @classmethod
    def trim_explanation(cls, value: Optional[str]) -> Optional[str]:
        return " ".join(value.split()) if value else value

    @field_validator("skill_tags")
    @classmethod
    def normalize_item_tags(cls, values: List[str]) -> List[str]:
        return sorted({value.strip().lower() for value in values if value.strip()})

    @model_validator(mode="after")
    def validate_item(self) -> "QuestionItemCandidate":
        if not self.source_url.startswith(("https://", "manual://")):
            raise ValueError("source_url must use https:// or manual://.")
        normalized_options: list[dict[str, Any]] = []
        option_ids: set[str] = set()
        for option in self.options:
            if not isinstance(option, dict):
                raise ValueError("Question options must be objects.")
            option_id = str(option.get("id", "")).strip()
            option_text = " ".join(str(option.get("text", "")).split())
            if not option_id or not option_text:
                raise ValueError("Every question option needs an id and text.")
            if option_id in option_ids:
                raise ValueError("Question option ids must be unique.")
            option_ids.add(option_id)
            normalized_options.append({**option, "id": option_id, "text": option_text})
        self.options = normalized_options
        if not self.content_hash:
            canonical = {
                "chapter_id": str(self.chapter_id),
                "question_type": self.question_type,
                "question_text": self.question_text,
                "options": self.options,
            }
            encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            self.content_hash = hashlib.sha256(encoded).hexdigest()
        return self

    @classmethod
    def from_mcq(cls, candidate: "QuestionCandidate", question_type: QuestionItemType = "mcq_single") -> "QuestionItemCandidate":
        answer_spec = dict(candidate.answer_spec)
        answer_spec.setdefault("correct_option_id", candidate.correct_option_id)
        return cls(
            chapter_id=candidate.chapter_id,
            concept_id=candidate.concept_id,
            question_type=question_type,
            question_text=candidate.question_text,
            options=[option.model_dump() for option in candidate.options],
            explanation=candidate.explanation,
            difficulty=candidate.difficulty,
            cognitive_level=candidate.cognitive_level,
            skill_tags=candidate.skill_tags,
            question_style=candidate.question_style,
            estimated_time_seconds=candidate.estimated_time_seconds,
            marks=candidate.marks,
            source_url=candidate.source_url,
            source_locator=candidate.source_locator,
            source_question_id=candidate.source_question_id,
            media=candidate.media,
            option_media=candidate.option_media,
            render_config=candidate.render_config,
            answer_spec=answer_spec,
            generation_spec=candidate.generation_spec,
            license_status=candidate.license_status,
            review_status=candidate.review_status,
        )

    def to_db_row(
        self,
        curriculum_version_id: UUID,
        source_id: Optional[UUID] = None,
        ingestion_job_id: Optional[UUID] = None,
        status_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "curriculum_version_id": str(curriculum_version_id),
            "chapter_id": str(self.chapter_id),
            "concept_id": str(self.concept_id),
            "ingestion_job_id": str(ingestion_job_id) if ingestion_job_id else None,
            "source_id": str(source_id) if source_id else None,
            "question_type": self.question_type,
            "question_text": self.question_text,
            "options_json": self.options,
            "explanation": self.explanation,
            "difficulty": self.difficulty,
            "cognitive_level": self.cognitive_level,
            "skill_tags": self.skill_tags,
            "question_style": self.question_style,
            "estimated_time_seconds": self.estimated_time_seconds,
            "marks": self.marks,
            "media_json": self.media,
            "option_media_json": self.option_media,
            "render_config": self.render_config,
            "answer_spec": self.answer_spec,
            "generation_spec": self.generation_spec,
            "source_url": self.source_url,
            "source_question_id": self.source_question_id,
            "source_locator": self.source_locator,
            "content_hash": self.content_hash,
            "license_status": self.license_status,
            "review_status": self.review_status,
            "status": status_override or ("published" if self.review_status == "approved" else "review"),
        }
