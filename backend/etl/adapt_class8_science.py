"""Retarget previous-chapter question structures to the current syllabus.

This is a conservative fallback for thin source coverage. It does not copy the
previous question wording or answer; it reuses the previous item's assessment
shape (difficulty, cognitive level, style, and template where available) and
builds a new question from the current chapter's concept manifest. Every output
carries the previous source question's provenance for later inspection.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

from .class8_science_catalog import ChapterSpec
from .generate_class8_science_packs import TOPICS, make_question
from .models import QuestionCandidate


def _template_for(source: QuestionCandidate, offset: int) -> int:
    raw_template = source.generation_spec.get("template")
    if isinstance(raw_template, int) and 0 <= raw_template <= 9:
        return raw_template
    return offset % 10


def build_adapted_candidates(
    target_spec: ChapterSpec,
    previous_spec: ChapterSpec,
    previous_candidates: Iterable[QuestionCandidate],
    *,
    current_count: int,
    minimum_count: int,
    existing_hashes: set[str] | None = None,
) -> list[QuestionCandidate]:
    """Create enough target-chapter MCQs to reach a configurable source floor."""

    target_topics = TOPICS.get(target_spec.sequence_number)
    source_candidates = list(previous_candidates)
    if not target_topics or not source_candidates or current_count >= minimum_count:
        return []

    needed = minimum_count - current_count
    seen = set(existing_hashes or set())
    adapted: list[QuestionCandidate] = []
    attempts = 0
    max_attempts = max(needed * 20, 100)

    while len(adapted) < needed and attempts < max_attempts:
        source = source_candidates[attempts % len(source_candidates)]
        topic_index = (attempts // 10) % len(target_topics)
        target_topic = target_topics[topic_index]
        template = _template_for(source, attempts)
        base = make_question(
            target_spec,
            current_count + attempts + 1,
            topic_index,
            target_topic,
            template,
        )
        source_hash = source.content_hash or hashlib.sha256(source.question_text.encode()).hexdigest()
        adaptation_hash = hashlib.sha256(
            f"{target_spec.id}:{source_hash}:{target_topic.slug}:{template}".encode("utf-8")
        ).hexdigest()[:16]
        candidate = base.model_copy(
            update={
                "difficulty": source.difficulty,
                "cognitive_level": source.cognitive_level,
                "question_style": source.question_style,
                "estimated_time_seconds": source.estimated_time_seconds,
                "marks": source.marks,
                "source_locator": (
                    f"Adapted for Chapter {target_spec.sequence_number} from previous "
                    f"Chapter {previous_spec.sequence_number} question {source.source_question_id}"
                ),
                "source_question_id": (
                    f"adapted:class8-science:ch{target_spec.sequence_number:02d}:"
                    f"from-ch{previous_spec.sequence_number:02d}:{adaptation_hash}"
                ),
                "question_family_key": f"class8-science-{target_spec.sequence_number:02d}:{target_topic.slug}:adapted",
                "variant_key": f"previous-chapter-{previous_spec.sequence_number:02d}-template-{template + 1:02d}",
                "answer_spec": {
                    **base.answer_spec,
                    "adaptation": "previous_chapter_structure_to_current_topic",
                    "adapted_from_source_question_id": source.source_question_id,
                    "adapted_from_source_url": source.source_url,
                    "adapted_from_source_hash": source_hash,
                    "review_required": True,
                },
                "generation_spec": {
                    "generator": "class8-science-previous-chapter-adaptation-v1",
                    "adaptation_mode": "structure_and_assessment_shape_only",
                    "target_chapter_sequence": target_spec.sequence_number,
                    "previous_chapter_sequence": previous_spec.sequence_number,
                    "previous_source_question_id": source.source_question_id,
                    "previous_source_url": source.source_url,
                    "previous_source_hash": source_hash,
                    "target_topic_slug": target_topic.slug,
                    "template": template,
                    "review_state": "pending_subject_and_license_review",
                },
                "license_status": "adapted_review_required",
                "review_status": "review",
            }
        )
        if candidate.content_hash not in seen:
            adapted.append(candidate)
            seen.add(candidate.content_hash or "")
        attempts += 1

    return adapted
