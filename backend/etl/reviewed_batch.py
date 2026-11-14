"""Validate a reviewed proposal and prepare REVIEW rows, without network or publication.

This does not replace validate_publishable_pack or infer owner approval. Publishing
requires a separately reviewed path after the owner decision and live DB preflight.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from uuid import UUID

from .models import QuestionCandidate


class ReviewedBatchError(ValueError):
    pass


def load_reviewed_batch(directory: str | Path) -> tuple[dict, list[dict]]:
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    raw = (directory / 'questions.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['questions_sha256']:
        raise ReviewedBatchError('Question proposal changed since the reviewed manifest.')
    questions = json.loads(raw)['questions']
    ids = [q['draft_id'] for q in questions]
    if len(ids) != manifest['expected_count'] or sorted(ids) != manifest['question_ids'] or len(set(ids)) != len(ids):
        raise ReviewedBatchError('Only the exact reviewed first-release IDs are permitted.')
    if Counter(q['suggested_difficulty'] for q in questions) != manifest['difficulty_counts']:
        raise ReviewedBatchError('Difficulty labels differ from the reviewed proposal.')
    counts = Counter(q['concept_id'] for q in questions)
    if dict(counts) != {key: value['expected_count'] for key, value in manifest['concepts'].items()}:
        raise ReviewedBatchError('Concept counts differ from the reviewed proposal.')
    for q in questions:
        spec = manifest['concepts'][q['concept_id']]
        if q['chapter_id'] != spec['chapter_id']:
            raise ReviewedBatchError('Concept/chapter mapping differs from the reviewed proposal.')
        if q['release_recommendation'] != 'recommend_for_publication_decision':
            raise ReviewedBatchError('Reserve or unselected question cannot enter the first release.')
    for name, digest in manifest['guides'].items():
        if Path(name).name != name or hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ReviewedBatchError('Guide copy differs from the reviewed manifest.')
    return manifest, questions


def prepare_review_rows(directory: str | Path) -> list[dict]:
    manifest, questions = load_reviewed_batch(directory)
    rows = []
    for q in questions:
        candidate = QuestionCandidate.model_validate({
            **q,
            'difficulty': q['suggested_difficulty'],
            'cognitive_level': 'apply' if q['suggested_difficulty'] == 'medium' else 'understand',
            'source_url': 'manual://evater-original/' + q['draft_id'],
            'source_locator': q['curriculum_mapping']['source_url'] + ' | ' + q['curriculum_mapping']['source_section'],
            'source_question_id': q['draft_id'],
            'source_tier': 'manual',
            'source_attribution': None,
            'review_status': 'review',
            'license_status': 'review_required',
        })
        version_id = UUID(manifest['concepts'][q['concept_id']]['curriculum_version_id'])
        row = candidate.to_db_row(version_id, status_override='review')
        if row['status'] != 'review' or row['quality_score'] is not None:
            raise ReviewedBatchError('Preparation must not publish or certify a row.')
        rows.append(row)
    if len({row['content_hash'] for row in rows}) != len(rows):
        raise ReviewedBatchError('Duplicate content hashes in reviewed proposal.')
    return rows
