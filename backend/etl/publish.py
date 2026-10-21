from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
from uuid import UUID

from supabase import Client

from .models import QuestionCandidate


# Prod question_bank does not yet carry source_tier /
# source_attribution_display (migration 20261020143000 is not applied). Strip
# them until the migration lands; provenance still flows via source_url,
# source_question_id, source_locator, and content_sources.
_PENDING_MIGRATION_COLUMNS = ("source_tier", "source_attribution_display")


def build_publish_rows(
    candidates: Iterable[QuestionCandidate],
    curriculum_version_id: UUID,
    source_id: Optional[UUID] = None,
    ingestion_job_id: Optional[UUID] = None,
    status_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return [
        {
            key: value
            for key, value in candidate.to_db_row(
                curriculum_version_id=curriculum_version_id,
                source_id=source_id,
                ingestion_job_id=ingestion_job_id,
                status_override=status_override,
            ).items()
            if key not in _PENDING_MIGRATION_COLUMNS
        }
        for candidate in candidates
    ]


def publish_candidates(
    client: Client,
    candidates: List[QuestionCandidate],
    curriculum_version_id: UUID,
    source_id: Optional[UUID] = None,
    ingestion_job_id: Optional[UUID] = None,
    status_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows = build_publish_rows(
        candidates=candidates,
        curriculum_version_id=curriculum_version_id,
        source_id=source_id,
        ingestion_job_id=ingestion_job_id,
        status_override=status_override,
    )
    if not rows:
        return []
    response = (
        client.table("question_bank")
        .upsert(rows, on_conflict="chapter_id,content_hash")
        .execute()
    )
    return response.data or []
