"""Idempotently seed the first Class 8 Science question-bank chapter.

The default command is a local dry run. Use ``--publish`` only after the
question pack has passed validation and the Supabase service-role key is
available in the environment (or Modal secret).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from .generate_first_chapter_pack import (
    CHAPTER_ID,
    CONCEPTS,
    CURRICULUM_VERSION_ID,
    SOURCE_URL,
)
from .models import QuestionCandidate
from .publish import publish_candidates
from .validate import QuestionPackError, load_candidates, validate_publishable_pack

PACK_PATH = Path(__file__).parent / "data" / "class8_science_chapter1.json"
SOURCE_ID = "44444444-4444-4444-8444-444444444444"
INGESTION_JOB_ID = "55555555-5555-4555-8555-555555555555"

CONCEPT_ROWS = [
    {
        "id": str(CONCEPTS["questions"]),
        "sequence_number": 1,
        "title": "Asking focused questions",
        "slug": "asking-focused-questions",
        "learning_outcome": "Frame an investigation question with a deliberate condition and an observable outcome.",
    },
    {
        "id": str(CONCEPTS["variables"]),
        "sequence_number": 2,
        "title": "Variables and fair tests",
        "slug": "variables-and-fair-tests",
        "learning_outcome": "Identify changed, measured, and controlled conditions in a fair test.",
    },
    {
        "id": str(CONCEPTS["observation"]),
        "sequence_number": 3,
        "title": "Observation versus explanation",
        "slug": "observation-versus-explanation",
        "learning_outcome": "Separate recorded observations from explanations about why they occurred.",
    },
    {
        "id": str(CONCEPTS["measurement"]),
        "sequence_number": 4,
        "title": "Measurement and records",
        "slug": "measurement-and-records",
        "learning_outcome": "Choose measurable outcomes and record evidence in a dated, organised way.",
    },
    {
        "id": str(CONCEPTS["evidence"]),
        "sequence_number": 5,
        "title": "Evidence and repeatability",
        "slug": "evidence-and-repeatability",
        "learning_outcome": "Use repeated, recorded evidence to evaluate a testable explanation.",
    },
]


def _source_hash(candidates: List[QuestionCandidate]) -> str:
    digest = hashlib.sha256()
    digest.update(SOURCE_URL.encode("utf-8"))
    digest.update(b"\n")
    digest.update("\n".join(question.content_hash or "" for question in candidates).encode("utf-8"))
    return digest.hexdigest()


def build_seed_rows(candidates: List[QuestionCandidate]) -> Dict[str, Any]:
    source_hash = _source_hash(candidates)
    return {
        "curriculum_version": {
            "id": str(CURRICULUM_VERSION_ID),
            "board": "NCERT/CBSE",
            "grade": 8,
            "subject": "Science",
            "language": "en",
            "academic_year": "2026-27",
            "version_label": "curiosity-2026-27",
            "textbook_name": "Curiosity — Textbook of Science for Grade 8",
            "status": "published",
            "source_url": SOURCE_URL,
            "license_status": "review_required",
        },
        "chapter": {
            "id": str(CHAPTER_ID),
            "curriculum_version_id": str(CURRICULUM_VERSION_ID),
            "sequence_number": 1,
            "title": "Exploring the Investigative World of Science",
            "slug": "exploring-the-investigative-world-of-science",
            "description": "Original practice questions for investigation skills: questions, variables, observations, measurement, and evidence.",
            "status": "published",
        },
        "concepts": [
            {
                "id": row["id"],
                "chapter_id": str(CHAPTER_ID),
                "sequence_number": row["sequence_number"],
                "title": row["title"],
                "slug": row["slug"],
                "learning_outcome": row["learning_outcome"],
                "status": "published",
            }
            for row in CONCEPT_ROWS
        ],
        "source": {
            "id": SOURCE_ID,
            "source_url": SOURCE_URL,
            "source_type": "official_pdf",
            "publisher": "National Council of Educational Research and Training",
            "license": "NCERT copyrighted source; original question text; review commercial-use rights",
            "attribution": "Source reference: NCERT Grade 8 Science, Curiosity, Chapter 1.",
            "content_hash": source_hash,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "status": "fetched",
        },
        "ingestion_job": {
            "id": INGESTION_JOB_ID,
            "source_id": SOURCE_ID,
            "curriculum_version_id": str(CURRICULUM_VERSION_ID),
            "job_type": "publish",
            "status": "succeeded",
            "input_count": len(candidates),
            "output_count": len(candidates),
            "error_count": 0,
            "metadata": {
                "pack": "class8_science_chapter1",
                "generated_questions_are_original": True,
                "source_license_review_required": True,
            },
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def seed_remote(candidates: List[QuestionCandidate]) -> Dict[str, Any]:
    from app.supabase_client import create_supabase_service_client

    client = create_supabase_service_client()
    rows = build_seed_rows(candidates)
    client.table("curriculum_versions").upsert(
        [rows["curriculum_version"]],
        on_conflict="board,grade,subject,language,academic_year,version_label",
    ).execute()
    client.table("chapters").upsert(
        [rows["chapter"]],
        on_conflict="curriculum_version_id,slug",
    ).execute()
    client.table("concepts").upsert(
        rows["concepts"],
        on_conflict="chapter_id,slug",
    ).execute()
    client.table("content_sources").upsert(
        [rows["source"]],
        on_conflict="id",
    ).execute()
    client.table("ingestion_jobs").upsert(
        [rows["ingestion_job"]],
        on_conflict="id",
    ).execute()
    published = publish_candidates(
        client=client,
        candidates=candidates,
        curriculum_version_id=CURRICULUM_VERSION_ID,
        source_id=SOURCE_ID,
        ingestion_job_id=INGESTION_JOB_ID,
    )
    return {"published_questions": len(published), "chapter_id": str(CHAPTER_ID)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the Class 8 Science Chapter 1 question bank.")
    parser.add_argument("--pack", default=str(PACK_PATH))
    parser.add_argument("--publish", action="store_true", help="Write curriculum and questions to Supabase.")
    parser.add_argument(
        "--acknowledge-source-review",
        action="store_true",
        help="Acknowledge that the NCERT source/license review is still required.",
    )
    args = parser.parse_args()
    load_dotenv()

    try:
        candidates = load_candidates(args.pack)
        report = validate_publishable_pack(candidates)
        print(json.dumps({"validation": report}, indent=2))
        if not args.publish:
            print("Dry run only. Use --publish --acknowledge-source-review to seed Supabase.")
            return 0
        if not args.acknowledge_source_review:
            raise QuestionPackError(
                "Source/license review is required. Re-run with --acknowledge-source-review after checking rights."
            )
        print(json.dumps({"publish": seed_remote(candidates)}, indent=2))
        return 0
    except (QuestionPackError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
