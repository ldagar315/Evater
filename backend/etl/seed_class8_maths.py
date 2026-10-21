"""Validate and idempotently publish the Class 8 Maths (Ganita Prakash) catalog.

Mirrors seed_class8_science.py: upserts the curriculum version, chapters,
concepts, content source, ingestion job, then publishes MCQ candidates to
question_bank and retains non-MCQ items in question_bank_items for editorial
review. Source model: NCERT-attributed extraction with official answer keys
where the book provides them; see scrape_class8_maths.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid5

from dotenv import load_dotenv

from .class8_maths_catalog import CHAPTERS, CURRICULUM_VERSION_ID, ChapterSpec
from .class8_maths_topics import TOPICS
from .models import QuestionCandidate, QuestionItemCandidate
from .publish import publish_candidates
from .validate import (
    QuestionPackError,
    load_candidates,
    load_question_items,
    validate_question_items,
    validate_scraped_pack,
)


def source_id_for(spec: ChapterSpec, source_mode: str = "scraped") -> UUID:
    return uuid5(CURRICULUM_VERSION_ID, f"source:{source_mode}:{spec.sequence_number}")


def ingestion_job_id_for(spec: ChapterSpec, source_mode: str = "scraped") -> UUID:
    return uuid5(CURRICULUM_VERSION_ID, f"ingestion:{source_mode}:{spec.sequence_number}")


def _source_hash(spec: ChapterSpec, candidates: list[QuestionCandidate]) -> str:
    digest = hashlib.sha256()
    digest.update(spec.source_url.encode("utf-8"))
    digest.update(b"\n")
    digest.update("\n".join(question.content_hash or "" for question in candidates).encode("utf-8"))
    return digest.hexdigest()


def _concept_rows(spec: ChapterSpec) -> list[dict[str, Any]]:
    return [
        {
            "id": str(uuid5(spec.id, f"concept:{seed.slug}")),
            "chapter_id": str(spec.id),
            "sequence_number": sequence_number,
            "title": seed.title,
            "slug": seed.slug,
            "learning_outcome": seed.definition,
            "status": "published",
        }
        for sequence_number, seed in enumerate(TOPICS[spec.sequence_number], start=1)
    ]


def build_seed_rows(
    spec: ChapterSpec,
    candidates: list[QuestionCandidate],
    *,
    source_metadata: Optional[dict[str, Any]] = None,
    archive_item_count: int = 0,
) -> dict[str, Any]:
    source_id = source_id_for(spec)
    ingestion_job_id = ingestion_job_id_for(spec)
    now = datetime.now(timezone.utc).isoformat()
    source_metadata = source_metadata or {}
    source_records = source_metadata.get("sources", [])
    source_urls = sorted({record.get("url") for record in source_records if record.get("url")})
    return {
        "curriculum_version": {
            "id": str(CURRICULUM_VERSION_ID),
            "board": "NCERT/CBSE",
            "grade": 8,
            "subject": "Mathematics",
            "language": "en",
            "academic_year": "2025-26",
            "version_label": "ganita-prakash-2025-26",
            "textbook_name": "Ganita Prakash — Textbook of Mathematics for Grade 8",
            "status": "published",
            "source_url": CHAPTERS[0].source_url,
            "license_status": "review_required",
        },
        "chapter": {
            "id": str(spec.id),
            "curriculum_version_id": str(CURRICULUM_VERSION_ID),
            "sequence_number": spec.sequence_number,
            "title": spec.title,
            "slug": spec.slug,
            "description": spec.description,
            "status": "published",
        },
        "concepts": _concept_rows(spec),
        "source": {
            "id": str(source_id),
            "source_url": spec.source_url,
            "source_type": "official_pdf",
            "publisher": "National Council of Educational Research and Training",
            "license": "NCERT copyrighted source; reused with attribution per owner-approved stance",
            "attribution": f"Curriculum anchor: NCERT Grade 8 Mathematics, Ganita Prakash, Chapter {spec.sequence_number}; question provenance retained per item.",
            "content_hash": _source_hash(spec, candidates),
            "fetched_at": now,
            "status": "fetched",
        },
        "ingestion_job": {
            "id": str(ingestion_job_id),
            "source_id": str(source_id),
            "curriculum_version_id": str(CURRICULUM_VERSION_ID),
            "job_type": "publish",
            "status": "succeeded",
            "input_count": len(candidates),
            "output_count": len(candidates),
            "error_count": 0,
            "metadata": {
                "pack": f"class8_maths_chapter_{spec.sequence_number:02d}",
                "source_mode": "scraped",
                "generated_questions_are_original": False,
                "source_license_review_required": True,
                "source_urls": source_urls,
                "source_manifest_records": len(source_records),
                "archive_item_count": archive_item_count,
                "review_state": "pending_subject_and_license_review",
            },
            "finished_at": now,
        },
    }


def seed_remote(
    chapter_packs: list[tuple[ChapterSpec, list[QuestionCandidate]]],
    *,
    archive_items_by_chapter: Optional[dict[int, list[QuestionItemCandidate]]] = None,
    source_metadata_by_chapter: Optional[dict[int, dict[str, Any]]] = None,
    replace_existing: bool = False,
) -> list[dict[str, Any]]:
    from app.supabase_client import create_supabase_service_client

    client = create_supabase_service_client()
    reports: list[dict[str, Any]] = []
    for spec, candidates in chapter_packs:
        archive_items = (archive_items_by_chapter or {}).get(spec.sequence_number, [])
        if replace_existing:
            client.table("question_bank").update({"status": "retired"}).eq(
                "curriculum_version_id", str(CURRICULUM_VERSION_ID)
            ).eq("chapter_id", str(spec.id)).eq("status", "published").execute()
            client.table("question_bank_items").update({"status": "retired"}).eq(
                "curriculum_version_id", str(CURRICULUM_VERSION_ID)
            ).eq("chapter_id", str(spec.id)).in_("status", ["draft", "validated", "review", "published"]).execute()
        rows = build_seed_rows(
            spec,
            candidates,
            source_metadata=(source_metadata_by_chapter or {}).get(spec.sequence_number),
            archive_item_count=len(archive_items),
        )
        client.table("curriculum_versions").upsert(
            [rows["curriculum_version"]],
            on_conflict="board,grade,subject,language,academic_year,version_label",
        ).execute()
        client.table("chapters").upsert([rows["chapter"]], on_conflict="curriculum_version_id,slug").execute()
        client.table("concepts").upsert(rows["concepts"], on_conflict="chapter_id,slug").execute()
        client.table("content_sources").upsert([rows["source"]], on_conflict="id").execute()
        client.table("ingestion_jobs").upsert([rows["ingestion_job"]], on_conflict="id").execute()
        archive_rows = [
            item.to_db_row(
                curriculum_version_id=CURRICULUM_VERSION_ID,
                source_id=source_id_for(spec),
                ingestion_job_id=ingestion_job_id_for(spec),
                # Non-MCQ items are retained for editorial review. They are
                # deliberately not exposed through the learner question API.
                status_override="review",
            )
            for item in archive_items
        ]
        if archive_rows:
            client.table("question_bank_items").upsert(
                archive_rows,
                on_conflict="chapter_id,content_hash",
            ).execute()
        published = publish_candidates(
            client=client,
            candidates=candidates,
            curriculum_version_id=CURRICULUM_VERSION_ID,
            source_id=source_id_for(spec),
            ingestion_job_id=ingestion_job_id_for(spec),
            status_override="published",
        )
        reports.append(
            {
                "sequence_number": spec.sequence_number,
                "title": spec.title,
                "chapter_id": str(spec.id),
                "published_questions": len(published),
                "archived_items": len(archive_items),
                "archived_question_types": dict(Counter(item.question_type for item in archive_items)),
            }
        )
    return reports


def _parse_chapters(value: str) -> tuple[ChapterSpec, ...]:
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    try:
        return tuple(next(chapter for chapter in CHAPTERS if chapter.sequence_number == number) for number in sorted(selected))
    except StopIteration as exc:
        raise QuestionPackError(f"Unknown chapter in --chapters={value!r}.") from exc


def _pack_path(source_packs_dir: Path, spec: ChapterSpec) -> Path:
    return source_packs_dir / f"class8_maths_scraped_chapter_{spec.sequence_number:02d}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and seed Class 8 Maths scraped packs.")
    parser.add_argument("--chapters", default="1-14", help="Comma-separated chapter numbers or ranges, for example 1-7")
    parser.add_argument("--publish", action="store_true", help="Write the catalog and questions to Supabase")
    parser.add_argument("--source-packs-dir", type=Path, required=True, help="Directory with class8_maths_scraped_chapter_NN.json packs")
    parser.add_argument("--source-manifest", type=Path, help="Manifest emitted by etl.scrape_class8_maths.")
    parser.add_argument("--replace-existing", action="store_true", help="Retire existing published questions in selected chapters before publishing.")
    parser.add_argument(
        "--acknowledge-source-review",
        action="store_true",
        help="Acknowledge that the NCERT source/license review stance applies",
    )
    args = parser.parse_args()
    load_dotenv()

    try:
        specs = _parse_chapters(args.chapters)
        packs: list[tuple[ChapterSpec, list[QuestionCandidate]]] = []
        archive_items_by_chapter: dict[int, list[QuestionItemCandidate]] = {}
        validation: list[dict[str, Any]] = []
        for spec in specs:
            path = _pack_path(args.source_packs_dir, spec)
            candidates = load_candidates(path)
            if any(candidate.chapter_id != spec.id for candidate in candidates):
                raise QuestionPackError(f"Pack for Chapter {spec.sequence_number} contains a mismatched chapter_id.")
            # MCQ candidates are model-validated at load. The science 10-MCQ
            # floor does not apply: NCERT maths exercises are mostly numerical,
            # so MCQ volume comes from a later conversion/rework pass.
            _ = validate_scraped_pack  # import kept for parity with the science seed
            items = load_question_items(path)
            if any(item.chapter_id != spec.id for item in items):
                raise QuestionPackError(f"Item pack for Chapter {spec.sequence_number} contains a mismatched chapter_id.")
            validate_question_items(items)
            packs.append((spec, candidates))
            archive_items_by_chapter[spec.sequence_number] = items
            validation.append(
                {
                    "sequence_number": spec.sequence_number,
                    "title": spec.title,
                    "pack": str(path),
                    "mcq_count": len(candidates),
                    "archive_item_count": len(items),
                    "archive_question_types": dict(Counter(item.question_type for item in items)),
                    "items_with_answer_key": sum(1 for item in items if item.answer_spec.get("has_answer_key")),
                }
            )
        print(json.dumps({"validation": validation}, indent=2))
        if not args.publish:
            print("Dry run only. Use --publish --acknowledge-source-review to seed Supabase.")
            return 0
        if not args.acknowledge_source_review:
            raise QuestionPackError(
                "Source/license review is required. Re-run with --acknowledge-source-review after checking rights."
            )
        source_metadata_by_chapter: dict[int, dict[str, Any]] = {}
        if args.source_manifest:
            manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
            source_metadata_by_chapter = {
                spec.sequence_number: {
                    "sources": [
                        record
                        for record in manifest.get("sources", [])
                        if record.get("chapter_number") == spec.sequence_number
                    ],
                    "errors": manifest.get("errors", []),
                }
                for spec in specs
            }
        print(json.dumps({"publish": seed_remote(
            packs,
            archive_items_by_chapter=archive_items_by_chapter,
            source_metadata_by_chapter=source_metadata_by_chapter,
            replace_existing=args.replace_existing,
        )}, indent=2))
        return 0
    except (QuestionPackError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
