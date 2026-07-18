"""Validate and idempotently publish the complete Class 8 Science catalog."""

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

from .class8_science_catalog import CHAPTERS, CURRICULUM_VERSION_ID, ChapterSpec
from .generate_class8_science_packs import TOPICS, generate
from .models import QuestionCandidate, QuestionItemCandidate
from .publish import publish_candidates
from .seed_first_chapter import (
    CONCEPT_ROWS as FIRST_CONCEPT_ROWS,
    INGESTION_JOB_ID as FIRST_INGESTION_JOB_ID,
    PACK_PATH as FIRST_PACK_PATH,
    SOURCE_ID as FIRST_SOURCE_ID,
)
from .validate import (
    QuestionPackError,
    load_candidates,
    load_question_items,
    validate_publishable_pack,
    validate_question_items,
    validate_scraped_pack,
)


CHAPTER_ONE = CHAPTERS[0]


def source_id_for(spec: ChapterSpec, source_mode: str = "generated") -> UUID:
    if spec.sequence_number == 1:
        if source_mode == "generated":
            return UUID(FIRST_SOURCE_ID)
        return uuid5(CURRICULUM_VERSION_ID, f"source:{source_mode}:{spec.sequence_number}")
    if source_mode == "generated":
        return uuid5(CURRICULUM_VERSION_ID, f"source:{spec.sequence_number}")
    return uuid5(CURRICULUM_VERSION_ID, f"source:{source_mode}:{spec.sequence_number}")


def ingestion_job_id_for(spec: ChapterSpec, source_mode: str = "generated") -> UUID:
    if spec.sequence_number == 1:
        if source_mode == "generated":
            return UUID(FIRST_INGESTION_JOB_ID)
        return uuid5(CURRICULUM_VERSION_ID, f"ingestion:{source_mode}:{spec.sequence_number}")
    if source_mode == "generated":
        return uuid5(CURRICULUM_VERSION_ID, f"ingestion:{spec.sequence_number}")
    return uuid5(CURRICULUM_VERSION_ID, f"ingestion:{source_mode}:{spec.sequence_number}")


def _source_hash(spec: ChapterSpec, candidates: list[QuestionCandidate]) -> str:
    digest = hashlib.sha256()
    digest.update(spec.source_url.encode("utf-8"))
    digest.update(b"\n")
    digest.update("\n".join(question.content_hash or "" for question in candidates).encode("utf-8"))
    return digest.hexdigest()


def _concept_rows(spec: ChapterSpec, candidates: list[QuestionCandidate]) -> list[dict[str, Any]]:
    if spec.sequence_number == 1:
        return [
            {
                **row,
                "chapter_id": str(spec.id),
                "status": "published",
            }
            for row in FIRST_CONCEPT_ROWS
        ]

    rows: list[dict[str, Any]] = []
    for sequence_number, seed in enumerate(TOPICS[spec.sequence_number], start=1):
        rows.append(
            {
                "id": str(uuid5(spec.id, f"concept:{seed.slug}")),
                "chapter_id": str(spec.id),
                "sequence_number": sequence_number,
                "title": seed.title,
                "slug": seed.slug,
                "learning_outcome": seed.definition,
                "status": "published",
            }
        )
    return rows


def build_seed_rows(
    spec: ChapterSpec,
    candidates: list[QuestionCandidate],
    *,
    source_mode: str = "generated",
    source_metadata: Optional[dict[str, Any]] = None,
    archive_item_count: int = 0,
) -> dict[str, Any]:
    source_id = source_id_for(spec, source_mode)
    ingestion_job_id = ingestion_job_id_for(spec, source_mode)
    now = datetime.now(timezone.utc).isoformat()
    source_metadata = source_metadata or {}
    source_records = source_metadata.get("sources", [])
    source_urls = sorted({record.get("url") for record in source_records if record.get("url")})
    candidate_is_scraped = source_mode == "scraped"
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
            "source_url": CHAPTER_ONE.source_url,
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
        "concepts": _concept_rows(spec, candidates),
        "source": {
            "id": str(source_id),
            "source_url": spec.source_url,
            "source_type": "official_pdf",
            "publisher": "National Council of Educational Research and Training",
            "license": "NCERT copyrighted source; external question sources require permission/license review",
            "attribution": f"Curriculum anchor: NCERT Grade 8 Science, Curiosity, Chapter {spec.sequence_number}; scraped question URLs are retained per question.",
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
                "pack": f"class8_science_chapter_{spec.sequence_number:02d}",
                "source_mode": source_mode,
                "generated_questions_are_original": not candidate_is_scraped,
                "source_license_review_required": True,
                "source_urls": source_urls,
                "source_manifest_records": len(source_records),
                "archive_item_count": archive_item_count,
                "generator": "class8-science-topic-template-v1" if source_mode == "generated" and spec.sequence_number > 1 else "first-chapter-pilot-v1" if source_mode == "generated" else None,
                "review_state": "pending_subject_and_license_review" if candidate_is_scraped else "approved_original_content",
            },
            "finished_at": now,
        },
    }


def seed_remote(
    chapter_packs: list[tuple[ChapterSpec, list[QuestionCandidate]]],
    *,
    archive_items_by_chapter: Optional[dict[int, list[QuestionItemCandidate]]] = None,
    source_mode: str = "generated",
    source_metadata_by_chapter: Optional[dict[int, dict[str, Any]]] = None,
    stage_publish: bool = False,
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
            source_mode=source_mode,
            source_metadata=(source_metadata_by_chapter or {}).get(spec.sequence_number),
            archive_item_count=len(archive_items),
        )
        client.table("curriculum_versions").upsert(
            [rows["curriculum_version"]],
            on_conflict="board,grade,subject,language,academic_year,version_label",
        ).execute()
        client.table("chapters").upsert([rows["chapter"]], on_conflict="curriculum_version_id,slug").execute()
        client.table("concepts").upsert(rows["concepts"], on_conflict="chapter_id,slug").execute()
        client.table("content_sources").upsert(
            [rows["source"]],
            on_conflict="id",
        ).execute()
        client.table("ingestion_jobs").upsert([rows["ingestion_job"]], on_conflict="id").execute()
        archive_rows = [
            item.to_db_row(
                curriculum_version_id=CURRICULUM_VERSION_ID,
                source_id=source_id_for(spec, source_mode),
                ingestion_job_id=ingestion_job_id_for(spec, source_mode),
                # Non-MCQ items are retained for editorial review. They are
                # deliberately not exposed through the learner question API.
                status_override="review" if source_mode == "scraped" else ("published" if stage_publish else None),
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
            source_id=source_id_for(spec, source_mode),
            ingestion_job_id=ingestion_job_id_for(spec, source_mode),
            status_override="published" if stage_publish else None,
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


def _pack_path(spec: ChapterSpec) -> Path:
    return FIRST_PACK_PATH if spec.sequence_number == 1 else spec.pack_path


def load_and_validate(spec: ChapterSpec, source_packs_dir: Optional[Path] = None) -> list[QuestionCandidate]:
    path = (
        source_packs_dir / f"class8_science_scraped_chapter_{spec.sequence_number:02d}.json"
        if source_packs_dir
        else _pack_path(spec)
    )
    candidates = load_candidates(path)
    if any(candidate.chapter_id != spec.id for candidate in candidates):
        raise QuestionPackError(f"Pack for Chapter {spec.sequence_number} contains a mismatched chapter_id.")
    if source_packs_dir:
        validate_scraped_pack(candidates)
    else:
        validate_publishable_pack(candidates)
    return candidates


def load_archive_items(
    spec: ChapterSpec,
    candidates: list[QuestionCandidate],
    source_packs_dir: Optional[Path] = None,
) -> list[QuestionItemCandidate]:
    if not source_packs_dir:
        return [QuestionItemCandidate.from_mcq(candidate) for candidate in candidates]
    path = source_packs_dir / f"class8_science_scraped_chapter_{spec.sequence_number:02d}.json"
    items = load_question_items(path)
    if any(item.chapter_id != spec.id for item in items):
        raise QuestionPackError(f"Item pack for Chapter {spec.sequence_number} contains a mismatched chapter_id.")
    validate_question_items(items)
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and seed Class 8 Science chapter packs.")
    parser.add_argument("--chapters", default="1-13", help="Comma-separated chapter numbers or ranges, for example 2-13")
    parser.add_argument("--generate", action="store_true", help="Regenerate packs for selected chapters after Chapter 1")
    parser.add_argument("--publish", action="store_true", help="Write the catalog and questions to Supabase")
    parser.add_argument("--source-packs-dir", type=Path, help="Use source-scraped candidate packs instead of generated packs.")
    parser.add_argument("--source-manifest", type=Path, help="Manifest emitted by etl.scrape_class8_science.")
    parser.add_argument("--replace-existing", action="store_true", help="Retire existing published questions in selected chapters before publishing.")
    parser.add_argument(
        "--acknowledge-source-review",
        action="store_true",
        help="Acknowledge that the NCERT source/license review remains required",
    )
    args = parser.parse_args()
    load_dotenv()

    try:
        specs = _parse_chapters(args.chapters)
        if args.generate and args.source_packs_dir:
            raise QuestionPackError("--generate cannot be combined with --source-packs-dir.")
        source_mode = "scraped" if args.source_packs_dir else "generated"
        if args.generate:
            generate(spec for spec in specs if spec.sequence_number > 1)
        packs = [(spec, load_and_validate(spec, args.source_packs_dir)) for spec in specs]
        archive_items_by_chapter = {
            spec.sequence_number: load_archive_items(spec, candidates, args.source_packs_dir)
            for spec, candidates in packs
        }
        validation = [
            {
                "sequence_number": spec.sequence_number,
                "title": spec.title,
                "pack": str((args.source_packs_dir / f"class8_science_scraped_chapter_{spec.sequence_number:02d}.json") if args.source_packs_dir else _pack_path(spec)),
                "count": len(candidates),
                "archive_item_count": len(archive_items_by_chapter[spec.sequence_number]),
                "archive_question_types": dict(Counter(item.question_type for item in archive_items_by_chapter[spec.sequence_number])),
            }
            for spec, candidates in packs
        ]
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
            source_mode=source_mode,
            source_metadata_by_chapter=source_metadata_by_chapter,
            stage_publish=source_mode == "scraped",
            replace_existing=args.replace_existing,
        )}, indent=2))
        return 0
    except (QuestionPackError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
