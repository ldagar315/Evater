"""Public, read-only sample-paper endpoints backing the SEO pages.

Unlike the practice routers this router carries NO auth dependency: the pages
must render for anonymous visitors and crawlers. Only published curriculum
data and published questions are returned, read with the server-only client.
The answer key is exposed on purpose - a sample paper's key is public content
(this is separate from the gen-flow key leak tracked on the integrity card).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

import threading
import time
from collections import deque

from fastapi import APIRouter, HTTPException, Query, Request, Response

from ..question_bank import _correct_summary, to_public_question
from ..supabase_client import create_supabase_service_client

logger = logging.getLogger(__name__)
router = APIRouter()

MIN_PUBLISHED_QUESTIONS = 15
PAPER_QUESTION_COUNT = 15
CACHE_CONTROL = "public, max-age=3600"

# --- Bounded abuse/cost protection (in-memory, per container) ---------------
# Anonymous read-only endpoints still cost a handful of PostgREST reads per
# hit, so responses are cached server-side and each client IP is capped.
CACHE_TTL_SECONDS = 3600
RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW_SECONDS = 60
_MAX_TRACKED_IPS = 5000

_cache_lock = threading.Lock()
_response_cache: Dict[str, Any] = {}
_rate_lock = threading.Lock()
_rate_windows: Dict[str, deque] = {}


def _cache_get(key: str) -> Any:
    now = time.monotonic()
    with _cache_lock:
        entry = _response_cache.get(key)
        if entry and now - entry[0] < CACHE_TTL_SECONDS:
            return entry[1]
        if entry:
            _response_cache.pop(key, None)
    return None


def _cache_set(key: str, value: Any) -> None:
    with _cache_lock:
        if len(_response_cache) > 500:
            _response_cache.clear()
        _response_cache[key] = (time.monotonic(), value)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _enforce_rate_limit(request: Request) -> None:
    ip = _client_ip(request)
    now = time.monotonic()
    with _rate_lock:
        window = _rate_windows.setdefault(ip, deque())
        while window and now - window[0] > RATE_LIMIT_WINDOW_SECONDS:
            window.popleft()
        if len(window) >= RATE_LIMIT_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please retry shortly.",
                headers={"Retry-After": str(RATE_LIMIT_WINDOW_SECONDS)},
            )
        if len(_rate_windows) > _MAX_TRACKED_IPS:
            _rate_windows.clear()
            window = _rate_windows.setdefault(ip, deque())
        window.append(now)

# URL subject slugs -> question-bank subject names. "maths" is accepted as an
# alias; the canonical slug emitted by the index is derived from the subject
# name itself (e.g. "mathematics").
SUBJECT_SLUGS = {
    "science": "Science",
    "maths": "Mathematics",
    "mathematics": "Mathematics",
}

SAMPLE_PAPER_COLUMNS = (
    "id,chapter_id,concept_id,question_type,question_text,options_json,explanation,hint,"
    "difficulty,cognitive_level,skill_tags,misconception_tags,question_style,"
    "estimated_time_seconds,marks,media_json,option_media_json,render_config,"
    "source_attribution_display,content_hash,correct_option_id,answer_spec"
)


def slugify(title: str) -> str:
    """kebab-case slug. The frontend routes, sitemap and PRERENDER_ROUTES use
    the same rule: lowercase, non-alphanumeric runs become a single dash."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def _load_published_chapters(client: Any, grade: int, subject_name: str) -> List[Dict[str, Any]]:
    curriculum_response = (
        client.table("curriculum_versions")
        .select("id")
        .eq("grade", grade)
        .eq("subject", subject_name)
        .eq("status", "published")
        .execute()
    )
    curriculum_ids = [str(row["id"]) for row in (curriculum_response.data or [])]
    if not curriculum_ids:
        return []
    chapter_response = (
        client.table("chapters")
        .select("id,sequence_number,title,description")
        .in_("curriculum_version_id", curriculum_ids)
        .eq("status", "published")
        .order("sequence_number")
        .execute()
    )
    return chapter_response.data or []


def _question_counts(client: Any, chapter_ids: List[str]) -> Dict[str, int]:
    if not chapter_ids:
        return {}
    rows = (
        client.table("question_bank")
        .select("chapter_id")
        .in_("chapter_id", chapter_ids)
        .eq("status", "published")
        .execute()
    ).data or []
    counts: Dict[str, int] = {}
    for row in rows:
        key = str(row["chapter_id"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def _pick_paper_questions(questions: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    """Deterministic mixed-type pick.

    Sort by content_hash, bucket by question_type, then round-robin one per
    bucket (mcq_single, true_false, numerical first) until `count` is reached.
    The same chapter always yields the same paper, so crawlers and users see
    identical content.
    """
    ordered = sorted(questions, key=lambda row: str(row.get("content_hash") or row["id"]))
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in ordered:
        buckets.setdefault(str(row.get("question_type") or "mcq_single"), []).append(row)
    preferred = ["mcq_single", "true_false", "numerical"]
    bucket_order = [t for t in preferred if t in buckets] + sorted(
        t for t in buckets if t not in preferred
    )
    picked: List[Dict[str, Any]] = []
    while len(picked) < count and any(buckets[t] for t in bucket_order):
        for bucket in bucket_order:
            if buckets[bucket]:
                picked.append(buckets[bucket].pop(0))
                if len(picked) >= count:
                    break
    return picked


@router.get("/sample-papers/index")
def sample_papers_index(request: Request, response: Response, grade: int = Query(8, ge=6, le=10)) -> Dict[str, Any]:
    """Chapters with published-question counts, for the hub/subject pages."""
    _enforce_rate_limit(request)
    cache_key = f"index:{grade}"
    cached = _cache_get(cache_key)
    if cached is not None:
        response.headers["Cache-Control"] = CACHE_CONTROL
        return cached
    client = create_supabase_service_client()
    curricula = (
        client.table("curriculum_versions")
        .select("id,subject")
        .eq("grade", grade)
        .eq("status", "published")
        .execute()
    ).data or []
    subjects: List[Dict[str, Any]] = []
    for subject_name in sorted({str(row["subject"]) for row in curricula}):
        chapters = _load_published_chapters(client, grade, subject_name)
        counts = _question_counts(client, [str(chapter["id"]) for chapter in chapters])
        subjects.append(
            {
                "subject": slugify(subject_name),
                "subject_name": subject_name,
                "chapters": [
                    {
                        "sequence_number": chapter["sequence_number"],
                        "title": chapter["title"],
                        "slug": slugify(str(chapter["title"])),
                        "published_count": counts.get(str(chapter["id"]), 0),
                    }
                    for chapter in chapters
                ],
            }
        )
    payload = {"grade": grade, "subjects": subjects}
    _cache_set(cache_key, payload)
    response.headers["Cache-Control"] = CACHE_CONTROL
    return payload


@router.get("/sample-papers/{subject}/{chapter_slug}")
def sample_paper(
    request: Request,
    response: Response,
    subject: str,
    chapter_slug: str,
    grade: int = Query(8, ge=6, le=10),
) -> Dict[str, Any]:
    """One deterministic sample paper: chapter info, questions, answer key."""
    _enforce_rate_limit(request)
    subject_name = SUBJECT_SLUGS.get(subject.lower())
    if not subject_name:
        raise HTTPException(status_code=404, detail="Unknown subject.")

    cache_key = f"paper:{grade}:{subject}:{chapter_slug}"
    cached = _cache_get(cache_key)
    if cached is not None:
        response.headers["Cache-Control"] = CACHE_CONTROL
        return cached

    client = create_supabase_service_client()
    chapters = _load_published_chapters(client, grade, subject_name)
    chapter = next((c for c in chapters if slugify(str(c["title"])) == chapter_slug), None)
    if chapter is None:
        raise HTTPException(status_code=404, detail="Sample paper not found.")

    questions = (
        client.table("question_bank")
        .select(SAMPLE_PAPER_COLUMNS)
        .eq("chapter_id", str(chapter["id"]))
        .eq("status", "published")
        .execute()
    ).data or []
    if len(questions) < MIN_PUBLISHED_QUESTIONS:
        raise HTTPException(
            status_code=404,
            detail="Sample paper not available yet for this chapter.",
        )

    picked = _pick_paper_questions(questions, PAPER_QUESTION_COUNT)
    payload_questions: List[Dict[str, Any]] = []
    for index, row in enumerate(picked):
        public = to_public_question(row)
        answer_summary = _correct_summary(
            {"options": public["options"], "correct_option_id": row.get("correct_option_id")},
            row.get("answer_spec") or {},
        )
        # Anonymous payload is trimmed to exactly the fields the pages render
        # (position, id, question_type, question_text, options, answer_summary).
        payload_questions.append(
            {
                "position": index + 1,
                "id": public["id"],
                "question_type": public["question_type"],
                "question_text": public["question_text"],
                "options": public["options"],
                "answer_summary": answer_summary,
            }
        )

    payload = {
        "grade": grade,
        "subject": subject,
        "subject_name": subject_name,
        "chapter": {
            "sequence_number": chapter["sequence_number"],
            "title": chapter["title"],
            "slug": chapter_slug,
            "description": chapter.get("description"),
        },
        "question_count": len(payload_questions),
        "questions": payload_questions,
    }
    _cache_set(cache_key, payload)
    response.headers["Cache-Control"] = CACHE_CONTROL
    return payload
