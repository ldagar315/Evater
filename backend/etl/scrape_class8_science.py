"""Fetch and normalize source-backed Class 8 Science questions.

The scraper is intentionally a provenance-preserving ingestion step. Every
question keeps the original URL, question locator, answer-key source, and
downloaded-file hash attached. When source coverage is thin, a configurable
previous-chapter adaptation fallback creates new target-syllabus MCQs from the
previous chapter's assessment shape; those items carry explicit adaptation
provenance. The local stage publishes these questions for product testing, and
learners can flag any questionable item by question ID for internal follow-up.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import UUID, uuid5

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from .class8_science_catalog import CHAPTERS, CURRICULUM_VERSION_ID, ChapterSpec
from .adapt_class8_science import build_adapted_candidates
from .generate_class8_science_packs import TOPICS
from .generate_first_chapter_pack import CONCEPTS as FIRST_CONCEPTS
from .models import CandidateOption, QuestionCandidate, QuestionItemCandidate, QuestionItemType


TIWARI_URL_ROOT = "https://www.tiwariacademy.in/ncert-solutions-class-8-science-curiosity-chapter"
PAPER_URL_ROOT = "https://kumarsir34.wordpress.com/wp-content/uploads"


@dataclass(frozen=True)
class PaperSource:
    chapter_number: int
    paper_number: int
    upload_folder: str
    slug: str

    @property
    def qp_url(self) -> str:
        return (
            f"{PAPER_URL_ROOT}/{self.upload_folder}/"
            f"science-class-viii-chapter-{self.chapter_number:02d}-{self.slug}-"
            f"practice-paper-{self.paper_number:02d}-2025-qp.pdf"
        )

    @property
    def answer_url(self) -> str:
        return self.qp_url.replace("-qp.pdf", "-answers.pdf")


PAPER_SOURCES = (
    *(PaperSource(2, number, "2025/08", "the-invisible-living-world-beyond-our-naked-eye") for number in range(1, 4)),
    *(PaperSource(3, number, "2025/08", "health-the-ultimate-treasure") for number in range(4, 7)),
    *(PaperSource(4, number, "2025/08", "electricity-magnetic-and-heating-effects") for number in range(7, 10)),
    *(PaperSource(5, number, "2025/10", "exploring-forces") for number in range(10, 13)),
)


def tiwari_url(chapter_number: int) -> str:
    return f"{TIWARI_URL_ROOT}-{chapter_number}/"


# Known source-quality exception found during the agentic source review. The
# public page's answer key marks option A for this stem even though option C is
# the definition shown by its own options. Keep the raw download for audit, but
# do not put the contradictory candidate into stage or production packs.
EXCLUDED_SOURCE_QUESTIONS = {
    "tiwari:class8-science:ch08:q01": "answer key conflicts with the option text; requires source-editor correction",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split()).strip()


def _bounded(value: str, limit: int) -> str:
    value = _normalize(value)
    if len(value) <= limit:
        return value
    shortened = value[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return shortened + "…"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_filename(url: str) -> str:
    name = url.rstrip("/").rsplit("/", 1)[-1] or "index"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name)


def _fetch(url: str, *, verify_tls: bool = True) -> tuple[bytes, str]:
    headers = {
        "User-Agent": "Evater curriculum research bot/1.0 (+local-stage; contact project owner)",
        "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8",
    }
    try:
        response = httpx.get(url, headers=headers, follow_redirects=True, timeout=90, verify=verify_tls)
    except httpx.TransportError:
        if verify_tls:
            response = httpx.get(url, headers=headers, follow_redirects=True, timeout=90, verify=False)
        else:
            raise
    response.raise_for_status()
    return response.content, response.headers.get("content-type", "")


def _fetch_with_cache(
    url: str,
    target: Path,
    *,
    verify_tls: bool = True,
) -> tuple[bytes, str, bool, str | None]:
    """Refresh a source, falling back to a prior verified download on failure."""

    try:
        content, content_type = _fetch(url, verify_tls=verify_tls)
        return content, content_type, False, None
    except Exception as exc:
        if not target.exists():
            raise
        content_type = "application/pdf" if target.suffix.casefold() == ".pdf" else "text/html"
        return target.read_bytes(), content_type, True, str(exc)


def _record_download(
    *,
    url: str,
    kind: str,
    content: bytes,
    content_type: str,
    target: Path,
    publisher: str,
    license_status: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    record: dict[str, Any] = {
        "url": url,
        "kind": kind,
        "publisher": publisher,
        "license_status": license_status,
        "content_type": content_type,
        "bytes": len(content),
        "sha256": _sha256_bytes(content),
        "downloaded_at": _now(),
        "path": str(target),
        "status": "fetched",
    }
    if metadata:
        record.update(metadata)
    return record


def _first_concept_id(spec: ChapterSpec) -> UUID:
    if spec.sequence_number == 1:
        return FIRST_CONCEPTS["questions"]
    return uuid5(spec.id, f"concept:{TOPICS[spec.sequence_number][0].slug}")


def _concept_id_for(spec: ChapterSpec, question_text: str, options: Iterable[str]) -> tuple[UUID, str]:
    corpus = f"{question_text} {' '.join(options)}".casefold()
    if spec.sequence_number == 1:
        concepts = (
            ("questions", ("question", "investigation")),
            ("variables", ("variable", "factor", "condition")),
            ("observation", ("observe", "observation", "evidence")),
            ("measurement", ("measure", "measurement", "unit")),
            ("evidence", ("repeat", "record", "data")),
        )
        best = max(concepts, key=lambda item: sum(term in corpus for term in item[1]))
        return FIRST_CONCEPTS[best[0]], best[0]

    seeds = TOPICS[spec.sequence_number]
    best_seed = max(
        seeds,
        key=lambda seed: sum(
            term in corpus
            for term in (seed.slug.replace("-", " "), seed.title.casefold(), *seed.slug.split("-"))
        ),
    )
    return uuid5(spec.id, f"concept:{best_seed.slug}"), best_seed.slug


def _make_candidate(
    *,
    spec: ChapterSpec,
    question_text: str,
    options: list[tuple[str, str]],
    correct_option_id: str,
    explanation: str,
    source_url: str,
    source_locator: str,
    source_question_id: str,
    source_kind: str,
    answer_source_url: str | None = None,
) -> QuestionCandidate:
    option_values = [text for _, text in options]
    concept_id, concept_slug = _concept_id_for(spec, question_text, option_values)
    question_style = "scenario" if any(token in question_text.casefold() for token in ("student", "farmer", "case", "observe", "experiment")) else "direct"
    return QuestionCandidate(
        chapter_id=spec.id,
        concept_id=concept_id,
        question_text=_bounded(question_text, 1000),
        options=[CandidateOption(id=option_id, text=_bounded(text, 500)) for option_id, text in options],
        correct_option_id=correct_option_id,  # type: ignore[arg-type]
        explanation=_bounded(explanation or f"Source answer key: option {correct_option_id}.", 2000),
        hint=None,
        # Sources generally do not publish a difficulty tag. Keep the value
        # explicit and conservative until an editor or assessment model rates it.
        difficulty="medium",
        cognitive_level="understand",
        skill_tags=["source_mcq", concept_slug],
        misconception_tags=[],
        question_style=question_style,  # type: ignore[arg-type]
        estimated_time_seconds=60,
        marks=1,
        source_url=source_url,
        source_locator=source_locator,
        source_question_id=source_question_id,
        question_family_key=f"curiosity:{spec.sequence_number:02d}:{concept_slug}:{hashlib.sha1(_normalize(question_text).encode()).hexdigest()[:12]}",
        variant_key=source_kind,
        answer_spec={
            "type": "mcq_single",
            "source_answer": correct_option_id,
            "verification": "pending_subject_review",
            "answer_source_url": answer_source_url or source_url,
        },
        generation_spec={
            "ingestion_method": "scraped",
            "source_kind": source_kind,
            "retrieved_at": _now(),
            "verification_state": "pending_subject_and_license_review",
        },
        license_status="review_required",
        review_status="review",
    )


def _has_diagram_reference(question_text: str) -> bool:
    lowered = question_text.casefold()
    return any(
        marker in lowered
        for marker in ("fig.", "fig ", "figure", "diagram", "graph", "shown below", "draw the", "set-up")
    )


def _looks_numerical(question_text: str) -> bool:
    lowered = question_text.casefold()
    if re.search(r"\b(calculate|compute|density|mass|volume|pressure|speed|angle of|how much|how many|what is the value|by how much|approximately how many years)\b", lowered):
        return True
    return bool(re.search(r"\d", lowered) and re.search(r"\b(value|measurement|measure|distance|time|weight)\b", lowered))


def _inline_options(question_text: str) -> tuple[str, list[dict[str, str]]]:
    """Extract textbook-style (i)–(iv) or (a)–(d) options when present."""

    markers = list(re.finditer(r"(?<!\w)\(([ivx]+|[a-d])\)\s+", question_text, flags=re.I))
    if len(markers) < 4:
        return question_text, []
    labels = [marker.group(1).casefold() for marker in markers[:4]]
    if labels not in (["i", "ii", "iii", "iv"], ["a", "b", "c", "d"]):
        return question_text, []
    stem = _normalize(question_text[: markers[0].start()])
    options: list[dict[str, str]] = []
    for index, marker in enumerate(markers[:4]):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(question_text)
        option_text = _normalize(question_text[marker.end() : end])
        if not option_text:
            return question_text, []
        options.append({"id": chr(ord("A") + index), "text": option_text})
    return stem, options


def _classify_item_type(question_text: str, section: str | None = None) -> QuestionItemType:
    lowered = question_text.casefold()
    if "assertion" in lowered and "reason" in lowered:
        return "assertion_reason"
    if (
        "true [t]" in lowered
        or "true or false" in lowered
        or "true [t] or false [f]" in lowered
        or re.search(r"\btrue\s*/\s*false\b", lowered)
        or "state whether" in lowered
    ):
        return "true_false"
    if "match the" in lowered or "match each" in lowered or "column i" in lowered:
        return "matching"
    if "fill in the blank" in lowered or "fill in the blanks" in lowered or "____" in question_text:
        return "fill_blank"
    _, inline_options = _inline_options(question_text)
    if len(inline_options) == 4:
        return "diagram_based" if _has_diagram_reference(question_text) else "mcq_single"
    if _has_diagram_reference(question_text):
        return "diagram_based"
    if _looks_numerical(question_text):
        return "numerical"
    if section == "E" or "case study" in lowered or ("(i)" in question_text and len(question_text) > 350):
        return "case_study"
    if section in {"D"} or len(question_text) > 260 or any(
        lowered.startswith(prefix) for prefix in ("explain", "describe", "design", "justify", "analyse", "analyze")
    ):
        return "long_answer"
    if section in {"B", "C"}:
        return "short_answer"
    return "short_answer"


def _media_reference(source_url: str, source_locator: str, page_number: int | None = None) -> list[dict[str, Any]]:
    media: dict[str, Any] = {
        "kind": "diagram_reference",
        "source_url": source_url,
        "source_locator": source_locator,
        "rights_status": "review_required",
    }
    if page_number is not None:
        media["page_number"] = page_number
    return [media]


def _make_item(
    *,
    spec: ChapterSpec,
    question_text: str,
    question_type: QuestionItemType,
    source_url: str,
    source_locator: str,
    source_question_id: str,
    source_kind: str,
    options: list[dict[str, Any]] | None = None,
    answer_spec: dict[str, Any] | None = None,
    explanation: str | None = None,
    marks: int = 1,
    media: list[dict[str, Any]] | None = None,
) -> QuestionItemCandidate:
    option_values = [str(option.get("text", "")) for option in (options or [])]
    concept_id, concept_slug = _concept_id_for(spec, question_text, option_values)
    style = "diagram" if question_type == "diagram_based" else "scenario" if question_type == "case_study" else "direct"
    return QuestionItemCandidate(
        chapter_id=spec.id,
        concept_id=concept_id,
        question_type=question_type,
        question_text=_bounded(question_text, 5000),
        options=options or [],
        explanation=_bounded(explanation, 5000) if explanation else None,
        difficulty="medium",
        cognitive_level="understand",
        skill_tags=["source_item", concept_slug, question_type],
        question_style=style,  # type: ignore[arg-type]
        estimated_time_seconds=120 if question_type in {"long_answer", "case_study", "diagram_based"} else 60,
        marks=max(1, min(marks, 20)),
        source_url=source_url,
        source_locator=source_locator,
        source_question_id=source_question_id,
        media=media or [],
        answer_spec={
            **(answer_spec or {}),
            "source_kind": source_kind,
            "verification": "pending_subject_and_license_review",
        },
        license_status="review_required",
        review_status="review",
    )


def parse_tiwari_html(html: str, spec: ChapterSpec, source_url: str) -> list[QuestionCandidate]:
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[QuestionCandidate] = []
    for block in soup.select("div.sseo_faqcont"):
        title = block.select_one(".sseo_faqtitle h4")
        title_match = re.match(r"Q\s*(\d+)\.\s*(.+)", _normalize(title.get_text(" ", strip=True)) if title else "")
        if not title_match:
            continue
        question_number = int(title_match.group(1))
        question_text = title_match.group(2)
        options: list[tuple[str, str]] = []
        for option_node in block.select(".sseo_answeroption"):
            marker = option_node.select_one("span")
            option_id_match = re.search(r"([A-D])", marker.get_text(" ", strip=True).upper() if marker else "")
            if not option_id_match:
                continue
            option_text = option_node.get_text(" ", strip=True)
            if marker:
                option_text = option_text.replace(marker.get_text(" ", strip=True), "", 1)
            options.append((option_id_match.group(1), _normalize(option_text)))
        answer_node = block.select_one(".sseo_correct_option")
        answer_match = re.search(r"(?:Answer|Ans(?:wer)?)\s*:\s*Option\s*([A-D])", answer_node.get_text(" ", strip=True) if answer_node else "", re.I)
        explanation_node = block.select_one(".sseo_answer_detail")
        explanation = explanation_node.get_text(" ", strip=True) if explanation_node else ""
        explanation = re.sub(r"^Explanation\s*:\s*", "", explanation, flags=re.I)
        if len(options) != 4 or not answer_match:
            continue
        options.sort(key=lambda item: item[0])
        source_question_id = f"tiwari:class8-science:ch{spec.sequence_number:02d}:q{question_number:02d}"
        if source_question_id in EXCLUDED_SOURCE_QUESTIONS:
            continue
        try:
            candidates.append(
                _make_candidate(
                    spec=spec,
                    question_text=question_text,
                    options=options,
                    correct_option_id=answer_match.group(1).upper(),
                    explanation=explanation,
                    source_url=source_url,
                    source_locator=f"HTML MCQ block, Question {question_number}",
                    source_question_id=source_question_id,
                    source_kind="tiwari_academy_html",
                )
            )
        except ValueError:
            # Keep the crawl moving when a public page has malformed options;
            # the skipped block remains discoverable in the downloaded HTML.
            continue
    return candidates


def parse_tiwari_items(html: str, spec: ChapterSpec, source_url: str) -> list[QuestionItemCandidate]:
    items: list[QuestionItemCandidate] = []
    for candidate in parse_tiwari_html(html, spec, source_url):
        item_type = _classify_item_type(candidate.question_text)
        if item_type in {"short_answer", "other"}:
            item_type = "mcq_single"
        items.append(QuestionItemCandidate.from_mcq(candidate, item_type))
    return items


def _section_a(text: str) -> str:
    start_match = re.search(r"SECTION\s*[-–—]\s*A", text, flags=re.I)
    if not start_match:
        raise ValueError("Could not find Section A in question paper.")
    end_match = re.search(r"SECTION\s*[-–—]\s*B", text[start_match.end() :], flags=re.I)
    end = start_match.end() + end_match.start() if end_match else len(text)
    return text[start_match.end() : end]


def _numbered_blocks(text: str, maximum: int = 10) -> dict[int, str]:
    starts = list(re.finditer(r"(?m)^\s*(\d{1,2})\.\s+", text))
    blocks: dict[int, str] = {}
    for index, start in enumerate(starts):
        number = int(start.group(1))
        if number > maximum:
            continue
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        blocks[number] = text[start.end() : end]
    return blocks


def _parse_pdf_question_section(section_text: str) -> dict[int, tuple[str, list[tuple[str, str]]]]:
    assertion_options = [
        ("A", "Both A and R are true and R is the correct explanation of A."),
        ("B", "Both A and R are true but R is not the correct explanation of A."),
        ("C", "A is true but R is false."),
        ("D", "A is false but R is true."),
    ]
    parsed: dict[int, tuple[str, list[tuple[str, str]]]] = {}
    for question_number, block in _numbered_blocks(section_text).items():
        option_markers = list(re.finditer(r"\(([A-Da-d])\)\s*", block))
        if len(option_markers) < 4 and not ("Assertion" in block and "Reason" in block):
            continue
        if len(option_markers) < 4:
            parsed[question_number] = (_normalize(block), assertion_options)
            continue
        # Question 8 often ends immediately before the shared assertion/reason
        # instructions, so its block can contain eight markers. The first four
        # still belong to Question 8; Questions 9–10 use the shared choices.
        option_markers = option_markers[:4]
        stem = _normalize(block[: option_markers[0].start()])
        options: list[tuple[str, str]] = []
        for index, marker in enumerate(option_markers[:4]):
            end = option_markers[index + 1].start() if index + 1 < len(option_markers) else len(block)
            option_text = _normalize(block[marker.end() : end])
            options.append((marker.group(1).upper(), option_text))
        parsed[question_number] = (stem, options)
    return parsed


def _parse_pdf_questions(text: str) -> dict[int, tuple[str, list[tuple[str, str]]]]:
    return _parse_pdf_question_section(_section_a(text))


def _parse_pdf_answers(text: str) -> dict[int, tuple[str, str]]:
    answers: dict[int, tuple[str, str]] = {}
    for question_number, block in _numbered_blocks(text, maximum=10).items():
        answer_match = re.search(r"\bAns(?:wer)?\.\s*\(([A-Da-d])\)", block)
        if not answer_match:
            continue
        explanation = block[answer_match.end() :]
        explanation = re.sub(r"^\s*[^\n]*\n", "", explanation, count=1)
        answers[question_number] = (answer_match.group(1).upper(), _normalize(explanation))
    return answers


def _section_blocks(text: str) -> dict[str, str]:
    markers = list(re.finditer(r"SECTION\s*[-–—]\s*([A-E])", text, flags=re.I))
    sections: dict[str, str] = {}
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        sections[marker.group(1).upper()] = text[marker.end() : end]
    return sections


def _clean_source_prompt(value: str) -> str:
    cleaned = re.sub(r"Prepared by:.*?Page\s*-\s*\d+\s*-", " ", value, flags=re.I | re.S)
    cleaned = re.sub(r"Chapter\s+\d+\.indd.*", " ", cleaned, flags=re.I | re.S)
    for marker in (
        "Prepare some questions based on your learnings so far",
        "Reflect on the questions framed by your friends and try to answer",
        "Discover , design, and debate",
        "Discover, design, and debate",
    ):
        cleaned = cleaned.split(marker, 1)[0]
    return _normalize(cleaned)


def _answer_block_text(answer_blocks: dict[int, str], question_number: int) -> str | None:
    block = answer_blocks.get(question_number)
    if not block:
        return None
    answer_marker = re.search(r"\bAns(?:wer)?\.\s*", block, flags=re.I)
    return _clean_source_prompt(block[answer_marker.end() :] if answer_marker else block)


def _page_number_for(question_pages: list[str], question_number: int, start_page: int = 0) -> int:
    return next(
        (
            index + 1
            for index, page_text in enumerate(question_pages[start_page:], start=start_page)
            if re.search(rf"(?m)^\s*{question_number}\.\s+", page_text)
        ),
        start_page + 1,
    )


def parse_paper_items(
    question_pdf: Path,
    answer_pdf: Path,
    spec: ChapterSpec,
    question_url: str,
    answer_url: str,
    paper_number: int,
) -> list[QuestionItemCandidate]:
    question_reader = PdfReader(str(question_pdf))
    question_pages = [page.extract_text() or "" for page in question_reader.pages]
    question_text = "\n".join(question_pages)
    answer_text = "\n".join(page.extract_text() or "" for page in PdfReader(str(answer_pdf)).pages)
    answer_blocks = _numbered_blocks(answer_text, maximum=200)
    items: list[QuestionItemCandidate] = []

    for section, section_text in _section_blocks(question_text).items():
        if section == "A":
            section_questions = _parse_pdf_question_section(section_text)
            answer_choices = _parse_pdf_answers(answer_text)
            for question_number, (stem, options) in sorted(section_questions.items()):
                answer = answer_choices.get(question_number)
                if not answer:
                    continue
                question_type = _classify_item_type(stem)
                if question_type in {"short_answer", "other"}:
                    question_type = "mcq_single"
                option_rows = [{"id": option_id, "text": option_text} for option_id, option_text in options]
                locator = f"PDF page {_page_number_for(question_pages, question_number)}, Section A, Question {question_number}"
                items.append(
                    _make_item(
                        spec=spec,
                        question_text=stem,
                        question_type=question_type if question_type != "other" else "mcq_single",
                        options=option_rows,
                        answer_spec={
                            "correct_option_id": answer[0],
                            "answer_source_url": answer_url,
                            "source_answer_text": answer[1],
                        },
                        explanation=answer[1],
                        source_url=question_url,
                        source_locator=locator,
                        source_question_id=f"kumar-swamy:class8-science:ch{spec.sequence_number:02d}:paper{paper_number:02d}:q{question_number:02d}",
                        source_kind="kv_practice_paper_pdf",
                        media=_media_reference(question_url, locator, _page_number_for(question_pages, question_number)) if _has_diagram_reference(stem) else [],
                    )
                )
            continue

        for question_number, raw_block in _numbered_blocks(section_text, maximum=200).items():
            prompt = _clean_source_prompt(raw_block)
            if len(prompt) < 10:
                continue
            question_type = _classify_item_type(prompt, section)
            page_number = _page_number_for(question_pages, question_number)
            locator = f"PDF page {page_number}, Section {section}, Question {question_number}"
            answer_text_for_item = _answer_block_text(answer_blocks, question_number)
            answer_spec: dict[str, Any] = {
                "answer_source_url": answer_url,
                "source_answer_text": answer_text_for_item,
                "has_answer_key": bool(answer_text_for_item),
                "subquestion_count": len(re.findall(r"\([ivx]+\)|\([a-z]\)", prompt, flags=re.I)),
            }
            items.append(
                _make_item(
                    spec=spec,
                    question_text=prompt,
                    question_type=question_type,
                    answer_spec=answer_spec,
                    source_url=question_url,
                    source_locator=locator,
                    source_question_id=f"kumar-swamy:class8-science:ch{spec.sequence_number:02d}:paper{paper_number:02d}:q{question_number:02d}",
                    source_kind="kv_practice_paper_pdf",
                    marks=2 if section == "B" else 3 if section == "C" else 4 if section == "E" else 5,
                    media=_media_reference(question_url, locator, page_number) if _has_diagram_reference(prompt) else [],
                )
            )
    return items


def parse_official_chapter_items(content: bytes, spec: ChapterSpec) -> list[QuestionItemCandidate]:
    reader = PdfReader(io.BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages)
    keep_match = re.search(r"Keep\s+the\s+curiosity\s+alive", text, flags=re.I)
    if keep_match:
        section = text[keep_match.end() :]
        section = re.split(r"Discover\s*,?\s*design\s*,?\s*and\s*debate", section, flags=re.I)[0]
        section_start_page = next(
            (index for index, page_text in enumerate(pages) if re.search(r"Keep\s+the\s+curiosity\s+alive", page_text, flags=re.I)),
            0,
        )
        section_label = "Keep the curiosity alive"
        source_id_prefix = "keep"
    else:
        # Chapter 1 is an investigation primer. It has no end-of-chapter
        # exercise block; its four opening "Probe and ponder" prompts are the
        # source-backed questions to retain instead.
        first_page = pages[0] if pages else ""
        probe_match = re.search(r"Dear Young Scientists,", first_page, flags=re.I)
        section = first_page[probe_match.end() :] if probe_match else first_page
        section = re.split(r"Probe\s+and\s+ponder", section, flags=re.I)[0]
        first_question = re.search(r"(?m)^\s*(?:Why|Are|What|How|Is|Can|Would)\b", section, flags=re.I)
        section = section[first_question.start() :] if first_question else ""
        section_start_page = 0
        section_label = "Probe and ponder"
        source_id_prefix = "probe"
        question_blocks = {index: match.group(0) for index, match in enumerate(re.finditer(r".+?\?", section, flags=re.S), start=1)}
        items: list[QuestionItemCandidate] = []
        for question_number, raw_block in question_blocks.items():
            prompt = _clean_source_prompt(raw_block)
            if len(prompt) < 10:
                continue
            locator = f"Official NCERT PDF page 1, {section_label}, Prompt {question_number}"
            items.append(
                _make_item(
                    spec=spec,
                    question_text=prompt,
                    question_type="short_answer",
                    source_url=spec.source_url,
                    source_locator=locator,
                    source_question_id=f"ncert:class8-science:ch{spec.sequence_number:02d}:{source_id_prefix}:q{question_number:02d}",
                    source_kind="official_ncert_pdf",
                    answer_spec={"has_answer_key": False},
                )
            )
        return items
    items: list[QuestionItemCandidate] = []
    for question_number, raw_block in _numbered_blocks(section, maximum=200).items():
        prompt = _clean_source_prompt(raw_block)
        if len(prompt) < 10:
            continue
        stem, options = _inline_options(prompt)
        # Classify from the full prompt so inline textbook options can turn
        # into a real MCQ record; store the cleaned stem separately.
        question_type = _classify_item_type(prompt)
        page_number = _page_number_for(pages, question_number, section_start_page)
        locator = f"Official NCERT PDF page {page_number}, {section_label}, Question {question_number}"
        items.append(
            _make_item(
                spec=spec,
                question_text=stem,
                question_type=question_type,
                options=options,
                source_url=spec.source_url,
                source_locator=locator,
                source_question_id=f"ncert:class8-science:ch{spec.sequence_number:02d}:{source_id_prefix}:q{question_number:02d}",
                source_kind="official_ncert_pdf",
                answer_spec={
                    "has_answer_key": False,
                    "option_answer_status": "not_provided" if options else None,
                    "subquestion_count": len(re.findall(r"\([ivx]+\)|\([a-z]\)", prompt, flags=re.I)),
                },
                media=_media_reference(spec.source_url, locator, page_number) if _has_diagram_reference(prompt) else [],
            )
        )
    return items


def parse_paper_pair(
    question_pdf: Path,
    answer_pdf: Path,
    spec: ChapterSpec,
    question_url: str,
    answer_url: str,
    paper_number: int,
) -> list[QuestionCandidate]:
    question_reader = PdfReader(str(question_pdf))
    question_pages = [page.extract_text() or "" for page in question_reader.pages]
    question_text = "\n".join(question_pages)
    answer_text = "\n".join(page.extract_text() or "" for page in PdfReader(str(answer_pdf)).pages)
    questions = _parse_pdf_questions(question_text)
    answers = _parse_pdf_answers(answer_text)
    candidates: list[QuestionCandidate] = []
    for question_number, (stem, options) in sorted(questions.items()):
        answer = answers.get(question_number)
        if not answer or len(options) != 4:
            continue
        page_number = next(
            (index + 1 for index, page_text in enumerate(question_pages) if re.search(rf"(?m)^\s*{question_number}\.\s+", page_text)),
            1,
        )
        candidates.append(
            _make_candidate(
                spec=spec,
                question_text=stem,
                options=options,
                correct_option_id=answer[0],
                explanation=answer[1],
                source_url=question_url,
                source_locator=f"PDF page {page_number}, Section A, Question {question_number}",
                source_question_id=f"kumar-swamy:class8-science:ch{spec.sequence_number:02d}:paper{paper_number:02d}:q{question_number:02d}",
                source_kind="kv_practice_paper_pdf",
                answer_source_url=answer_url,
            )
        )
    return candidates


def _chapter_paths(output_dir: Path, spec: ChapterSpec) -> tuple[Path, Path]:
    return (
        output_dir / "packs" / f"class8_science_scraped_chapter_{spec.sequence_number:02d}.json",
        output_dir / "downloads" / f"official-hecu1{spec.sequence_number:02d}.pdf",
    )


def _deduplicate_items(items: Iterable[QuestionItemCandidate]) -> list[QuestionItemCandidate]:
    """Deduplicate prompt/options while merging answer and source provenance."""

    by_hash: dict[str, QuestionItemCandidate] = {}
    for item in items:
        if not item.content_hash:
            continue
        existing = by_hash.get(item.content_hash)
        if existing is None:
            by_hash[item.content_hash] = item
            continue

        answer_spec = dict(existing.answer_spec)
        for key, value in item.answer_spec.items():
            if key == "has_answer_key":
                answer_spec[key] = bool(answer_spec.get(key)) or bool(value)
            elif value not in (None, "") and answer_spec.get(key) in (None, ""):
                answer_spec[key] = value

        source_records = list(answer_spec.get("additional_source_records", []))
        source_record = {
            "source_url": item.source_url,
            "source_question_id": item.source_question_id,
            "source_locator": item.source_locator,
        }
        if source_record not in source_records:
            source_records.append(source_record)
        if source_records:
            answer_spec["additional_source_records"] = source_records

        by_hash[item.content_hash] = existing.model_copy(
            update={
                "answer_spec": answer_spec,
                "explanation": existing.explanation or item.explanation,
                "media": existing.media or item.media,
                "option_media": existing.option_media or item.option_media,
            }
        )
    return list(by_hash.values())


def scrape(
    chapters: Iterable[ChapterSpec],
    output_dir: str | Path,
    *,
    include_papers: bool = True,
    verify_tls: bool = True,
    adapt_to_minimum: int | None = 60,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    selected = tuple(chapters)
    selected_by_number = {chapter.sequence_number: chapter for chapter in selected}
    manifest: list[dict[str, Any]] = []
    errors: list[str] = []
    candidates_by_chapter: dict[int, list[QuestionCandidate]] = {chapter.sequence_number: [] for chapter in selected}
    items_by_chapter: dict[int, list[QuestionItemCandidate]] = {chapter.sequence_number: [] for chapter in selected}

    for spec in selected:
        _, official_path = _chapter_paths(output_path, spec)
        try:
            content, content_type, cached, fetch_error = _fetch_with_cache(
                spec.source_url,
                official_path,
                verify_tls=verify_tls,
            )
            official_items = parse_official_chapter_items(content, spec)
            items_by_chapter[spec.sequence_number].extend(official_items)
            reader = PdfReader(io.BytesIO(content))
            manifest.append(
                _record_download(
                    url=spec.source_url,
                    kind="official_curriculum_pdf",
                    content=content,
                    content_type=content_type,
                    target=official_path,
                    publisher="National Council of Educational Research and Training",
                    license_status="copyright_review_required",
                    metadata={
                        "chapter_number": spec.sequence_number,
                        "page_count": len(reader.pages),
                        "extracted_text_characters": sum(len(page.extract_text() or "") for page in reader.pages),
                        "extracted_item_count": len(official_items),
                        "retrieval_mode": "cached_after_refresh_failure" if cached else "fresh",
                        "refresh_error": fetch_error,
                    },
                )
            )
        except Exception as exc:  # the question source below is independently retriable
            errors.append(f"official chapter {spec.sequence_number}: {exc}")

        source_url = tiwari_url(spec.sequence_number)
        try:
            target = output_path / "downloads" / f"tiwari-chapter-{spec.sequence_number:02d}.html"
            content, content_type, cached, fetch_error = _fetch_with_cache(
                source_url,
                target,
                verify_tls=verify_tls,
            )
            html_candidates = parse_tiwari_html(content.decode("utf-8", "ignore"), spec, source_url)
            html_items = parse_tiwari_items(content.decode("utf-8", "ignore"), spec, source_url)
            items_by_chapter[spec.sequence_number].extend(html_items)
            manifest.append(
                _record_download(
                    url=source_url,
                    kind="chapter_mcq_html",
                    content=content,
                    content_type=content_type,
                    target=target,
                    publisher="Tiwari Academy",
                    license_status="copyright_review_required",
                    metadata={
                        "chapter_number": spec.sequence_number,
                        "question_count": len(html_candidates),
                        "extracted_item_count": len(html_items),
                        "excluded_source_question_ids": [
                            question_id
                            for question_id in EXCLUDED_SOURCE_QUESTIONS
                            if question_id.startswith(f"tiwari:class8-science:ch{spec.sequence_number:02d}:")
                        ],
                        "retrieval_mode": "cached_after_refresh_failure" if cached else "fresh",
                        "refresh_error": fetch_error,
                    },
                )
            )
            candidates_by_chapter[spec.sequence_number].extend(html_candidates)
        except Exception as exc:
            errors.append(f"Tiwari chapter {spec.sequence_number}: {exc}")

    if include_papers:
        for paper in PAPER_SOURCES:
            spec = selected_by_number.get(paper.chapter_number)
            if not spec:
                continue
            qp_path = output_path / "downloads" / _safe_filename(paper.qp_url)
            answer_path = output_path / "downloads" / _safe_filename(paper.answer_url)
            try:
                qp_content, qp_type, qp_cached, qp_fetch_error = _fetch_with_cache(
                    paper.qp_url,
                    qp_path,
                    verify_tls=verify_tls,
                )
                answer_content, answer_type, answer_cached, answer_fetch_error = _fetch_with_cache(
                    paper.answer_url,
                    answer_path,
                    verify_tls=verify_tls,
                )
                qp_record = _record_download(
                    url=paper.qp_url,
                    kind="practice_question_paper_pdf",
                    content=qp_content,
                    content_type=qp_type,
                    target=qp_path,
                    publisher="Kendriya Vidyalaya Embassy of India, Kathmandu (M. S. KumarSwamy)",
                    license_status="copyright_review_required",
                    metadata={
                        "chapter_number": paper.chapter_number,
                        "paper_number": paper.paper_number,
                        "retrieval_mode": "cached_after_refresh_failure" if qp_cached else "fresh",
                        "refresh_error": qp_fetch_error,
                    },
                )
                answer_record = _record_download(
                    url=paper.answer_url,
                    kind="practice_answer_key_pdf",
                    content=answer_content,
                    content_type=answer_type,
                    target=answer_path,
                    publisher="Kendriya Vidyalaya Embassy of India, Kathmandu (M. S. KumarSwamy)",
                    license_status="copyright_review_required",
                    metadata={
                        "chapter_number": paper.chapter_number,
                        "paper_number": paper.paper_number,
                        "retrieval_mode": "cached_after_refresh_failure" if answer_cached else "fresh",
                        "refresh_error": answer_fetch_error,
                    },
                )
                manifest.extend((qp_record, answer_record))
                paper_candidates = parse_paper_pair(qp_path, answer_path, spec, paper.qp_url, paper.answer_url, paper.paper_number)
                paper_items = parse_paper_items(qp_path, answer_path, spec, paper.qp_url, paper.answer_url, paper.paper_number)
                candidates_by_chapter[paper.chapter_number].extend(paper_candidates)
                items_by_chapter[paper.chapter_number].extend(paper_items)
            except Exception as exc:
                errors.append(f"paper chapter {paper.chapter_number} paper {paper.paper_number}: {exc}")

    # Source coverage is uneven across chapters. Reuse the previous chapter's
    # assessment shape, but retarget the content to the current chapter's
    # concept manifest. Keep the pre-adaptation snapshot so adaptations do not
    # silently cascade through the whole book.
    adaptation_reports: list[dict[str, Any]] = []
    original_candidates_by_chapter = {
        chapter_number: list(candidates)
        for chapter_number, candidates in candidates_by_chapter.items()
    }
    if adapt_to_minimum is not None:
        for spec in selected:
            previous_spec = next(
                (chapter for chapter in selected if chapter.sequence_number == spec.sequence_number - 1),
                None,
            )
            if previous_spec is None:
                continue
            adapted = build_adapted_candidates(
                spec,
                previous_spec,
                original_candidates_by_chapter[previous_spec.sequence_number],
                current_count=len(candidates_by_chapter[spec.sequence_number]),
                minimum_count=adapt_to_minimum,
                existing_hashes={candidate.content_hash or "" for candidate in candidates_by_chapter[spec.sequence_number]},
            )
            candidates_by_chapter[spec.sequence_number].extend(adapted)
            items_by_chapter[spec.sequence_number].extend(QuestionItemCandidate.from_mcq(candidate) for candidate in adapted)
            adaptation_reports.append(
                {
                    "chapter": spec.sequence_number,
                    "previous_chapter": previous_spec.sequence_number,
                    "minimum_candidate_floor": adapt_to_minimum,
                    "source_candidates_before_adaptation": len(original_candidates_by_chapter[spec.sequence_number]),
                    "adapted_candidate_count": len(adapted),
                    "adaptation_mode": "structure_and_assessment_shape_only",
                }
            )

    reports: list[dict[str, Any]] = []
    for spec in selected:
        seen: set[str] = set()
        unique_candidates: list[QuestionCandidate] = []
        for candidate in candidates_by_chapter[spec.sequence_number]:
            if candidate.content_hash in seen:
                continue
            seen.add(candidate.content_hash or "")
            unique_candidates.append(candidate)
        unique_items = _deduplicate_items(items_by_chapter[spec.sequence_number])
        pack_path, _ = _chapter_paths(output_path, spec)
        payload = {
            "curriculum_version_id": str(CURRICULUM_VERSION_ID),
            "chapter_id": str(spec.id),
            "chapter_title": spec.title,
            "source_mode": "scraped_candidate_pack",
            "review_status": "review_required",
            "source_urls": sorted({candidate.source_url for candidate in unique_candidates}),
            "questions": [candidate.model_dump(mode="json") for candidate in unique_candidates],
            "items": [item.model_dump(mode="json") for item in unique_items],
        }
        pack_path.parent.mkdir(parents=True, exist_ok=True)
        pack_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        reports.append(
            {
                "chapter": spec.sequence_number,
                "title": spec.title,
                "candidate_count": len(unique_candidates),
                "item_count": len(unique_items),
                "adapted_candidate_count": sum(
                    1
                    for candidate in unique_candidates
                    if candidate.generation_spec.get("adaptation_mode") == "structure_and_assessment_shape_only"
                ),
                "question_types": dict(Counter(item.question_type for item in unique_items)),
                "sources": len(payload["source_urls"]),
                "pack": str(pack_path),
            }
        )

    manifest_path = output_path / "source_manifest.json"
    manifest_payload = {
        "run_type": "class8_science_source_scrape",
        "started_or_finished_at": _now(),
        "curriculum_version": "curiosity-2026-27",
        "rights_policy": "All downloaded material is candidate-only until permission/license and subject review are complete.",
        "sources": manifest,
        "errors": errors,
        "adaptation_policy": {
            "minimum_candidate_floor": adapt_to_minimum,
            "uses_previous_chapter": True,
            "mode": "structure_and_assessment_shape_only",
            "review_required": True,
        },
        "adaptations": adaptation_reports,
        "reports": reports,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"reports": reports, "manifest": str(manifest_path), "errors": errors}


def _parse_chapters(value: str) -> tuple[ChapterSpec, ...]:
    numbers: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            numbers.update(range(start, end + 1))
        elif part:
            numbers.add(int(part))
    return tuple(chapter for chapter in CHAPTERS if chapter.sequence_number in numbers)


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape source-backed Class 8 Science questions and adapt thin chapters.")
    parser.add_argument("--chapters", default="1-13")
    parser.add_argument("--output-dir", default="../tmp/class8-science-source-run")
    parser.add_argument("--no-papers", action="store_true", help="Skip the downloadable KV practice-paper corpus.")
    parser.add_argument(
        "--adapt-minimum-candidates",
        type=int,
        default=60,
        help="Adapt previous-chapter question structures until each eligible chapter has this many MCQ candidates; use 0 to disable.",
    )
    parser.add_argument("--insecure-tls", action="store_true", help="Allow the local fallback for broken certificate chains.")
    args = parser.parse_args()
    try:
        result = scrape(
            _parse_chapters(args.chapters),
            args.output_dir,
            include_papers=not args.no_papers,
            verify_tls=not args.insecure_tls,
            adapt_to_minimum=args.adapt_minimum_candidates or None,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if all(report["candidate_count"] >= 10 for report in result["reports"]) else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
