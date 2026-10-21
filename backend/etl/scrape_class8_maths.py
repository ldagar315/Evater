"""Scrape the NCERT Ganita Prakash Grade 8 Mathematics chapter PDFs.

Source model (mirrors the Class 8 Science scraper's owner-approved stance):
- NCERT items are staged close to source with attribution (source tier
  ``ncert_attributed``); provenance (source_url, locator, question id) is kept
  privately for audit.
- The Ganita Prakash books ship an official answer key at the back of every
  chapter PDF, keyed by the printed book page ("Page No. N"). This scraper
  pairs each "Figure it Out" exercise question and each in-text prompt with its
  official answer, so most maths items carry a verifiable answer key - unlike
  the science book, whose answers were not in the PDF.
- MCQ-shaped exercise questions (inline (i)-(iv) options with a single-letter
  official answer) are additionally emitted as publishable MCQ candidates.
  Everything else is retained as a QuestionItemCandidate for the archive.
- Third-party aggregator content for this brand-new book barely exists yet; a
  later inspiration pass can map old-syllabus class 8 maths banks onto these
  chapters through the rework + QC pipeline (never published verbatim).
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from uuid import UUID, uuid5

from pypdf import PdfReader
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTChar, LTTextContainer, LTTextLine

from .class8_maths_catalog import (
    CHAPTERS,
    CURRICULUM_VERSION_ID,
    ChapterSpec,
)
from .class8_maths_topics import TOPICS
from .models import CandidateOption, QuestionCandidate, QuestionItemCandidate, QuestionItemType
from .scrape_class8_science import (
    NCERT_LICENSE,
    _bounded,
    _classify_item_type,
    _clean_source_prompt,
    _deduplicate_items,
    _fetch_with_cache,
    _has_diagram_reference,
    _inline_options,
    _media_reference,
    _normalize,
    _now,
    _numbered_blocks,
    _record_download,
)

MATHS_SOURCE_ID_PREFIX = "ncert:class8-maths"
TEXTBOOK_SHORT = "Ganita Prakash - Textbook of Mathematics for Grade 8"

_HEADER_PATTERNS = (
    re.compile(r"(?m)^\s*Ganita Prakash\s*\|\s*Grade\s*8\s*$"),
    re.compile(r"(?m)^\s*Chapter\s+\d+\.indd.*$"),
    re.compile(r"(?m)^\s*REPRINT.*$"),
)



def _extract_pages_fontaware(content: bytes) -> list[str]:
    """Extract page text with superscripts restored as ^ notation.

    NCERT maths PDFs render exponents as smaller raised glyphs; plain text
    extraction silently merges them into the base ("64^2" becomes "642"),
    corrupting the mathematics. pdfminer exposes per-character font size and
    baseline, so superscript runs are detected within each text line and
    re-emitted as base^exponent. A superscript run directly before a radical
    sign is left attached ("3V64", standard inline cube-root notation).
    """

    pages: list[str] = []
    with io.BytesIO(content) as buffer:
        for page_layout in extract_pages(buffer):
            page_lines: list[str] = []
            for element in page_layout:
                if not isinstance(element, LTTextContainer):
                    continue
                for line in element:
                    if not isinstance(line, LTTextLine):
                        continue
                    chars = [c for c in line if isinstance(c, LTChar)]
                    if not chars:
                        continue
                    body_size = max(c.size for c in chars)
                    baseline = sorted(c.y0 for c in chars)[len(chars) // 2]
                    out: list[str] = []
                    superscript_run: list[str] = []
                    for char in chars:
                        is_superscript = (
                            char.size <= 0.7 * body_size
                            and char.y0 > baseline + 0.2 * body_size
                            and char.get_text().strip()
                        )
                        if is_superscript:
                            superscript_run.append(char.get_text())
                            continue
                        if superscript_run:
                            out.append(_emit_superscript("".join(superscript_run), char.get_text()))
                            superscript_run = []
                        out.append(char.get_text())
                    if superscript_run:
                        out.append("^" + "".join(superscript_run))
                    page_lines.append("".join(out))
            pages.append("\n".join(page_lines))
    return pages


def _emit_superscript(run: str, next_text: str) -> str:
    """Render a superscript run; keep radical indices attached to the sign."""

    if next_text.startswith("\u221a"):
        return run
    return "^" + run


def _strip_page_furniture(page_text: str, spec: ChapterSpec) -> str:
    """Remove running headers/footers so blocks parse cleanly across pages."""

    text = page_text
    for pattern in _HEADER_PATTERNS:
        text = pattern.sub("", text)
    # Chapter-title running header, e.g. "A Square and A Cube" alone on a line.
    title_pattern = re.compile(rf"(?m)^\s*{re.escape(spec.title)}\s*$", flags=re.I)
    text = title_pattern.sub("", text)
    # Print-shop datetime footers, e.g. "10-07-2025   14:06:44".
    text = re.sub(r"(?m)^\s*\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2}\s*$", "", text)
    return text


def _printed_page_number(page_text: str) -> int | None:
    """Recover the book's printed page number from the page header or footer."""

    raw_lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    lines = raw_lines[:6] + raw_lines[-6:]
    for line in lines:
        if re.fullmatch(r"\d{1,3}", line):
            return int(line)
        match = re.search(r"\|\s*(\d{1,3})\s*$", line)
        if match:
            return int(match.group(1))
    return None


def _split_answer_key(pages: list[str]) -> tuple[list[str], list[str]]:
    """Split extracted pages into body pages and answer-key pages."""

    for index, page_text in enumerate(pages):
        if re.search(r"Page\s+No\.\s*\d+", page_text):
            return pages[:index], pages[index:]
    return pages, []


def _fix_answer_marker(text: str) -> str:
    """Repair answer markers split across lines by the PDF text layer."""

    text = re.sub(r"A\s*\n+\s*ns\.", "Ans.", text)
    text = re.sub(r"\(\s*i\s*\n+\s*i\s*\)", "(ii)", text)
    return text


_ROMAN_LABELS = ("i", "ii", "iii", "iv", "v", "vi")


def _options_by_label(text: str) -> tuple[str, list[dict[str, str]]]:
    """Extract roman-numeral options regardless of two-column interleaving.

    The Ganita Prakash exercises set options in two visual columns; pdfminer
    boxes keep each option's text with its label but emit the labels out of
    order (i, iv, ii, v, iii). Slice each option's text between markers, then
    sort by label. Accept only a complete set of four or five options starting
    at (i); anything else is returned as plain text.
    """

    text = re.split(r"\bSUMMARY\b", text)[0]
    text = "".join(char for char in text if char == "\n" or char == "\t" or ord(char) >= 32)
    text = re.sub("[\x7f-\x9f]", "", text)
    markers = list(re.finditer(r"(?<!\w)\((i{1,3}|iv|v|vi)\)\s+", text, flags=re.I))
    if len(markers) < 4:
        return text, []
    labels = [marker.group(1).casefold() for marker in markers]
    if sorted(labels, key=_ROMAN_LABELS.index) != list(_ROMAN_LABELS[: len(labels)]) or len(labels) > 5:
        return text, []
    stem = _normalize(text[: markers[0].start()])
    slices: dict[str, str] = {}
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        slices[marker.group(1).casefold()] = _normalize(text[marker.end() : end])
    options = [
        {"id": chr(ord("A") + position), "text": slices[label]}
        for position, label in enumerate(_ROMAN_LABELS[: len(labels)])
    ]
    if any(not option["text"] for option in options):
        return text, []
    return stem, options


def _reattach_stray_options(questions: dict[int, str]) -> dict[int, str]:
    """Move option lines that column layout dropped into the next question.

    A two-column option row can land inside the following numbered block. If a
    block holds an incomplete label set (e.g. i, ii, iv, v) and a later block
    that has no roman markers of its own starts with lines filling the gap
    (iii), move those lines back.
    """

    result = dict(questions)
    numbers = sorted(result)
    for position, number in enumerate(numbers):
        labels = re.findall(r"(?<!\w)\(((?:i{1,3}|iv|vi?))\)\s+", result[number], flags=re.I)
        if not labels:
            continue
        ordered = [label.casefold() for label in labels]
        missing = [label for label in _ROMAN_LABELS if label not in ordered]
        if not missing or _ROMAN_LABELS.index(missing[0]) > len(set(ordered)):
            continue
        for later in numbers[position + 1 :]:
            later_labels = {
                match.casefold()
                for match in re.findall(r"(?<!\w)\(((?:i{1,3}|iv|vi?))\)\s+", result[later], flags=re.I)
            }
            if later_labels - set(missing):
                break  # later block has an option set of its own
            moved: list[str] = []
            kept: list[str] = []
            for line in result[later].splitlines():
                marker = re.match(r"^\s*\((i{1,3}|iv|vi?)\)\s+", line, flags=re.I)
                if marker and marker.group(1).casefold() in missing:
                    moved.append(line)
                else:
                    kept.append(line)
            if moved:
                result[number] = result[number].rstrip() + "\n" + "\n".join(moved)
                result[later] = "\n".join(kept)
            break
    return result


def _numbered_question_blocks(block_text: str) -> dict[int, str]:
    """Parse the consecutively numbered questions of a Figure it Out block."""

    starts = list(re.finditer(r"(?m)^\s*(\d{1,2})\.\s+", block_text))
    blocks: dict[int, str] = {}
    expected = 1
    for index, start in enumerate(starts):
        number = int(start.group(1))
        if number != expected:
            continue
        end = starts[index + 1].start() if index + 1 < len(starts) else len(block_text)
        blocks[number] = block_text[start.end() : end]
        expected += 1
    return blocks


def _extract_figure_it_out(body_pages: list[str], spec: ChapterSpec) -> list[dict[str, Any]]:
    """Extract every Figure it Out block, tagged with its printed page number."""

    blocks: list[dict[str, Any]] = []
    for page_index, raw_page in enumerate(body_pages):
        page_text = _strip_page_furniture(raw_page, spec)
        book_page_for_page = _printed_page_number(raw_page)
        if book_page_for_page is not None:
            page_text = re.sub(rf"(?m)^\s*{book_page_for_page}\s*$", "", page_text)
        for match in re.finditer(r"Figure\s+it\s+Out", page_text):
            book_page = book_page_for_page
            blocks.append(
                {
                    "book_page": book_page,
                    "pdf_page": page_index + 1,
                    "text": page_text[match.end() :],
                }
            )
    # A block's questions may flow onto following pages; the next Figure it Out
    # or the end of the body terminates it. Since we sliced per page, append
    # following pages' text to each block until the next block start.
    for index, block in enumerate(blocks):
        if index + 1 < len(blocks):
            continue
        block["text"] = block["text"]  # last block keeps only its own page; extended below
    # Extend blocks with subsequent pages up to the next block's page.
    for index, block in enumerate(blocks):
        start_page = block["pdf_page"]
        end_page = blocks[index + 1]["pdf_page"] if index + 1 < len(blocks) else len(body_pages) + 1
        extra = "\n".join(
            _strip_page_furniture(body_pages[p], spec)
            for p in range(start_page, min(end_page - 1, len(body_pages)))
        )
        if extra:
            block["text"] = block["text"] + "\n" + extra
    return blocks


def _cut_answer_at_body_line(answer: str, body_text: str) -> str:
    """Truncate an answer where the key glued the next question onto it.

    The printed key sometimes flows an exercise's answer straight into the next
    question's text with no marker. Body pages carry the questions verbatim, so
    the first answer line (after line one) attested in the body is really the
    next question - cut there.
    """

    if not body_text:
        return answer
    normalized_body = _normalize(body_text)
    lines = answer.split("\n")
    for index in range(1, len(lines)):
        probe = _normalize(lines[index])
        if len(probe) >= 25 and probe in normalized_body:
            return "\n".join(lines[:index])
    return answer


def _parse_answer_key(answer_pages: list[str], body_text: str = "") -> dict[int, dict[str, Any]]:
    """Parse the back-of-chapter answer key, keyed by printed book page."""

    text = _fix_answer_marker("\n".join(answer_pages))
    sections: dict[int, dict[str, Any]] = {}
    markers = list(re.finditer(r"Page\s+No\.\s*(\d+)", text))
    open_fio_answers: dict[int, str] | None = None
    for index, marker in enumerate(markers):
        book_page = int(marker.group(1))
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        section = text[marker.end() : end]
        entry: dict[str, Any] = {"raw": section, "figure_it_out": None, "prompts": []}
        fio = re.search(r"Figure\s+it\s+Out", section)
        continuation = False
        if not fio and open_fio_answers:
            probe = _numbered_blocks(section, maximum=200)
            if probe and min(probe) == max(open_fio_answers) + 1:
                # The key splits one Figure it Out across "Page No." sections
                # when its questions flow onto the next printed page.
                fio = re.search(r"$^", section)  # never matches; reuse the FIO branch
                continuation = True
        if fio or continuation:
            numbered = _numbered_blocks(section[fio.end() :] if fio else section, maximum=200)
            answers: dict[int, str] = {}
            for number, raw_block in numbered.items():
                fragments = re.split(r"\bAns[.:]\s*", raw_block)
                if len(fragments) > 1:
                    fragments[1:] = [
                        _cut_answer_at_body_line(fragment, body_text)
                        for fragment in fragments[1:]
                    ]
                if len(fragments) > 2:
                    # Multi-part question: keep each keyed fragment, cutting the
                    # next sub-part's echoed prompt out of the fragment tail.
                    parts: list[str] = []
                    for fragment in fragments[1:]:
                        echo = re.search(r"\((?:i{1,3}|iv|vi?)\)\s", fragment)
                        if echo:
                            fragment = fragment[: echo.start()]
                        if fragment.strip():
                            parts.append(fragment)
                    answers[number] = _normalize("; ".join(parts))
                    answers[number] = re.sub(r"\s+\d{1,2}$", "", answers[number])
                elif len(fragments) == 2:
                    answers[number] = re.sub(r"\s+\d{1,2}$", "", _normalize(fragments[1]))
                else:
                    answers[number] = ""
            if continuation and open_fio_answers is not None:
                open_fio_answers.update(answers)
                entry["continuation_of_fio"] = True
            else:
                entry["figure_it_out"] = answers
                open_fio_answers = answers
        else:
            entry["prompts"] = _parse_prompt_answers(section, body_text)
            open_fio_answers = None
        sections[book_page] = entry
    return sections


def _verify_prompt_against_body(prompt: str, body_text: str) -> str:
    """Trim boundary junk from an answer-key prompt using the body text.

    Prompt/answer splitting can leave a previous answer's tail glued to the
    prompt start. The body carries the prompt verbatim, so drop leading lines
    until the remainder is attested in the body.
    """

    normalized_body = _normalize(body_text)
    candidate = prompt
    while candidate:
        probe = _normalize(candidate)[:60]
        if len(probe) >= 10 and probe in normalized_body:
            return candidate
        lines = candidate.split("\n")
        if len(lines) <= 1:
            return candidate
        candidate = "\n".join(lines[1:])
    return prompt


def _parse_prompt_answers(section: str, body_text: str = "") -> list[dict[str, str]]:
    """Parse (prompt, answer) pairs from a non-exercise answer-key section."""

    pairs: list[dict[str, str]] = []
    segments = re.split(r"\bAns[.:]\s*", section)
    if len(segments) < 2:
        return pairs
    prompt = segments[0]
    for index in range(1, len(segments)):
        segment = segments[index]
        if index == len(segments) - 1:
            answer, next_prompt = segment, ""
        else:
            answer, next_prompt = _split_answer_from_next_prompt(segment)
        if body_text:
            prompt = _verify_prompt_against_body(prompt, body_text)
            answer = _cut_answer_at_body_line(answer, body_text)
        prompt_clean = _normalize(prompt)
        answer_clean = _normalize(answer)
        if len(prompt_clean) >= 10 and answer_clean:
            pairs.append({"prompt": prompt_clean, "answer": answer_clean})
        prompt = next_prompt
    return pairs


def _split_answer_from_next_prompt(segment: str) -> tuple[str, str]:
    """Split 'answer text next prompt text' using the question-mark boundary."""

    qmark = segment.find("?")
    if qmark == -1:
        return segment, ""
    # The next prompt begins at the start of the line/sentence containing the
    # first question mark; answers in this book do not themselves ask questions.
    line_start = segment.rfind("\n", 0, qmark)
    candidate_start = line_start + 1
    # Walk further up while the line above looks like a continuation (no
    # sentence-ending punctuation), so multi-sentence prompts stay whole.
    while candidate_start > 0:
        prev_end = segment.rfind("\n", 0, candidate_start - 1)
        prev_line = segment[prev_end + 1 : candidate_start].strip()
        if not prev_line or prev_line.endswith((".", "?", ":", "!", ")", '"')):
            break
        candidate_start = prev_end + 1
    return segment[:candidate_start], segment[candidate_start:]


def _concept_id_for(spec: ChapterSpec, question_text: str, option_texts: Iterable[str]) -> tuple[UUID, str]:
    corpus = f"{question_text} {' '.join(option_texts)}".casefold()
    seeds = TOPICS[spec.sequence_number]
    best_seed = max(
        seeds,
        key=lambda seed: sum(
            term in corpus
            for term in (seed.slug.replace("-", " "), seed.title.casefold(), *seed.keywords)
        ),
    )
    return uuid5(spec.id, f"concept:{best_seed.slug}"), best_seed.slug


def _make_maths_item(
    *,
    spec: ChapterSpec,
    question_text: str,
    question_type: QuestionItemType,
    source_locator: str,
    source_question_id: str,
    options: list[dict[str, Any]] | None = None,
    answer: str | None = None,
    marks: int = 1,
    page_number: int | None = None,
) -> QuestionItemCandidate:
    option_values = [str(option.get("text", "")) for option in (options or [])]
    concept_id, concept_slug = _concept_id_for(spec, question_text, option_values)
    style = "diagram" if question_type == "diagram_based" else "direct"
    return QuestionItemCandidate(
        chapter_id=spec.id,
        concept_id=concept_id,
        question_type=question_type,
        question_text=_bounded(question_text, 5000),
        options=options or [],
        explanation=_bounded(answer, 5000) if answer else None,
        difficulty="medium",
        cognitive_level="apply" if question_type in {"numerical", "mcq_single"} else "understand",
        skill_tags=["source_item", concept_slug, question_type],
        question_style=style,  # type: ignore[arg-type]
        estimated_time_seconds=120 if question_type in {"long_answer", "diagram_based"} else 90,
        marks=max(1, min(marks, 20)),
        source_url=spec.source_url,
        source_locator=source_locator,
        source_question_id=source_question_id,
        source_tier="ncert_attributed",
        rework_status="not_required",
        source_license=NCERT_LICENSE,
        source_attribution=(
            f"NCERT, {TEXTBOOK_SHORT}, Chapter {spec.sequence_number}: {spec.title} (ncert.nic.in)"
        ),
        media=_media_reference(spec.source_url, source_locator, page_number) if _has_diagram_reference(question_text) else [],
        answer_spec={
            "has_answer_key": bool(answer),
            "source_answer": answer or None,
            "source_kind": "official_ncert_pdf",
            "verification": "official_answer_key" if answer else "no_key_in_source",
        },
        license_status="review_required",
        review_status="review",
    )


_ROMAN_TO_LETTER = {"i": "A", "ii": "B", "iii": "C", "iv": "D"}


def _mcq_candidate_from_item(
    *,
    spec: ChapterSpec,
    stem: str,
    options: list[dict[str, Any]],
    answer: str,
    source_question_id: str,
    source_locator: str,
) -> QuestionCandidate | None:
    """Promote an MCQ-shaped exercise question with an official single-letter key."""

    if len(options) != 4:
        return None
    key_match = re.match(r"^\(?\s*(i{1,3}|iv)\s*\)?(?:\s|$|,|\.)", answer.strip(), flags=re.I)
    if not key_match:
        return None
    if re.search(r"\((?:i{1,3}|iv|vi?)\)", answer[key_match.end() :], flags=re.I):
        return None  # multi-select key, not a single-answer MCQ
    correct = _ROMAN_TO_LETTER[key_match.group(1).casefold()]
    option_pairs = [(str(option["id"]), str(option["text"])) for option in options]
    concept_id, concept_slug = _concept_id_for(spec, stem, [text for _, text in option_pairs])
    import hashlib

    return QuestionCandidate(
        chapter_id=spec.id,
        concept_id=concept_id,
        question_text=_bounded(stem, 1000),
        options=[CandidateOption(id=option_id, text=_bounded(text, 500)) for option_id, text in option_pairs],
        correct_option_id=correct,  # type: ignore[arg-type]
        explanation=_bounded(f"Official NCERT answer key: option ({key_match.group(1).casefold()}).", 2000),
        hint=None,
        difficulty="medium",
        cognitive_level="apply",
        skill_tags=["source_mcq", concept_slug],
        misconception_tags=[],
        question_style="direct",  # type: ignore[arg-type]
        estimated_time_seconds=60,
        marks=1,
        source_url=spec.source_url,
        source_locator=source_locator,
        source_question_id=source_question_id,
        source_tier="ncert_attributed",
        rework_status="not_required",
        source_license=NCERT_LICENSE,
        source_attribution=(
            f"NCERT, {TEXTBOOK_SHORT}, Chapter {spec.sequence_number}: {spec.title} (ncert.nic.in)"
        ),
        question_family_key=f"ganita:{spec.sequence_number:02d}:{concept_slug}:{hashlib.sha1(_normalize(stem).encode()).hexdigest()[:12]}",
        variant_key="official_ncert_pdf",
        answer_spec={
            "type": "mcq_single",
            "source_answer": correct,
            "verification": "official_answer_key",
            "answer_source_url": spec.source_url,
        },
        generation_spec={
            "ingestion_method": "scraped",
            "source_kind": "official_ncert_pdf",
            "retrieved_at": _now(),
            "verification_state": "official_answer_key",
        },
        license_status="review_required",
        review_status="review",
    )


def parse_chapter(content: bytes, spec: ChapterSpec) -> tuple[list[QuestionCandidate], list[QuestionItemCandidate], dict[str, Any]]:
    reader = PdfReader(io.BytesIO(content))
    try:
        pages = _extract_pages_fontaware(content)
        if len(pages) != len(reader.pages) or not any(pages):
            raise ValueError("font-aware extraction returned incomplete pages")
    except Exception:
        pages = [page.extract_text() or "" for page in reader.pages]
    body_pages, answer_pages = _split_answer_key(pages)
    answer_key = _parse_answer_key(answer_pages, "\n".join(body_pages))

    candidates: list[QuestionCandidate] = []
    items: list[QuestionItemCandidate] = []
    unmatched_answers: list[int] = []

    for block in _extract_figure_it_out(body_pages, spec):
        book_page = block["book_page"]
        questions = _numbered_question_blocks(block["text"])
        answers = (answer_key.get(book_page) or {}).get("figure_it_out") or {}
        if book_page in answer_key and answer_key[book_page]["figure_it_out"] is None:
            unmatched_answers.append(book_page)
        questions = _reattach_stray_options(questions)
        for number, raw_block in questions.items():
            if answers and number not in answers:
                continue  # prose enumeration false-positive; not in the official key
            prompt = _clean_source_prompt(raw_block)
            if len(prompt) < 10:
                continue
            stem, options = _options_by_label(prompt)
            answer = answers.get(number, "")
            question_type = _classify_item_type(prompt)
            locator = (
                f"Official NCERT PDF page {block['pdf_page']}"
                + (f" (book page {book_page})" if book_page else "")
                + f", Figure it Out, Question {number}"
            )
            source_question_id = (
                f"{MATHS_SOURCE_ID_PREFIX}:ch{spec.sequence_number:02d}:"
                f"fio-p{book_page or block['pdf_page']:02d}:q{number:02d}"
            )
            item = _make_maths_item(
                spec=spec,
                question_text=stem if options else prompt,
                question_type=question_type,
                source_locator=locator,
                source_question_id=source_question_id,
                options=[{"id": option["id"], "text": option["text"]} for option in options],
                answer=answer or None,
                page_number=block["pdf_page"],
            )
            items.append(item)
            if options and answer:
                candidate = _mcq_candidate_from_item(
                    spec=spec,
                    stem=stem,
                    options=[{"id": option["id"], "text": option["text"]} for option in options],
                    answer=answer,
                    source_question_id=source_question_id,
                    source_locator=locator,
                )
                if candidate:
                    candidates.append(candidate)

    # In-text prompts (Math Talk etc.) recovered from the answer key.
    prompt_count = 0
    for book_page, entry in sorted(answer_key.items()):
        if entry["figure_it_out"] is not None:
            continue
        for pair in entry["prompts"]:
            prompt_count += 1
            source_question_id = (
                f"{MATHS_SOURCE_ID_PREFIX}:ch{spec.sequence_number:02d}:talk-p{book_page:02d}:q{prompt_count:02d}"
            )
            locator = f"Official NCERT answer key, Page No. {book_page}, in-text prompt {prompt_count}"
            items.append(
                _make_maths_item(
                    spec=spec,
                    question_text=pair["prompt"],
                    question_type=_classify_item_type(pair["prompt"]),
                    source_locator=locator,
                    source_question_id=source_question_id,
                    answer=pair["answer"],
                )
            )

    report = {
        "chapter": spec.sequence_number,
        "title": spec.title,
        "pdf_pages": len(pages),
        "answer_key_pages": len(answer_pages),
        "figure_it_out_blocks": len(_extract_figure_it_out(body_pages, spec)),
        "exercise_items": sum(1 for item in items if ":fio-" in (item.source_question_id or "")),
        "intext_items": sum(1 for item in items if ":talk-" in (item.source_question_id or "")),
        "items_with_answers": sum(1 for item in items if item.answer_spec.get("has_answer_key")),
        "mcq_candidates": len(candidates),
        "question_types": dict(Counter(item.question_type for item in items)),
    }
    return candidates, _deduplicate_items(items), report


def scrape(
    chapters: Iterable[ChapterSpec],
    output_dir: str | Path,
    *,
    verify_tls: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    (output_path / "downloads").mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    errors: list[str] = []
    reports: list[dict[str, Any]] = []

    for spec in chapters:
        official_path = output_path / "downloads" / f"class8_maths_chapter_{spec.sequence_number:02d}.pdf"
        try:
            content, content_type, cached, fetch_error = _fetch_with_cache(
                spec.source_url,
                official_path,
                verify_tls=verify_tls,
            )
            candidates, items, report = parse_chapter(content, spec)
            report.update(
                {
                    "pack": str(output_path / f"class8_maths_scraped_chapter_{spec.sequence_number:02d}.json"),
                    "retrieval_mode": "cached_after_refresh_failure" if cached else "fresh",
                    "refresh_error": fetch_error,
                }
            )
            manifest.append(
                _record_download(
                    url=spec.source_url,
                    kind="official_curriculum_pdf",
                    content=content,
                    content_type=content_type,
                    target=official_path,
                    publisher="National Council of Educational Research and Training",
                    license_status="copyright_ncert_attribution_owner_approved",
                    metadata={
                        "chapter_number": spec.sequence_number,
                        "extracted_item_count": report["exercise_items"] + report["intext_items"],
                        "mcq_candidate_count": report["mcq_candidates"],
                    },
                )
            )
            pack_payload = {
                "curriculum_version_id": str(CURRICULUM_VERSION_ID),
                "chapter_id": str(spec.id),
                "chapter_title": spec.title,
                "source_mode": "scraped_candidate_pack",
                "review_status": "review_required",
                "tiering": "ncert_attributed only; inspiration staging is a separate pass",
                "source_urls": [spec.source_url],
                "questions": [candidate.model_dump(mode="json") for candidate in candidates],
                "items": [item.model_dump(mode="json") for item in items],
            }
            pack_path = Path(report["pack"])
            pack_path.write_text(json.dumps(pack_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            reports.append(report)
        except Exception as exc:
            errors.append(f"chapter {spec.sequence_number}: {exc}")

    manifest_payload = {
        "run_type": "class8_maths_source_scrape",
        "started_or_finished_at": _now(),
        "curriculum_version": "ganita-prakash-2025-26",
        "rights_policy": (
            "NCERT items are staged close to source with attribution (owner-approved stance). "
            "The official back-of-chapter answer key supplies has_answer_key provenance. "
            "Provenance (source_url, locator, question id) is kept privately for audit."
        ),
        "sources": manifest,
        "errors": errors,
        "reports": reports,
    }
    manifest_path = output_path / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"reports": reports, "manifest": str(manifest_path), "errors": errors}


def _parse_chapters(value: str) -> tuple[ChapterSpec, ...]:
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            selected.update(range(start, end + 1))
        elif part:
            selected.add(int(part))
    try:
        return tuple(
            next(chapter for chapter in CHAPTERS if chapter.sequence_number == number)
            for number in sorted(selected)
        )
    except StopIteration as exc:
        raise ValueError(f"Unknown chapter in --chapters={value!r}.") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape Class 8 Maths NCERT chapter PDFs into candidate packs.")
    parser.add_argument("--chapters", default="1-14", help="Comma-separated chapter numbers or ranges, e.g. 1-3,5")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for packs, downloads, and the manifest")
    parser.add_argument("--no-verify-tls", action="store_true", help="Disable TLS verification for source fetches")
    args = parser.parse_args()
    result = scrape(
        _parse_chapters(args.chapters),
        args.output_dir,
        verify_tls=not args.no_verify_tls,
    )
    print(json.dumps(result, indent=2))
    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
