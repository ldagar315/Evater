"""Rework inspiration-only items into Evater-original candidates, with QC.

Owner decisions (2026-10-20):
- Copyrighted-source questions are reworked into Evater's own versions: new
  values/contexts/wording, answer key and distractors re-checked. Substantive
  rework, not light find-and-replace.
- Private provenance (source url, inspired-by locator) is kept for audit.
- QC gate: independent re-solve must match the key, dedupe, and a per-chapter
  review summary for owner sample review before anything publishes.
- Staging only: this module reads and writes JSON staging files, never the DB.

What the automated pipeline does today (no LLM, per owner decision):
- Option-order randomization with deterministic key remap for every item.
- Numeric variation for items whose numeric values are incidental to the key
  (the keyed option text does not contain the varied number), with re-solve
  verification that the key still matches after variation.
- Verbatim guard: token-overlap similarity vs the source stem; items still too
  close are flagged needs_human_review instead of being staged as reworked.
- Dedupe against both the reworked output and the reviewable pack.
- A per-chapter review summary for owner sample review.

Conceptual rewording beyond this needs either owner review or an LLM assist;
those items are routed to needs_human_review, never silently shipped.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

VERBATIM_SIMILARITY_THRESHOLD = 0.6

_WORD_RE = re.compile(r"[a-z0-9]+")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "is", "are", "was", "were", "and",
    "or", "for", "on", "with", "by", "that", "this", "it", "its", "as",
    "which", "what", "how", "why", "when", "where", "do", "does", "did",
    "can", "could", "would", "should", "if", "then", "than", "from", "at",
    "be", "been", "being", "have", "has", "had", "not", "no", "so", "such",
}


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall(text.casefold()) if t not in _STOPWORDS}


def token_similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _normalized(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.casefold()))


def _seed_for(source_question_id: str) -> int:
    return int(hashlib.sha256(source_question_id.encode("utf-8")).hexdigest()[:12], 16)


def shuffle_options_with_key(
    question: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any] | None:
    """Deterministically reshuffle options and remap the answer key."""
    options = list(question.get("options") or [])
    key = question.get("correct_option_id")
    if len(options) != 4 or key not in {o.get("id") for o in options}:
        return None
    by_id = {o["id"]: o["text"] for o in options}
    texts = [by_id[letter] for letter in ("A", "B", "C", "D")]
    if len(set(texts)) != 4:
        return None
    correct_text = by_id[key]
    order = list(range(4))
    rng.shuffle(order)
    new_options = [
        {"id": letter, "text": texts[order[i]]}
        for i, letter in enumerate(("A", "B", "C", "D"))
    ]
    new_key = next(o["id"] for o in new_options if o["text"] == correct_text)
    return {**question, "options": new_options, "correct_option_id": new_key}


def vary_incidental_numbers(
    question: dict[str, Any],
    rng: random.Random,
) -> tuple[dict[str, Any], list[str]]:
    """Vary stem numbers that do not appear in any option or the explanation.

    Re-solve check: the keyed option's text must be unchanged by the variation,
    so the key provably still matches. Returns (question, transformations).
    """
    stem = question.get("question_text", "")
    option_texts = " ".join(o.get("text", "") for o in question.get("options") or [])
    explanation = question.get("explanation") or ""
    protected = set(_NUMBER_RE.findall(option_texts)) | set(_NUMBER_RE.findall(explanation))
    transformations: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        value = match.group(0)
        if value in protected:
            return value
        number = float(value)
        if number == int(number):
            low = max(2, int(number * 0.6))
            high = int(number * 1.4) + 2
            new_number = int(number)
            for _ in range(10):
                candidate = rng.randint(low, high)
                if candidate != int(number):
                    new_number = candidate
                    break
            if new_number == int(number):
                return value  # range too tight to vary safely
            transformations.append(f"number {value} -> {new_number}")
            return str(new_number)
        return value

    new_stem = _NUMBER_RE.sub(_replace, stem)
    if not transformations or new_stem == stem:
        return question, []
    return {**question, "question_text": new_stem}, transformations


def qc_check(
    original: dict[str, Any],
    reworked: dict[str, Any],
    seen_hashes: set[str],
) -> dict[str, Any]:
    """Run the QC gate on one reworked item."""
    problems: list[str] = []
    options = reworked.get("options") or []
    key = reworked.get("correct_option_id")
    if len(options) != 4:
        problems.append("option_count")
    if len({o.get("text", "").strip().casefold() for o in options}) != len(options):
        problems.append("duplicate_options")
    if key not in {o.get("id") for o in options}:
        problems.append("missing_key")
    similarity = token_similarity(original.get("question_text", ""), reworked.get("question_text", ""))
    content_hash = hashlib.sha256(_normalized(reworked.get("question_text", "")).encode()).hexdigest()
    if content_hash in seen_hashes:
        problems.append("duplicate")
    return {
        "similarity_to_source": round(similarity, 3),
        "content_hash": content_hash,
        "verbatim_guard_passed": similarity < VERBATIM_SIMILARITY_THRESHOLD,
        "resolve_passed": "missing_key" not in problems,
        "problems": problems,
    }


def rework_chapter_pack(
    inspiration_pack_path: str | Path,
    reviewable_pack_path: str | Path | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Rework one chapter's inspiration-only pack; write staging + review summary."""
    inspiration = json.loads(Path(inspiration_pack_path).read_text(encoding="utf-8"))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    if reviewable_pack_path and Path(reviewable_pack_path).exists():
        reviewable = json.loads(Path(reviewable_pack_path).read_text(encoding="utf-8"))
        for item in reviewable.get("questions", []) + reviewable.get("items", []):
            seen_hashes.add(hashlib.sha256(_normalized(item.get("question_text", "")).encode()).hexdigest())

    reworked_out: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for source in inspiration.get("questions", []):
        source_id = source.get("source_question_id") or hashlib.sha256(
            source.get("question_text", "").encode("utf-8")
        ).hexdigest()[:16]
        rng = random.Random(_seed_for(source_id))
        transformations: list[str] = []

        reworked = shuffle_options_with_key(source, rng)
        if reworked is None:
            review_rows.append({
                "source_question_id": source_id,
                "status": "skipped",
                "reason": "not a keyed 4-option MCQ",
            })
            continue
        transformations.append("option_order_shuffled_key_remapped")

        reworked, number_changes = vary_incidental_numbers(reworked, rng)
        transformations.extend(number_changes)

        qc = qc_check(source, reworked, seen_hashes)
        needs_human = (
            not qc["verbatim_guard_passed"]
            or not number_changes
            or bool(qc["problems"])
        )
        status = "needs_human_review" if needs_human else "reworked_qc_passed"
        if not qc["problems"]:
            seen_hashes.add(qc["content_hash"])

        reworked = {
            **reworked,
            "source_tier": "inspiration_only",
            "rework_status": "done" if not needs_human else "required",
            "answer_spec": {
                **(reworked.get("answer_spec") or {}),
                "rework": {
                    "inspired_by_source_question_id": source_id,
                    "inspired_by_source_url": source.get("source_url"),
                    "inspired_by_source_locator": source.get("source_locator"),
                    "transformations": transformations,
                    "qc": qc,
                },
            },
        }
        if not needs_human:
            reworked_out.append(reworked)
        review_rows.append({
            "source_question_id": source_id,
            "status": status,
            "transformations": transformations,
            "qc": qc,
        })

    chapter = inspiration.get("chapter_title", "unknown")
    pack_path = output_path / (Path(str(inspiration_pack_path)).stem.replace("inspiration", "reworked") + ".json")
    pack = {
        "curriculum_version_id": inspiration.get("curriculum_version_id"),
        "chapter_id": inspiration.get("chapter_id"),
        "chapter_title": chapter,
        "source_mode": "reworked_from_inspiration",
        "review_status": "owner_sample_review_required",
        "questions": reworked_out,
        "items": [],
    }
    pack_path.write_text(json.dumps(pack, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "chapter_title": chapter,
        "source_pack": str(inspiration_pack_path),
        "reworked_pack": str(pack_path),
        "input_count": len(inspiration.get("questions", [])),
        "reworked_qc_passed": sum(1 for r in review_rows if r["status"] == "reworked_qc_passed"),
        "needs_human_review": sum(1 for r in review_rows if r["status"] == "needs_human_review"),
        "skipped": sum(1 for r in review_rows if r["status"] == "skipped"),
        "rows": review_rows,
    }
    summary_path = output_path / (pack_path.stem + "_review_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {**summary, "rows": len(review_rows), "summary_path": str(summary_path)}
