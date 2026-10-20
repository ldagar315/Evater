"""The NCERT Ganita Prakash Grade 8 Mathematics chapter catalog (2025-26).

Two parts, seven chapters each. NCERT PDF codes restart per part:
sequence 1-7 -> hegp1NN.pdf (Part I), sequence 8-14 -> hegp2NN.pdf (Part II,
whose internal chapter numbering restarts at 1). Verified against the printed
chapter title pages on 2026-10-20.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid5


CURRICULUM_VERSION_ID = UUID("77777777-7777-4777-8777-777777777777")
CURRICULUM_VERSION_LABEL = "ganita-prakash-2025-26"
TEXTBOOK_NAME = "Ganita Prakash — Textbook of Mathematics for Grade 8"
SOURCE_ROOT = "https://ncert.nic.in/textbook/pdf/hegp"
CHAPTER_ID_NAMESPACE = UUID("88888888-8888-4888-8888-888888888888")
GENERATED_PACK_DIR = Path(__file__).resolve().parents[2] / "tmp" / "class8-maths-generated-packs"


@dataclass(frozen=True)
class ChapterSpec:
    sequence_number: int  # 1-14 across both parts
    part: int  # 1 or 2
    part_chapter_number: int  # 1-7 within the part (matches the printed book)
    title: str
    description: str
    source_url: str
    id: UUID

    @property
    def slug(self) -> str:
        return "-".join("".join(character.lower() if character.isalnum() else " " for character in self.title).split())

    @property
    def pack_path(self) -> Path:
        return GENERATED_PACK_DIR / f"class8_maths_chapter_{self.sequence_number:02d}.json"


def chapter_id(sequence_number: int) -> UUID:
    return uuid5(CHAPTER_ID_NAMESPACE, f"chapter:{sequence_number}")


def _source_url(part: int, part_chapter_number: int) -> str:
    return f"{SOURCE_ROOT}{part}{part_chapter_number:02d}.pdf"


def _spec(sequence_number: int, part: int, part_chapter_number: int, title: str, description: str) -> ChapterSpec:
    return ChapterSpec(
        sequence_number=sequence_number,
        part=part,
        part_chapter_number=part_chapter_number,
        title=title,
        description=description,
        source_url=_source_url(part, part_chapter_number),
        id=chapter_id(sequence_number),
    )


CHAPTERS = (
    _spec(1, 1, 1, "A Square and A Cube", "Square and cube numbers, perfect squares and cubes, patterns, and puzzles built on them."),
    _spec(2, 1, 2, "Power Play", "Exponents, powers and their laws, scientific notation, and comparing large and small numbers."),
    _spec(3, 1, 3, "A Story of Numbers", "The history and structure of number systems, place value, and representations across cultures."),
    _spec(4, 1, 4, "Quadrilaterals", "Properties of quadrilaterals, angles, diagonals, and special families such as parallelograms."),
    _spec(5, 1, 5, "Number Play", "Factors, multiples, divisibility, primes, and number patterns and games."),
    _spec(6, 1, 6, "We Distribute, Yet Things Multiply", "The distributive property, identities, and algebraic thinking with multiplication."),
    _spec(7, 1, 7, "Proportional Reasoning-1", "Ratio, proportion, unitary thinking, and first applications of proportional relationships."),
    _spec(8, 2, 1, "Fractions in Disguise", "Fractions as percentages and decimals, conversions, and percentage problems in context."),
    _spec(9, 2, 2, "The Baudhayana-Pythagoras Theorem", "The Pythagoras theorem, its history from Baudhayana, proofs, and applications."),
    _spec(10, 2, 3, "Proportional Reasoning-2", "Further proportional reasoning: mixtures, rates, percentages in proportion, and inverse relationships."),
    _spec(11, 2, 4, "Exploring Some Geometric Themes", "Geometric constructions, transformations, symmetry, and spatial reasoning."),
    _spec(12, 2, 5, "Tales by Dots and Lines", "Graphs and coordinate thinking: plotting points, reading graphs, and stories told by data."),
    _spec(13, 2, 6, "Algebra Play", "Algebraic expressions, simple equations, patterns, and generalisation."),
    _spec(14, 2, 7, "Area", "Areas of polygons and composite shapes, deriving area formulas, and measurement in context."),
)


def get_chapter(sequence_number: int) -> ChapterSpec:
    try:
        return next(chapter for chapter in CHAPTERS if chapter.sequence_number == sequence_number)
    except StopIteration as exc:
        raise ValueError(f"Unknown Class 8 Maths chapter: {sequence_number}") from exc


def chapters_after(sequence_number: int = 1) -> tuple[ChapterSpec, ...]:
    return tuple(chapter for chapter in CHAPTERS if chapter.sequence_number > sequence_number)
