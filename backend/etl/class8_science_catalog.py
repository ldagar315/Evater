"""The current NCERT/CBSE Curiosity Grade 8 Science chapter catalog."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid5


CURRICULUM_VERSION_ID = UUID("11111111-1111-4111-8111-111111111111")
CURRICULUM_VERSION_LABEL = "curiosity-2026-27"
TEXTBOOK_NAME = "Curiosity — Textbook of Science for Grade 8"
SOURCE_ROOT = "https://ncert.nic.in/textbook/pdf/hecu1"
CHAPTER_ID_NAMESPACE = UUID("22222222-2222-4222-8222-222222222222")
CHAPTER_ONE_ID = CHAPTER_ID_NAMESPACE
GENERATED_PACK_DIR = Path(__file__).resolve().parents[2] / "tmp" / "class8-science-generated-packs"


@dataclass(frozen=True)
class ChapterSpec:
    sequence_number: int
    title: str
    description: str
    source_url: str
    id: UUID

    @property
    def slug(self) -> str:
        return "-".join("".join(character.lower() if character.isalnum() else " " for character in self.title).split())

    @property
    def pack_path(self) -> Path:
        return GENERATED_PACK_DIR / f"class8_science_chapter_{self.sequence_number:02d}.json"


def chapter_id(sequence_number: int) -> UUID:
    if sequence_number == 1:
        return CHAPTER_ONE_ID
    return uuid5(CHAPTER_ID_NAMESPACE, f"chapter:{sequence_number}")


CHAPTERS = (
    ChapterSpec(1, "Exploring the Investigative World of Science", "Investigation skills: focused questions, variables, observations, measurement, and evidence.", f"{SOURCE_ROOT}01.pdf", chapter_id(1)),
    ChapterSpec(2, "The Invisible Living World: Beyond Our Naked Eye", "Microorganisms, microscopes, beneficial and harmful microbes, and the living world beyond unaided sight.", f"{SOURCE_ROOT}02.pdf", chapter_id(2)),
    ChapterSpec(3, "Health: The Ultimate Treasure", "Health, nutrition, disease, immunity, hygiene, and public health choices.", f"{SOURCE_ROOT}03.pdf", chapter_id(3)),
    ChapterSpec(4, "Electricity: Magnetic and Heating Effects", "Electric circuits, conductors, heating effect, magnetic effect, electromagnets, motors, and safety.", f"{SOURCE_ROOT}04.pdf", chapter_id(4)),
    ChapterSpec(5, "Exploring Forces", "Forces, their effects, contact and non-contact interactions, and everyday applications.", f"{SOURCE_ROOT}05.pdf", chapter_id(5)),
    ChapterSpec(6, "Pressure, Winds, Storms, and Cyclones", "Pressure, moving air, storms, cyclones, forecasting, preparedness, and safety.", f"{SOURCE_ROOT}06.pdf", chapter_id(6)),
    ChapterSpec(7, "Particulate Nature of Matter", "Particles, their motion and spacing, states of matter, diffusion, and changes of state.", f"{SOURCE_ROOT}07.pdf", chapter_id(7)),
    ChapterSpec(8, "Nature of Matter: Elements, Compounds, and Mixtures", "Elements, atoms, compounds, mixtures, properties, and separation of substances.", f"{SOURCE_ROOT}08.pdf", chapter_id(8)),
    ChapterSpec(9, "The Amazing World of Solutes, Solvents, and Solutions", "Solutions, solubility, concentration, saturation, and separating dissolved substances.", f"{SOURCE_ROOT}09.pdf", chapter_id(9)),
    ChapterSpec(10, "Light: Mirrors and Lenses", "Reflection, refraction, mirrors, lenses, image formation, and optical applications.", f"{SOURCE_ROOT}10.pdf", chapter_id(10)),
    ChapterSpec(11, "Keeping Time with the Skies", "The Sun, Moon, apparent motion, lunar phases, calendars, and observing time cycles.", f"{SOURCE_ROOT}11.pdf", chapter_id(11)),
    ChapterSpec(12, "How Nature Works in Harmony", "Ecosystems, interdependence, food relationships, biodiversity, and environmental balance.", f"{SOURCE_ROOT}12.pdf", chapter_id(12)),
    ChapterSpec(13, "Our Home: Earth, a Unique Life Sustaining Planet", "Earth systems, habitability, atmosphere, water, climate, and responsible stewardship.", f"{SOURCE_ROOT}13.pdf", chapter_id(13)),
)


def get_chapter(sequence_number: int) -> ChapterSpec:
    try:
        return next(chapter for chapter in CHAPTERS if chapter.sequence_number == sequence_number)
    except StopIteration as exc:
        raise ValueError(f"Unknown Class 8 Science chapter: {sequence_number}") from exc


def chapters_after(sequence_number: int = 1) -> tuple[ChapterSpec, ...]:
    return tuple(chapter for chapter in CHAPTERS if chapter.sequence_number > sequence_number)
