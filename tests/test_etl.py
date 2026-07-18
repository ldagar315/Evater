from pathlib import Path
from uuid import uuid4

from etl.class8_science_catalog import CHAPTERS, chapters_after
from etl.adapt_class8_science import build_adapted_candidates
from etl.generate_class8_science_packs import build_pack
from etl.generate_first_chapter_pack import CHAPTER_ID, CURRICULUM_VERSION_ID
from etl.models import QuestionItemCandidate
from etl.scrape_class8_science import _deduplicate_items, _inline_options, _parse_pdf_questions, parse_tiwari_html
from etl.seed_class8_science import build_seed_rows as build_catalog_seed_rows, source_id_for
from etl.seed_first_chapter import build_seed_rows as build_first_seed_rows
from etl.validate import load_candidates, validate_publishable_pack, validate_question_items


PACK = Path(__file__).parents[1] / "backend" / "etl" / "data" / "class8_science_chapter1.json"


def test_first_chapter_pack_passes_the_publish_gate():
    candidates = load_candidates(PACK)
    report = validate_publishable_pack(candidates)

    assert report["publishable"] is True
    assert report["count"] == 100
    assert report["distribution"]["difficulty"] == {"easy": 40, "medium": 40, "hard": 20}
    assert report["distribution"]["cognitive_level"] == {
        "recall": 30,
        "understand": 35,
        "apply": 25,
        "analyze": 10,
    }


def test_seed_rows_are_idempotent_and_linked_to_the_pack():
    candidates = load_candidates(PACK)
    rows = build_first_seed_rows(candidates)

    assert rows["curriculum_version"]["id"] == str(CURRICULUM_VERSION_ID)
    assert rows["chapter"]["id"] == str(CHAPTER_ID)
    assert len(rows["concepts"]) == 5
    assert rows["ingestion_job"]["input_count"] == 100
    assert rows["source"]["content_hash"]


def test_current_curiosity_catalog_has_thirteen_chapters():
    assert [chapter.sequence_number for chapter in CHAPTERS] == list(range(1, 14))
    assert CHAPTERS[0].title == "Exploring the Investigative World of Science"
    assert CHAPTERS[-1].title == "Our Home: Earth, a Unique Life Sustaining Planet"
    assert len({chapter.source_url for chapter in CHAPTERS}) == 13
    assert len({chapter.id for chapter in CHAPTERS}) == 13


def test_remaining_chapter_packs_pass_the_same_publish_gate():
    for chapter in chapters_after():
        candidates = load_candidates(chapter.pack_path)
        report = validate_publishable_pack(candidates)

        assert report["publishable"] is True
        assert all(candidate.chapter_id == chapter.id for candidate in candidates)
        assert len({candidate.question_family_key for candidate in candidates}) == 10
        assert all(candidate.generation_spec["chapter_sequence"] == chapter.sequence_number for candidate in candidates)


def test_generated_chapter_seed_rows_are_stable_and_linked():
    chapter = CHAPTERS[1]
    candidates = build_pack(chapter)
    rows = build_catalog_seed_rows(chapter, candidates)

    assert rows["chapter"]["id"] == str(chapter.id)
    assert rows["concepts"][0]["chapter_id"] == str(chapter.id)
    assert rows["source"]["id"] == str(source_id_for(chapter))
    assert rows["ingestion_job"]["input_count"] == 100
    assert rows["ingestion_job"]["metadata"]["generated_questions_are_original"] is True


def test_scraped_html_parser_keeps_provenance_and_review_gate():
    html = """
    <div class="sseo_faqcont">
      <div class="sseo_faqtitle"><h4>Q1. Which process helps bread rise?</h4></div>
      <div class="answer_option_list_main">
        <div class="sseo_answeroption"><span>[A].</span> Freezing</div>
        <div class="sseo_answeroption"><span>[B].</span> Fermentation</div>
        <div class="sseo_answeroption"><span>[C].</span> Filtering</div>
        <div class="sseo_answeroption"><span>[D].</span> Melting</div>
      </div>
      <div class="sseo_correct_option">Answer: Option B</div>
      <div class="sseo_answer_detail"><strong>Explanation:</strong><p>Yeast fermentation releases gas.</p></div>
    </div>
    """
    candidate = parse_tiwari_html(html, CHAPTERS[1], "https://example.test/chapter-2")[0]

    assert candidate.correct_option_id == "B"
    assert candidate.review_status == "review"
    assert candidate.source_question_id == "tiwari:class8-science:ch02:q01"
    assert candidate.answer_spec["verification"] == "pending_subject_review"


def test_scraped_pdf_parser_uses_shared_assertion_choices():
    text = """
    SECTION – A
    1. Assertion (A): A cell can become dead.
    Reason (R): Its chemicals are used up.
    SECTION – B
    """
    parsed = _parse_pdf_questions(text)

    assert parsed[1][0].startswith("Assertion (A)")
    assert [option[0] for option in parsed[1][1]] == ["A", "B", "C", "D"]


def test_source_item_model_retains_non_mcq_answers_and_provenance():
    item = QuestionItemCandidate(
        chapter_id=CHAPTERS[1].id,
        concept_id=uuid4(),
        question_type="numerical",
        question_text="An object has a mass of 400 g and a volume of 40 cm³. What is its density?",
        answer_spec={"expected_unit": "g/cm³", "has_answer_key": False},
        source_url="https://ncert.nic.in/textbook/pdf/hecu102.pdf",
        source_locator="Official NCERT PDF page 25, Keep the curiosity alive, Question 7",
        source_question_id="ncert:class8-science:ch02:keep:q07",
    )

    report = validate_question_items([item])
    row = item.to_db_row(CURRICULUM_VERSION_ID)

    assert report["question_types"] == {"numerical": 1}
    assert row["question_type"] == "numerical"
    assert row["source_url"].endswith("hecu102.pdf")
    assert row["answer_spec"]["expected_unit"] == "g/cm³"


def test_official_inline_options_are_preserved_as_structured_options():
    stem, options = _inline_options(
        "What is the density of the object? (i) 2 g/cm³ (ii) 4 g/cm³ (iii) 8 g/cm³ (iv) 40 g/cm³"
    )

    assert stem == "What is the density of the object?"
    assert [option["id"] for option in options] == ["A", "B", "C", "D"]
    assert options[1]["text"] == "4 g/cm³"


def test_duplicate_source_items_merge_answer_key_provenance():
    base = QuestionItemCandidate(
        chapter_id=CHAPTERS[1].id,
        concept_id=uuid4(),
        question_type="mcq_single",
        question_text="Which process helps bread rise in this experiment?",
        options=[
            {"id": "A", "text": "Freezing"},
            {"id": "B", "text": "Fermentation"},
            {"id": "C", "text": "Filtering"},
            {"id": "D", "text": "Melting"},
        ],
        answer_spec={"has_answer_key": False},
        source_url="https://ncert.nic.in/textbook/pdf/hecu102.pdf",
        source_question_id="ncert:class8-science:ch02:keep:q01",
    )
    answer_key = base.model_copy(
        update={
            "source_url": "https://example.test/answers.pdf",
            "source_question_id": "paper:q01",
            "answer_spec": {"has_answer_key": True, "correct_option_id": "B"},
        }
    )

    merged = _deduplicate_items([base, answer_key])

    assert len(merged) == 1
    assert merged[0].answer_spec["has_answer_key"] is True
    assert merged[0].answer_spec["correct_option_id"] == "B"
    assert merged[0].answer_spec["additional_source_records"][0]["source_url"] == "https://example.test/answers.pdf"


def test_previous_chapter_adaptation_retargets_content_and_keeps_provenance():
    previous_spec = CHAPTERS[1]
    target_spec = CHAPTERS[2]
    previous_question = build_pack(previous_spec)[0]

    adapted = build_adapted_candidates(
        target_spec,
        previous_spec,
        [previous_question],
        current_count=1,
        minimum_count=3,
        existing_hashes=set(),
    )

    assert len(adapted) == 2
    assert all(question.chapter_id == target_spec.id for question in adapted)
    assert all(question.review_status == "review" for question in adapted)
    assert all(question.generation_spec["previous_chapter_sequence"] == previous_spec.sequence_number for question in adapted)
    assert all(question.answer_spec["adaptation"] == "previous_chapter_structure_to_current_topic" for question in adapted)
