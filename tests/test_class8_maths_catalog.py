"""Tests for the Ganita Prakash Grade 8 Maths chapter catalog."""

from __future__ import annotations

from backend.etl.class8_maths_catalog import CHAPTERS, get_chapter


def test_fourteen_chapters_in_two_parts():
    assert len(CHAPTERS) == 14
    assert [c.sequence_number for c in CHAPTERS] == list(range(1, 15))
    assert [c.part for c in CHAPTERS] == [1] * 7 + [2] * 7
    assert [c.part_chapter_number for c in CHAPTERS] == list(range(1, 8)) * 2


def test_source_urls_map_parts_to_ncert_codes():
    assert get_chapter(1).source_url == "https://ncert.nic.in/textbook/pdf/hegp101.pdf"
    assert get_chapter(7).source_url == "https://ncert.nic.in/textbook/pdf/hegp107.pdf"
    assert get_chapter(8).source_url == "https://ncert.nic.in/textbook/pdf/hegp201.pdf"
    assert get_chapter(14).source_url == "https://ncert.nic.in/textbook/pdf/hegp207.pdf"


def test_chapter_ids_are_stable_and_unique():
    ids = {c.id for c in CHAPTERS}
    assert len(ids) == 14
    assert get_chapter(3).id == get_chapter(3).id
    assert get_chapter(3).id != get_chapter(4).id


def test_titles_match_printed_book():
    assert get_chapter(1).title == "A Square and A Cube"
    assert get_chapter(9).title == "The Baudhayana-Pythagoras Theorem"
    assert get_chapter(14).title == "Area"
