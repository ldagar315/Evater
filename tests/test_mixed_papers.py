"""Dependency-light paper structure and existing-PDF byte regressions."""
import hashlib
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / 'content/papers/class8-mixed-2026-11-17'


def test_paper_structure_and_arithmetic():
    runpy.run_path(str(PACK / 'validate.py'))


def test_existing_pdfs_unchanged():
    hashes = json.loads((PACK / 'existing-chapter-pdf-sha256.json').read_text())
    assert len(hashes) == 27
    for name, expected in hashes.items():
        assert hashlib.sha256((ROOT / 'frontend/public' / name).read_bytes()).hexdigest() == expected


def test_released_pdfs_and_editable_source_agree():
    # This test deliberately uses lossless text extraction when available.
    import shutil
    import subprocess
    if not shutil.which('pdftotext'):
        import pytest
        pytest.skip('pdftotext not installed')
    for subject, short in [('mathematics', 'maths'), ('science', 'science')]:
        source = json.loads((PACK / (subject + '.json')).read_text())
        pdf = ROOT / 'frontend/public' / (('1-' if short == 'maths' else '2-') + 'class-8-' + short + '-mixed-chapter-practice-paper.pdf')
        text = subprocess.check_output(['pdftotext', str(pdf), '-']).decode()
        pages = [''.join(page.split()) for page in text.split('\f')]
        assert 'EVATER DRAFT' not in text and 'catalog' not in text and 'review draft' not in text
        for q in source['questions']:
            needle = ''.join((str(q['number']) + '. ' + q['stem']).split())
            matches = [p for p in pages if needle in p]
            assert len(matches) == 1, (subject, q['number'], 'stem/data split')
            for option in q.get('options', []):
                assert ''.join((option['id'] + '. ' + option['text']).split()) in matches[0]


def test_science_force_prompts_stay_with_chapter_level():
    paper = json.loads((PACK / 'science.json').read_text())
    q12, q23 = paper['questions'][11], paper['questions'][22]
    assert q12['chapter'] == q23['chapter'] == 5
    assert q12['topic'] == 'force-effects'
    assert q23['topic'] == 'magnetic-and-friction-forces'
    for q in [q12,q23]:
        assert not any(term in (q['stem'] + q['answer']).lower() for term in ['resultant','vector sum','unbalanced','velocity'])
    assert sum(item['marks'] for item in q23['marking']) == 5
