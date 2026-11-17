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
        pdf = ROOT / 'frontend/public' / ('class-8-' + short + '-mixed-chapter-practice-paper.pdf')
        text = subprocess.check_output(['pdftotext', str(pdf), '-']).decode()
        pages = [''.join(page.split()) for page in text.split('\f')]
        assert 'EVATER DRAFT' not in text and 'catalog' not in text and 'review draft' not in text
        for q in source['questions']:
            needle = ''.join((str(q['number']) + '. ' + q['stem']).split())
            matches = [p for p in pages if needle in p]
            assert len(matches) == 1, (subject, q['number'], 'stem/data split')
            for option in q.get('options', []):
                assert ''.join((option['id'] + '. ' + option['text']).split()) in matches[0]
