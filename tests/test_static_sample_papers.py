"""Offline checks for the static PDF release, not production availability."""
import hashlib
import json
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[1]
PAPERS = json.loads((ROOT / 'content/sample-papers/papers.json').read_text())
MANIFEST = json.loads((ROOT / 'content/sample-papers/pdf-manifest.json').read_text())

def test_complete_original_papers():
    assert len(PAPERS) == 27
    for paper in PAPERS:
        assert len(paper['objective']) == 10
        assert len(paper['subjective']) == 5
        assert sum(q[1] for q in paper['subjective']) == 12
        for prompt, options, key, explanation in paper['objective']:
            assert prompt and explanation and len(options) == len(set(options)) == 4
            assert key in 'ABCD'
        assert all(q[0] and q[2] and q[3] for q in paper['subjective'])

def test_pdf_bytes_and_library_destinations():
    library = (ROOT / 'frontend/src/data/samplePapers.ts').read_text()
    assert {m['filename'] for m in MANIFEST} == {p['filename'] for p in PAPERS}
    for item in MANIFEST:
        data = (ROOT / 'frontend/public' / item['filename']).read_bytes()
        assert data.startswith(b'%PDF-') and data.rstrip().endswith(b'%%EOF')
        assert hashlib.sha256(data).hexdigest() == item['sha256']
        assert len(data) == item['bytes']
        assert '/' + item['filename'] in library
    for page in ['LandingPage', 'HomePage']:
        assert 'SamplePaperDownloads' not in (ROOT / f'frontend/src/pages/{page}.tsx').read_text()

def test_maths_objective_answers_independently():
    questions = PAPERS[2]['objective']
    expected = ['15 cm', 'h²=u²+v²', '15', '6, 8, 10', '4√2 cm', '25', 'Hypotenuse', 'No, because 7²+10²≠12²', '13 m', 'Longer than each leg']
    assert math.hypot(9, 12) == 15
    assert math.sqrt(17**2 - 8**2) == 15
    assert math.hypot(7, 24) == 25
    assert math.hypot(5, 12) == 13
    assert 7**2 + 10**2 != 12**2 and 7 + 10 > 12
    for question, correct in zip(questions, expected):
        assert question[1][ord(question[2]) - 65] == correct

def test_maths_subjective_computations_and_bands():
    assert round(6 * math.sqrt(2), 2) == 8.49
    assert math.hypot(9, 12) == 15
    assert round(16 * math.sqrt(3), 2) == 27.71
    for a, b, c in [(9,12,15),(12,16,20),(10,24,26),(16,30,34),(14,48,50)]:
        assert a > 0 and b > 0 and a*a + b*b == c*c
    rule = PAPERS[2]['subjective'][3][3]
    assert 'Five without a check: 2 marks' in rule and 'Reordered duplicates count once' in rule

def test_science_safety_and_qualification():
    electricity = PAPERS[1]['subjective']
    assert 'does not uniquely identify' in electricity[0][2]
    assert 'do not use mains' in electricity[2][2]
    assert 'off/on comparison' in electricity[3][3]
    assert 'upstream' in electricity[4][2]
    assert 'one universal bend pattern is not required' in PAPERS[0]['subjective'][4][2]


def test_papers_public_route_and_banner_removal():
    app = (ROOT / 'frontend/src/App.tsx').read_text()
    assert '<Route path="/papers" element={<PapersPage />} />' in app
    assert "['/about', '/papers']" in app
    assert "'/papers'," in (ROOT / 'frontend/vite.config.ts').read_text()
    assert 'https://evater.xyz/papers/' in (ROOT / 'frontend/public/sitemap.xml').read_text()
    assert (ROOT / 'frontend/src/components/layout/Header.tsx').read_text().count('to="/papers/"') == 2
