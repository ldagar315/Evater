"""Read-only editorial pack checks. Never connect to or import into a database."""
import ast
import json
import math
from collections import Counter
from pathlib import Path
from uuid import uuid5
from backend.etl.models import QuestionCandidate
from backend.etl.class8_maths_catalog import get_chapter as maths, CURRICULUM_VERSION_ID as MV
from backend.etl.class8_science_catalog import get_chapter as science, CURRICULUM_VERSION_ID as SV
ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / 'content/reviewed/class8-reasoning2-2026-11-17/questions.json'

def load():
    return json.loads(PACK.read_text())

def topic_slugs(module, chapter):
    tree = ast.parse((ROOT / f'backend/etl/{module}.py').read_text())
    topics = next(n.value for n in tree.body if isinstance(n, ast.AnnAssign) and getattr(n.target, 'id', '') == 'TOPICS')
    group = next(v for k, v in zip(topics.keys, topics.values) if ast.literal_eval(k) == chapter)
    return {ast.literal_eval(call.args[0]) for call in group.elts}

def arithmetic(node):
    if isinstance(node, ast.Expression): return arithmetic(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub): return -arithmetic(node.operand)
    if isinstance(node, ast.BinOp):
        a, b = arithmetic(node.left), arithmetic(node.right)
        if isinstance(node.op, ast.Add): return a + b
        if isinstance(node.op, ast.Sub): return a - b
        if isinstance(node.op, ast.Mult): return a * b
        if isinstance(node.op, ast.Div): return a / b
        if isinstance(node.op, ast.Pow): return a ** b
    raise ValueError('Only literal arithmetic is allowed')

def test_pack_is_review_only_with_balanced_keys_and_distinct_topics():
    p = load()
    assert p['status'] == 'review_only_no_production_import'
    assert p['publication_approved'] is False
    rows = p['questions']
    assert len(rows) == 24
    assert len({r['draft_id'] for r in rows}) == 24
    assert len({r['candidate']['question_text'].casefold() for r in rows}) == 24
    assert Counter(r['candidate']['correct_option_id'] for r in rows) == dict(A=6, B=6, C=6, D=6)
    assert Counter(r['subject'] for r in rows) == {'Mathematics': 12, 'Science': 12}
    assert len({(r['subject'], r['chapter_number'], r['topic_slug']) for r in rows}) == 24

def test_each_candidate_validates_and_maps_to_the_existing_repository_catalog():
    for r in load()['questions']:
        q = QuestionCandidate.model_validate(r['candidate'])
        is_maths = r['subject'] == 'Mathematics'
        spec = (maths if is_maths else science)(r['chapter_number'])
        assert q.chapter_id == spec.id
        assert q.concept_id == uuid5(spec.id, 'concept:' + r['topic_slug'])
        assert r['topic_slug'] in topic_slugs('class8_maths_topics' if is_maths else 'generate_class8_science_packs', r['chapter_number'])
        assert r['curriculum_version_id'] == str(MV if is_maths else SV)
        assert q.source_url == spec.source_url
        assert q.review_status == 'review'
        assert q.to_db_row(MV if is_maths else SV)['status'] == 'review'
        assert q.source_tier == 'manual'
        assert q.license_status == 'original_authored'
        assert r['mapping_status'] == 'repository_catalog_mapping_not_live_DB_verified'

def test_all_annotated_arithmetic_and_extra_identity_geometry_checks():
    checks = [r['arithmetic_check'] for r in load()['questions'] if r['arithmetic_check']]
    assert len(checks) == 9
    for c in checks:
        assert math.isclose(arithmetic(ast.parse(c['expression'], mode='eval')), c['expected'], rel_tol=1e-10)
    for a,b,c in [(2,3,7),(0,0,0),(5,11,8)]:
        assert a+2*(b+2)+c == a+2*b+c+4
    assert 12**2+16**2 == 20**2

def test_stems_are_not_existing_static_paper_or_reviewed_batch_stems():
    previous = json.loads((ROOT / 'content/reviewed/class8-2026-11-14/questions.json').read_text())
    stems = {q['question_text'].strip().casefold() for q in previous['questions']}
    for p in json.loads((ROOT / 'content/sample-papers/papers.json').read_text()):
        stems.update(q[0].strip().casefold() for q in p['objective'] + p['subjective'])
    for prior_path in [ROOT / 'content/reviewed/class8-reasoning-2026-11-17/questions.json']:
        prior = json.loads(prior_path.read_text())
        stems.update(q['candidate']['question_text'].strip().casefold() for q in prior['questions'])
        old_chapters = {(q['subject'],q['chapter_number']) for q in prior['questions']}
        assert not old_chapters.intersection((q['subject'],q['chapter_number']) for q in load()['questions'])
    for name in ['mathematics','science']:
        path = ROOT / f'content/papers/class8-mixed-2026-11-17/{name}.json'
        if path.exists():
            stems.update(q['stem'].strip().casefold() for q in json.loads(path.read_text())['questions'])
    assert not stems.intersection(r['candidate']['question_text'].strip().casefold() for r in load()['questions'])
