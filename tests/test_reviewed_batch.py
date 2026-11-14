import hashlib
import json
from pathlib import Path
import shutil
import pytest
from backend.etl.reviewed_batch import ReviewedBatchError, load_reviewed_batch, prepare_review_rows

SOURCE = Path(__file__).resolve().parents[1] / 'content/reviewed/class8-2026-11-14'

@pytest.fixture
def batch(tmp_path):
    target = tmp_path / 'batch'
    shutil.copytree(SOURCE, target)
    return target


def edit_questions(batch, change, *, update_hash=False):
    file = batch / 'questions.json'
    payload = json.loads(file.read_text())
    change(payload['questions'])
    file.write_text(json.dumps(payload))
    if update_hash:
        manifest_file = batch / 'manifest.json'
        manifest = json.loads(manifest_file.read_text())
        manifest['questions_sha256'] = hashlib.sha256(file.read_bytes()).hexdigest()
        manifest_file.write_text(json.dumps(manifest))


def test_exact_reviewed80_only_prepare_review_rows(batch):
    manifest, questions = load_reviewed_batch(batch)
    rows = prepare_review_rows(batch)
    assert manifest['owner_publication_approval'] == 'pending'
    assert len(questions) == len(rows) == 80
    assert len({r['content_hash'] for r in rows}) == 80
    assert all(r['status'] == 'review' and r['quality_score'] is None for r in rows)
    assert all(r['source_tier'] == 'manual' and r['source_attribution_display'] is None for r in rows)
    assert len(manifest['guides']) == 3


def test_changed_wording_fails_hash(batch):
    edit_questions(batch, lambda qs: qs[0].update(question_text='Changed question text'))
    with pytest.raises(ReviewedBatchError, match='proposal changed'):
        load_reviewed_batch(batch)

@pytest.mark.parametrize('change', [
    lambda qs: qs.pop(),
    lambda qs: qs.append(qs[0]),
    lambda qs: qs[0].update(draft_id='reserve-or-unapproved-id'),
])
def test_missing_extra_reserve_ids_fail_even_with_new_hash(batch, change):
    edit_questions(batch, change, update_hash=True)
    with pytest.raises(ReviewedBatchError, match='exact reviewed'):
        load_reviewed_batch(batch)


def test_difficulty_relabeling_fails(batch):
    edit_questions(batch, lambda qs: qs[0].update(suggested_difficulty='hard'), update_hash=True)
    with pytest.raises(ReviewedBatchError, match='Difficulty'):
        load_reviewed_batch(batch)


def test_wrong_chapter_fails(batch):
    edit_questions(batch, lambda qs: qs[0].update(chapter_id='00000000-0000-4000-8000-000000000000'), update_hash=True)
    with pytest.raises(ReviewedBatchError, match='mapping'):
        load_reviewed_batch(batch)


def test_selection_boundary_fails(batch):
    edit_questions(batch, lambda qs: qs[0].update(release_recommendation='reserve_variant_not_first_release'), update_hash=True)
    with pytest.raises(ReviewedBatchError, match='Reserve'):
        load_reviewed_batch(batch)


def test_changed_guide_fails(batch):
    guide = next(batch.glob('*.md'))
    guide.write_text(guide.read_text() + '\nUnreviewed addition.')
    with pytest.raises(ReviewedBatchError, match='Guide copy'):
        load_reviewed_batch(batch)


def test_pending_owner_decision_cannot_change_output_to_published(batch):
    file = batch / 'manifest.json'
    manifest = json.loads(file.read_text())
    manifest['owner_publication_approval'] = 'claimed-approved'
    file.write_text(json.dumps(manifest))
    assert all(row['status'] == 'review' for row in prepare_review_rows(batch))

@pytest.mark.parametrize('change', [
    lambda qs: qs[0].update(correct_option_id='Z'),
    lambda qs: qs[0]['options'][0].update(text=qs[0]['options'][1]['text']),
])
def test_invalid_mcq_still_fails_existing_model(batch, change):
    from pydantic import ValidationError
    edit_questions(batch, change, update_hash=True)
    with pytest.raises(ValidationError):
        prepare_review_rows(batch)
