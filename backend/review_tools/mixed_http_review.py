"""Loopback-only review harness. Never imported by app.main; no DB client.

Inputs are supplied in memory by the reviewer. A snapshot is not live evidence.
Synthetic fixtures must use synthetic-* IDs; they cannot stand in for real data.
"""
from datetime import datetime, timezone
import copy
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from app.mixed_sample_paper import MixedSamplePaper, review_router

class NotReady(Exception):
    def __init__(self, issues):
        self.issues = sorted(set(issues))

def parse_json(value):
    return json.loads(value) if isinstance(value, str) else value

def inspect_objective(expected, actual):
    p = expected['position']
    prefix = f'Q{p}: '
    if actual is None:
        return [prefix + 'bank row missing']
    issues = []
    for field in ['id', 'content_hash', 'question_text', 'question_type', 'chapter_id']:
        if actual.get(field) != expected.get(field) or not expected.get(field):
            issues.append(prefix + field + ' differs from reviewed content')
    if actual.get('status') != 'published':
        issues.append(prefix + 'bank row is not published')
    try:
        options = parse_json(actual.get('options_json'))
        spec = parse_json(actual.get('answer_spec'))
        if options != expected['options']:
            issues.append(prefix + 'options differ from reviewed content')
        keys = [v for v in [actual.get('correct_option_id'), spec.get('correct_option_id')] if v is not None]
        # source_answer may be text/units rather than an option label. Compare it
        # as a key only when it is explicitly a label; do not invent equivalence.
        labels = {o.get('id') for o in options if isinstance(o, dict)}
        if isinstance(spec.get('source_answer'), str) and spec['source_answer'] in labels:
            keys.append(spec['source_answer'])
        if not keys:
            keys = [spec.get('expected_value', spec.get('accepted_answers'))]
        if any(k != expected['answer'] for k in keys) or not keys or keys[0] is None:
            issues.append(prefix + 'bank answer conflicts with reviewed answer')
    except (ValueError, TypeError, AttributeError, KeyError):
        issues.append(prefix + 'bank JSON is invalid')
    return issues

def build_payload(draft, snapshot, now=None):
    """All checks run before returning any paper content; never mutate inputs."""
    now = now or datetime.now(timezone.utc)
    issues = []
    synthetic = snapshot.get('synthetic') is True
    if synthetic:
        if not draft['title'].startswith('SYNTHETIC') or any(not q['id'].startswith('synthetic-') for q in draft['objective']):
            issues.append('synthetic fixture identity is invalid')
    else:
        if snapshot.get('verified_live') is not True:
            issues.append('current bank state is not verified live')
        try:
            checked = datetime.fromisoformat(snapshot['checked_at'].replace('Z', '+00:00'))
            age = (now - checked).total_seconds()
            if age < 0 or age > 900:
                issues.append('bank check is stale or future-dated')
        except (KeyError, ValueError, TypeError):
            issues.append('bank check time is missing or invalid')
    issues += draft.get('release_holds', [])
    rows = {}
    for row in snapshot['rows']:
        if row['id'] in rows:
            issues.append('duplicate bank row identity')
        rows[row['id']] = row
    objectives = []
    for expected in draft['objective']:
        issues += inspect_objective(expected, rows.get(expected['id']))
        objectives.append({
            'position': expected['position'], 'bank_id': expected['id'],
            'content_hash': expected['content_hash'], 'prompt': expected['question_text'],
            'options': [{'label': o['id'], 'text': o['text']} for o in expected['options']],
            'answer': str(expected['answer']),
        })
    payload = {k: copy.deepcopy(draft[k]) for k in ['format', 'status', 'grade', 'title', 'subjective']}
    payload['objective'] = objectives
    try:
        result = MixedSamplePaper.model_validate(payload).model_dump()
    except ValidationError as exc:
        # Only paths/types escape, never question/answer input values.
        issues += ['contract: ' + '.'.join(map(str, e['loc'])) + ' ' + e['type'] for e in exc.errors()]
        result = None
    if issues:
        raise NotReady(issues)
    return result

def create_app(papers, snapshots, *, loader=None):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    @app.middleware('http')
    async def local_only(request: Request, call_next):
        if request.client is None or request.client.host not in ('127.0.0.1', '::1', 'testclient'):
            return JSONResponse({'detail': 'Local review only.'}, status_code=403)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Robots-Tag'] = 'noindex, nofollow'
        return response
    @app.exception_handler(NotReady)
    async def not_ready(request, exc):
        return JSONResponse({'detail': {'code': 'review_not_ready', 'issues': exc.issues}}, status_code=503)
    def load(paper_id):
        if paper_id not in papers:
            raise KeyError(paper_id)
        try:
            return loader(paper_id) if loader else build_payload(papers[paper_id], snapshots[paper_id])
        except NotReady:
            raise
        except Exception:
            # Private inputs and exception internals do not go to the browser.
            raise NotReady(['review loader failed; no paper content returned'])
    app.include_router(review_router(load))
    return app

def synthetic_inputs():
    rows = []
    objectives = []
    for i in range(1, 11):
        row = {'id': f'synthetic-{i}', 'content_hash': f'synthetic-hash-{i}',
               'chapter_id': 'synthetic-chapter', 'question_type': 'mcq_single',
               'question_text': f'Synthetic wiring question {i}. Choose option A.',
               'options_json': [{'id': 'A', 'text': 'Synthetic correct option'}, {'id': 'B', 'text': 'Synthetic alternative'}],
               'correct_option_id': 'A', 'answer_spec': {'source_answer': 'A'}, 'status': 'published'}
        rows.append(row)
        objectives.append({**{k: row[k] for k in ['id', 'content_hash', 'chapter_id', 'question_type', 'question_text']},
                           'position': i, 'options': copy.deepcopy(row['options_json']), 'answer': 'A'})
    draft = {'format': '10-objective-5-subjective-v1', 'status': 'review-only', 'grade': 8,
             'title': 'SYNTHETIC HTTP WIRING CONTROL - NOT STUDY CONTENT', 'objective': objectives,
             'subjective': [{'position': i, 'prompt': f'Synthetic written practice question {i}.', 'marks': 2,
                             'model_answer': 'Synthetic answer for wiring only.', 'marking_criteria': ['Synthetic criterion 1', 'Synthetic criterion 2'],
                             'source_url': 'https://example.invalid/synthetic-source.pdf', 'source_page': 1} for i in range(11, 16)]}
    return draft, {'synthetic': True, 'rows': rows}
