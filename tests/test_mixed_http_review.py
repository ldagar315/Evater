import copy
from datetime import datetime, timezone, timedelta
import pytest
from fastapi.testclient import TestClient
from review_tools.mixed_http_review import create_app, synthetic_inputs, build_payload, inspect_objective, NotReady

@pytest.fixture
def inputs():
    return synthetic_inputs()

def client(d, s, **kwargs):
    return TestClient(create_app({'synthetic': d}, {'synthetic': s}, **kwargs))

def test_actual_router_validated_response(inputs):
    d,s=inputs
    result=client(d,s).get('/review/mixed-sample-papers/synthetic')
    assert result.status_code==200
    assert [q['position'] for q in result.json()['objective']]==list(range(1,11))
    assert [q['position'] for q in result.json()['subjective']]==list(range(11,16))
    assert result.headers['cache-control']=='no-store'
    assert result.headers['x-robots-tag']=='noindex, nofollow'

@pytest.mark.parametrize('field,value', [('id','wrong'),('content_hash','wrong'),('question_text','changed'),('question_type','numerical'),('chapter_id','wrong'),('status','retired'),('correct_option_id','B'),('options_json',[{'id':'A','text':'changed'}]),('answer_spec',{'source_answer':'B'}),('answer_spec','bad-json')])
def test_changed_bank_fails_without_content(inputs, field, value):
    d,s=inputs;s['rows'][0][field]=value
    r=client(d,s).get('/review/mixed-sample-papers/synthetic')
    assert r.status_code==503
    assert 'objective' not in r.json()
    assert 'Synthetic correct option' not in r.text

@pytest.mark.parametrize('mutation', [
    lambda d,s:s['rows'].pop(),
    lambda d,s:s['rows'].append(copy.deepcopy(s['rows'][0])),
    lambda d,s:d['objective'].pop(),
    lambda d,s:d['subjective'].pop(),
    lambda d,s:d['subjective'][0].pop('source_page'),
    lambda d,s:d['subjective'][0].update(source_page=0),
    lambda d,s:d['subjective'][0].update(model_answer=''),
    lambda d,s:d['subjective'][0].update(marking_criteria=[]),
    lambda d,s:d['subjective'][0].update(source_url='http://example.invalid'),
    lambda d,s:d['subjective'][0].update(position=12),
    lambda d,s:d['subjective'].append(copy.deepcopy(d['subjective'][0])),
    lambda d,s:d.update(status='published'),
    lambda d,s:d.update(title='Not a synthetic label'),
    lambda d,s:d.update(release_holds=['teacher review is unresolved']),
])
def test_contract_and_readiness_negatives(inputs,mutation):
    d,s=inputs;mutation(d,s)
    assert client(d,s).get('/review/mixed-sample-papers/synthetic').status_code==503

def test_unknown_404(inputs):
    d,s=inputs
    assert client(d,s).get('/review/mixed-sample-papers/unknown').status_code==404

def test_loader_failure_sanitized(inputs):
    d,s=inputs
    def broken(_):raise RuntimeError('SECRET_INTERNAL_VALUE')
    r=client(d,s,loader=broken).get('/review/mixed-sample-papers/synthetic')
    assert r.status_code==503 and 'SECRET_INTERNAL_VALUE' not in r.text

def test_default_actual_router_is_still_unconfigured():
    from app.mixed_sample_paper import review_router
    from fastapi import FastAPI
    app=FastAPI();app.include_router(review_router())
    assert TestClient(app).get('/review/mixed-sample-papers/x').status_code==503

def test_router_response_model_still_rejects_invalid_loader(inputs):
    d,s=inputs
    from fastapi.exceptions import ResponseValidationError
    with pytest.raises(ResponseValidationError):
        client(d,s,loader=lambda _: {'bad':'payload'}).get('/review/mixed-sample-papers/synthetic')

def test_pure_function_no_input_mutation(inputs):
    d,s=inputs;before=copy.deepcopy(inputs)
    build_payload(d,s)
    assert (d,s)==before

@pytest.mark.parametrize('minutes,verified', [(16,True),(-1,True),(0,False)])
def test_real_snapshot_freshness(inputs,minutes,verified):
    d,s=inputs;s['synthetic']=False;s['verified_live']=verified
    now=datetime.now(timezone.utc);s['checked_at']=(now-timedelta(minutes=minutes)).isoformat()
    with pytest.raises(NotReady):build_payload(d,s,now)

def test_real_positive_only_with_verified_fresh_snapshot(inputs):
    # Synthetic rows here exercise the timestamp branch only, not real readiness.
    d,s=inputs;s['synthetic']=False;s['verified_live']=True;s['checked_at']=datetime.now(timezone.utc).isoformat()
    assert build_payload(d,s)['status']=='review-only'

def test_non_loopback_client_refused(inputs):
    d,s=inputs
    app=create_app({'synthetic':d},{'synthetic':s})
    assert TestClient(app,client=('198.51.100.8',1234)).get('/review/mixed-sample-papers/synthetic').status_code==403

def test_option_text_source_answer_not_mistaken_for_wrong_label(inputs):
    d,s=inputs;s['rows'][0]['answer_spec']['source_answer']='Synthetic correct option'
    assert build_payload(d,s)['objective'][0]['answer']=='A'

def test_snapshot_light_conflict_is_not_silently_corrected():
    import os,json
    path=os.environ.get('HELD_REVIEW_INPUTS')
    if not path:pytest.skip('Private held input is deliberately not in the repository')
    data=json.load(open(path));d=data['papers']['light'];s=data['snapshots']['light']
    before=copy.deepcopy(data)
    r=client(d,s).get('/review/mixed-sample-papers/synthetic')
    assert r.status_code==503
    assert 'Q3: bank answer conflicts with reviewed answer' in r.json()['detail']['issues']
    assert data==before

def test_browser_iso_z_timestamp_accepted(inputs):
    d,s=inputs;s['synthetic']=False;s['verified_live']=True
    s['checked_at']=datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
    assert build_payload(d,s)['status']=='review-only'
