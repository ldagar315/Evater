"""Synthetic contract fixtures, not bank/source verification."""
import copy
import pytest
from fastapi import FastAPI
from fastapi.exceptions import ResponseValidationError
from fastapi.testclient import TestClient
from app.mixed_sample_paper import review_router,MixedSamplePaper

def fixture():
 return {'format':'10-objective-5-subjective-v1','status':'review-only','grade':8,'title':'SYNTHETIC CONTRACT FIXTURE', 'objective':[{'position':i,'bank_id':f'fixture-{i}','content_hash':f'synthetic-{i}','prompt':f'Fixture question {i}','options':[{'label':'A','text':'Fixture option'}],'answer':'Fixture answer'} for i in range(1,11)],'subjective':[{'position':i,'prompt':f'Fixture subjective {i}','marks':2,'model_answer':'Synthetic model answer','marking_criteria':['Synthetic criterion'],'source_url':'https://example.invalid/source.pdf','source_page':1} for i in range(11,16)]}

def test_actual_router_10_plus_5():
 app=FastAPI();router=review_router(lambda _:fixture());app.include_router(router);r=TestClient(app).get('/review/mixed-sample-papers/example');assert r.status_code==200
 d=r.json();assert len(d['objective'])==10 and len(d['subjective'])==5
 assert router.routes[0].response_model is MixedSamplePaper

def test_default_fail_closed():
 app=FastAPI();app.include_router(review_router());assert TestClient(app).get('/review/mixed-sample-papers/x').status_code==503

@pytest.mark.parametrize('bad',[lambda d:d['objective'].pop(),lambda d:d['subjective'].pop(),lambda d:d['subjective'][0].update(source_url='http://example.invalid/x'),lambda d:d['subjective'][0].update(source_page=0),lambda d:d['subjective'][0].update(model_answer=''),lambda d:d['subjective'][0].update(marking_criteria=[]),lambda d:d['objective'][0].update(position=2),lambda d:d['objective'][0].update(bank_id='fixture-2'),lambda d:d.update(status='published'),lambda d:d.update(unapproved='field')])
def test_bad_payload_rejected_at_actual_router(bad):
 d=fixture();bad(d);app=FastAPI();app.include_router(review_router(lambda _:d))
 with pytest.raises(ResponseValidationError):TestClient(app).get('/review/mixed-sample-papers/x')
