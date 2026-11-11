import copy
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.routers import sample_papers_router as r
from types import SimpleNamespace
app=FastAPI();app.include_router(r.router,prefix='/api/v1');client=TestClient(app)
chapter={'id':'chapter-fixture','sequence_number':1,'title':'Fixture Chapter','description':'Offline only'}
questions=[{'id':str(i),'chapter_id':'chapter-fixture','concept_id':'concept-fixture','question_type':'mcq_single','question_text':'Fixture scientific question '+str(i),'options_json':[{'id':x,'text':x+' answer'} for x in 'ABCD'],'cognitive_level':'understand','difficulty':'easy','correct_option_id':'A','answer_spec':{},'content_hash':str(i)} for i in range(15)]
class Query:
 def __init__(self,table):self.name=table
 def select(self,*a):return self
 def eq(self,*a):return self
 def in_(self,*a):return self
 def order(self,*a):return self
 def execute(self):
  return SimpleNamespace(data=([{'id':'curriculum-fixture','subject':'Science'}] if self.name=='curriculum_versions' else [chapter] if self.name=='chapters' else questions))
class FakeDB:
 def table(self,name):return Query(name)
r.create_supabase_service_client=lambda:FakeDB()
def reset():r._response_cache.clear();r._rate_windows.clear()
def test_success_key_is_public_only_here():
 reset();res=client.get('/api/v1/sample-papers/science/fixture-chapter');assert res.status_code==200;d=res.json();assert d['question_count']==15;assert d['questions'][0]['answer_summary']=='A. A answer';assert set(d['questions'][0])=={'position','id','question_type','question_text','options','answer_summary'}
def test_index():
 reset();res=client.get('/api/v1/sample-papers/index');assert res.status_code==200;assert res.json()['subjects'][0]['chapters'][0]['published_count']==15
def test_grade_bounds():
 for grade in [5,11]:assert client.get('/api/v1/sample-papers/index?grade='+str(grade)).status_code==422
def test_unknowns_fail():
 reset();assert client.get('/api/v1/sample-papers/unknown/fixture-chapter').status_code==404;assert client.get('/api/v1/sample-papers/science/unknown').status_code==404
def test_insufficient_questions():
 reset();saved=questions.pop();assert client.get('/api/v1/sample-papers/science/fixture-chapter').status_code==404;questions.append(saved)
def test_rate_cap():
 reset()
 for _ in range(60):assert client.get('/api/v1/sample-papers/index').status_code==200
 assert client.get('/api/v1/sample-papers/index').status_code==429

import pytest
from fastapi.exceptions import ResponseValidationError
from app.sample_paper_models import SamplePaper,SamplePapersIndex

def test_router_models_are_concrete():
 routes={route.path:route.response_model for route in r.router.routes}
 assert routes['/sample-papers/index'] is SamplePapersIndex
 assert routes['/sample-papers/{subject}/{chapter_slug}'] is SamplePaper

def prime():
 reset();res=client.get('/api/v1/sample-papers/science/fixture-chapter');assert res.status_code==200
 return copy.deepcopy(res.json())

@pytest.mark.parametrize('damage',[lambda d:d.update(secret='not allowed'),lambda d:d['questions'][0].update(correct_option_id='B'),lambda d:d['questions'][0].update(answer_summary=''),lambda d:d['questions'][0].update(position=2),lambda d:d['questions'][0].update(question_type='invented'),lambda d:d['questions'].pop(),lambda d:d['questions'][0].update(id=d['questions'][1]['id']),lambda d:d.update(grade='8')])
def test_actual_fastapi_route_rejects_bad_cached_wire_payload(damage):
 d=prime();damage(d);r._cache_set('paper:8:science:fixture-chapter',d)
 with pytest.raises(ResponseValidationError):client.get('/api/v1/sample-papers/science/fixture-chapter')

def test_actual_index_rejects_bad_field():
 reset();assert client.get('/api/v1/sample-papers/index').status_code==200
 d=copy.deepcopy(r._cache_get('index:8'));d['subjects'][0]['chapters'][0]['published_count']='15';r._cache_set('index:8',d)
 with pytest.raises(ResponseValidationError):client.get('/api/v1/sample-papers/index')
