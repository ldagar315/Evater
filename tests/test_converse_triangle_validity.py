"""A non-right triangle must still be a possible triangle."""
import random
import re
from dataclasses import replace

import pytest
from backend.etl import author_class8_maths as author


@pytest.mark.parametrize('sides', [(2,2,15),(2,20,25),(5,11,18),(4,10,23),(3,4,7),(0,4,5)])
def test_impossible_converse_triangle_is_rejected(sides):
    a,b,c=sides
    sample=author.gen_converse_check(random.Random(0),9)
    item=replace(sample,question_text=f'A triangle has sides {a} cm, {b} cm, {c} cm. Is it right-angled?',key='No, it is not right-angled')
    assert not author._verify_converse_check(item)


@pytest.mark.parametrize('sides', [(2,2,15),(2,20,25),(5,11,18),(4,10,23),(3,4,7),(0,4,5)])
def test_impossible_true_false_triangle_is_rejected(sides):
    a,b,c=sides
    with pytest.raises(ValueError,match='do not form a triangle'):
        author.v_tf_pyth_triple(f'True or false: a triangle with sides {a} cm, {b} cm, and {c} cm is right-angled.')


@pytest.mark.parametrize('seed',range(200))
def test_both_generators_always_describe_valid_triangles(seed):
    for generator in (author.gen_converse_check,author.gen_tf_pyth_triple):
        item=generator(random.Random(seed),9)
        m=re.search(r'sides (\d+) cm, (\d+) cm, (?:and )?(\d+) cm',item.question_text)
        a,b,c=sorted(map(int,m.groups()))
        assert 0<a and a+b>c
        right=a*a+b*b==c*c
        if item.family=='converse_check':
            assert author._verify_converse_check(item)
            assert item.key==('Yes, it is right-angled' if right else 'No, it is not right-angled')
        else:
            assert author.v_tf_pyth_triple(item.question_text)==item.key==('True' if right else 'False')
