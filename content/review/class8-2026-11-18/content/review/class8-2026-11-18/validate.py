"""Review-pack checks. Does not import, publish or call any external system."""
from pathlib import Path
import ast
import collections
import hashlib
import json
import operator

ROOT = Path(__file__).resolve().parent
EXPECTED = '84a088f64b8032ebf75065155ba22599734f69cb33030fca3b38a88f3756a173'
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
       ast.FloorDiv: operator.floordiv}


def arithmetic(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](arithmetic(node.left), arithmetic(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -arithmetic(node.operand)
    raise ValueError('Non-arithmetic expression')


def validate():
    raw = (ROOT / 'questions.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED, 'Accepted revision changed'
    pack = json.loads(raw)
    assert pack['publication_approved'] is False
    qs = pack['questions']
    assert len(qs) == 54
    chapters = collections.Counter((q['subject'], q['chapter_number']) for q in qs)
    assert len(chapters) == 27 and set(chapters.values()) == {2}
    assert {n for (s, n) in chapters if s == 'Mathematics'} == set(range(1, 15))
    assert {n for (s, n) in chapters if s == 'Science'} == set(range(1, 14))
    assert len({q['draft_id'] for q in qs}) == 54
    assert len({q['question_text'].strip().casefold() for q in qs}) == 54
    checks = 0
    for q in qs:
        assert q['review_state'] == 'review' and q['question_type'] == 'mcq'
        assert q['marks'] == 1
        assert [o['id'] for o in q['options']] == list('ABCD')
        assert len({o['text'].strip().casefold() for o in q['options']}) == 4
        assert q['correct_option_id'] in 'ABCD'
        assert all(q[k].strip() for k in ('explanation', 'question_text', 'private_source_fit_note', 'provenance'))
        if q['arithmetic_check']:
            check = q['arithmetic_check']
            value = arithmetic(ast.parse(check['expression'], mode='eval').body)
            assert abs(value - check['expected']) < 1e-8, q['draft_id']
            checks += 1
    assert checks == 26
    keys = collections.Counter(q['correct_option_id'] for q in qs)
    assert keys == {'A': 11, 'B': 17, 'C': 14, 'D': 12}
    print('PASS: accepted SHA256; 54 review MCQs; 27 chapters x2; 26 arithmetic checks; unique IDs/stems/options; keys A11 B17 C14 D12')
    print('Exact-stem uniqueness is not a semantic originality guarantee. Content/source-fit challenge is separate.')


if __name__ == '__main__':
    validate()
