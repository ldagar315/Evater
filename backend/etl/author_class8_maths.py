"""Author Evater-original Class 8 Maths MCQs with code-verified answers.

Owner decisions (2026-10-21):
- Maths questions may be created (rework pass). Types mcq_single / true_false /
  numerical are allowed; the publish path is mcq_single-only today, so this
  module authors mcq_single items.
- Every answer must be verified by code. Each generator emits the key from its
  parameters; an independent verifier re-derives the key by parsing the
  rendered question text. Items whose two paths disagree are dropped and
  reported, never staged.
- Provenance: evater_original content is tier "manual" with attribution
  "Evater original (authored and code-verified)". NCERT-close items keep
  ncert_attributed; this module never produces those.

Output: staging pack JSONs (same shape as scraped packs: "questions" +
"items": []) that seed_class8_maths can validate and publish.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid5

from .class8_maths_catalog import CHAPTERS

SOURCE_URL = "manual://evater/authored/class8-maths"
ATTRIBUTION = "Evater original (authored and code-verified)"


@dataclass
class AuthoredItem:
    family: str
    chapter: int
    topic_slug: str
    question_text: str
    key: str
    distractors: list[str]
    explanation: str
    difficulty: str
    skill_tags: list[str]
    params: dict = field(default_factory=dict)
    qtype: str = "mcq_single"  # mcq_single | true_false | numerical


# ---------------------------------------------------------------------------
# Verifiers: re-derive the key by parsing the rendered text, never params.
# ---------------------------------------------------------------------------

def _nums(text: str) -> list[float]:
    return [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", text)]


def verify_linear_eq(text: str) -> str:
    # "Solve for x: 4x + 7 = 35."
    a, b, c = _nums(text)
    x = (c - b) / a
    assert abs(x - round(x)) < 1e-9
    return str(int(round(x)))


def verify_substitute(text: str) -> str:
    # "If x = 3, what is the value of 4x + 5?"
    k, p, q = _nums(text)
    return str(int(p * k + q))


def verify_square_sum(text: str) -> str:
    # "What is (7 + 3)^2?"
    m = re.search(r"\((\d+) \+ (\d+)\)\^2", text)
    a, b = int(m.group(1)), int(m.group(2))
    return str((a + b) ** 2)


def verify_expand_product(text: str) -> str:
    # "Expand (x + 2)(x + 5)." -> x^2 + 7x + 10
    a, b = _nums(text)
    return f"x^2 + {int(a+b)}x + {int(a*b)}"


def verify_like_terms(text: str) -> str:
    # "Simplify: 3x + 4x - 2x."
    a, b, c = _nums(text)
    return f"{int(a+b-c)}x"


def verify_nth_term(text: str) -> str:
    # "A pattern follows the rule: term = 3n + 2. What is the 10th term?"
    a, b, k = _nums(text)
    return str(int(a * k + b))


def verify_consecutive(text: str) -> str:
    # "The sum of three consecutive whole numbers is 45. What is the smallest?"
    (s,) = _nums(text)
    assert s % 3 == 0
    return str(int(s // 3 - 1))


def verify_word_equation(text: str) -> str:
    # "A number is multiplied by 4 and then 7 is added. The result is 35. What is the number?"
    a, b, c = _nums(text)
    x = (c - b) / a
    assert abs(x - round(x)) < 1e-9
    return str(int(round(x)))


def verify_triangle_angle(text: str) -> str:
    a, b = _nums(text)
    return f"{int(180 - a - b)}°"


def verify_right_set(text: str) -> str:
    # key is the triple listed in params; verify by re-testing every option
    return "SEE_OPTIONS"


def verify_complement(text: str) -> str:
    (a,) = _nums(text)
    return f"{int(90 - a)}°"


def verify_supplement(text: str) -> str:
    (a,) = _nums(text)
    return f"{int(180 - a)}°"


def verify_exterior(text: str) -> str:
    a, b = _nums(text)
    return f"{int(a + b)}°"


def verify_quad_angle(text: str) -> str:
    a, b, c = _nums(text)
    return f"{int(360 - a - b - c)}°"


def verify_polygon_sum(text: str) -> str:
    # "sum of the interior angles of a hexagon (6 sides)"
    m = re.search(r"\((\d+) sides\)", text)
    n = int(m.group(1))
    return f"{(n - 2) * 180}°"


def verify_symmetry(text: str) -> str:
    shapes = {"square": 4, "rectangle": 2, "equilateral triangle": 3, "rhombus": 2, "isosceles triangle": 1}
    for name, count in shapes.items():
        if name in text:
            return str(count)
    raise AssertionError("no shape found")


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def _unique_distractors(key: str, candidates: list[str]) -> list[str]:
    out: list[str] = []
    for d in candidates:
        if d != key and d not in out:
            out.append(d)
        if len(out) == 3:
            return out
    raise ValueError(f"not enough distractors for {key}")


def gen_linear_eq(rng: random.Random, chapter: int) -> AuthoredItem:
    x = rng.randint(2, 12)
    a = rng.randint(2, 9)
    b = rng.randint(1, 20)
    c = a * x + b
    key = str(x)
    return AuthoredItem(
        family="linear_eq", chapter=chapter, topic_slug="simple-equations",
        question_text=f"Solve for x: {a}x + {b} = {c}.",
        key=key,
        distractors=_unique_distractors(key, [str(x + 1), str(x - 1), str(x * 2), str(x + 2), str((c - b + a) // a + 1), str(x - 2)]),
        explanation=f"Subtract {b} from both sides: {a}x = {c - b}. Divide by {a}: x = {x}.",
        difficulty="easy", skill_tags=["algebra", "linear-equation"],
        params={"a": a, "b": b, "c": c},
    )


def gen_substitute(rng: random.Random, chapter: int) -> AuthoredItem:
    k = rng.randint(2, 9)
    p = rng.randint(2, 9)
    q = rng.randint(1, 15)
    key = str(p * k + q)
    return AuthoredItem(
        family="substitute", chapter=chapter, topic_slug="substitution",
        question_text=f"If x = {k}, what is the value of {p}x + {q}?",
        key=key,
        distractors=_unique_distractors(key, [str(p * k + q - p), str(p * (k + 1) + q), str(p + k + q), str(p * k + q + 1), str(p * k - q if p * k > q else p * k + q + 2)]),
        explanation=f"Substitute x = {k}: {p} x {k} + {q} = {p * k} + {q} = {key}.",
        difficulty="easy", skill_tags=["algebra", "substitution"],
        params={"k": k, "p": p, "q": q},
    )


def gen_square_sum(rng: random.Random, chapter: int) -> AuthoredItem:
    a = rng.randint(3, 15)
    b = rng.randint(2, 12)
    key = str((a + b) ** 2)
    return AuthoredItem(
        family="square_sum", chapter=chapter, topic_slug="expressions",
        question_text=f"What is ({a} + {b})^2?",
        key=key,
        distractors=_unique_distractors(key, [str(a**2 + b**2), str((a - b) ** 2), str(2 * a * b), str(a**2 + 2 * a * b), str((a + b + 1) ** 2)]),
        explanation=f"({a} + {b})^2 = {a}^2 + 2 x {a} x {b} + {b}^2 = {a*a} + {2*a*b} + {b*b} = {key}.",
        difficulty="medium", skill_tags=["algebra", "expansion"],
        params={"a": a, "b": b},
    )


def gen_expand_product(rng: random.Random, chapter: int) -> AuthoredItem:
    a = rng.randint(1, 9)
    b = rng.randint(1, 9)
    while b == a:
        b = rng.randint(1, 9)
    key = f"x^2 + {a + b}x + {a * b}"
    return AuthoredItem(
        family="expand_product", chapter=chapter, topic_slug="expressions",
        question_text=f"Expand (x + {a})(x + {b}).",
        key=key,
        distractors=_unique_distractors(key, [f"x^2 + {a * b}x + {a + b}", f"x^2 + {a + b}x + {a * b + 1}", f"x^2 + {a + b + 1}x + {a * b}", f"x^2 + {abs(a - b)}x + {a * b}", f"x^2 + {a + b}x + {a * b - 1}"]),
        explanation=f"(x + {a})(x + {b}) = x^2 + ({a} + {b})x + {a} x {b} = x^2 + {a + b}x + {a * b}.",
        difficulty="medium", skill_tags=["algebra", "expansion"],
        params={"a": a, "b": b},
    )


def gen_like_terms(rng: random.Random, chapter: int) -> AuthoredItem:
    a = rng.randint(2, 9)
    b = rng.randint(2, 9)
    c = rng.randint(1, min(a + b - 1, 6))
    total = a + b - c
    key = f"{total}x"
    return AuthoredItem(
        family="like_terms", chapter=chapter, topic_slug="like-terms",
        question_text=f"Simplify: {a}x + {b}x - {c}x.",
        key=key,
        distractors=_unique_distractors(key, [f"{a + b + c}x", f"{total + 1}x", f"{abs(a - b) + c}x", f"{total - 1}x", f"{a + b}x"]),
        explanation=f"Combine like terms: ({a} + {b} - {c})x = {total}x.",
        difficulty="easy", skill_tags=["algebra", "like-terms"],
        params={"a": a, "b": b, "c": c},
    )


def gen_nth_term(rng: random.Random, chapter: int) -> AuthoredItem:
    a = rng.randint(2, 9)
    b = rng.randint(1, 10)
    k = rng.choice([5, 8, 10, 12, 20])
    key = str(a * k + b)
    return AuthoredItem(
        family="nth_term", chapter=chapter, topic_slug="patterns-generalisation",
        question_text=f"A pattern follows the rule: term = {a}n + {b}. What is the {k}th term?",
        key=key,
        distractors=_unique_distractors(key, [str(a * k + b - a), str(a * (k + 1) + b), str(a * k), str(a + b + k), str(a * k + b + 1)]),
        explanation=f"Substitute n = {k}: {a} x {k} + {b} = {a * k} + {b} = {key}.",
        difficulty="medium", skill_tags=["algebra", "patterns"],
        params={"a": a, "b": b, "k": k},
    )


def gen_consecutive(rng: random.Random, chapter: int) -> AuthoredItem:
    n = rng.randint(4, 30)
    s = 3 * (n + 1)
    key = str(n)
    return AuthoredItem(
        family="consecutive", chapter=chapter, topic_slug="forming-equations",
        question_text=f"The sum of three consecutive whole numbers is {s}. What is the smallest of the three numbers?",
        key=key,
        distractors=_unique_distractors(key, [str(n + 1), str(n + 2), str(n - 1), str(s // 3), str(n + 3)]),
        explanation=f"Let the numbers be n, n+1, n+2. Then 3n + 3 = {s}, so n = {s // 3 - 1}.",
        difficulty="medium", skill_tags=["algebra", "word-problem"],
        params={"s": s},
    )


def gen_word_equation(rng: random.Random, chapter: int) -> AuthoredItem:
    x = rng.randint(3, 15)
    a = rng.randint(2, 8)
    b = rng.randint(1, 15)
    c = a * x + b
    key = str(x)
    return AuthoredItem(
        family="word_equation", chapter=chapter, topic_slug="forming-equations",
        question_text=f"A number is multiplied by {a} and then {b} is added. The result is {c}. What is the number?",
        key=key,
        distractors=_unique_distractors(key, [str(x + 1), str(x - 1), str(x * 2), str(x + 2), str((c + b) // a if (c + b) % a == 0 else x + 3)]),
        explanation=f"Let the number be n. {a}n + {b} = {c}, so {a}n = {c - b} and n = {x}.",
        difficulty="medium", skill_tags=["algebra", "word-problem"],
        params={"a": a, "b": b, "c": c},
    )


def gen_triangle_angle(rng: random.Random, chapter: int) -> AuthoredItem:
    a = rng.randint(25, 110)
    b = rng.randint(20, min(110, 150 - a))
    c = 180 - a - b
    key = f"{c}°"
    return AuthoredItem(
        family="triangle_angle", chapter=chapter, topic_slug="spatial-reasoning",
        question_text=f"In triangle ABC, angle A = {a}° and angle B = {b}°. What is angle C?",
        key=key,
        distractors=_unique_distractors(key, [f"{c + 10}°", f"{c - 10}°", f"{180 - a}°", f"{a + b}°", f"{c + 5}°"]),
        explanation=f"Angles of a triangle add to 180°: C = 180 - {a} - {b} = {c}°.",
        difficulty="easy", skill_tags=["geometry", "angles"],
        params={"a": a, "b": b},
    )


_TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (6, 8, 10), (9, 12, 15), (12, 16, 20), (20, 21, 29)]


def gen_right_set(rng: random.Random, chapter: int) -> AuthoredItem:
    triple = rng.choice(_TRIPLES)
    key = f"{triple[0]} cm, {triple[1]} cm, {triple[2]} cm"
    wrong: list[str] = []
    while len(wrong) < 3:
        a, b, c = sorted(rng.randint(2, 26) for _ in range(3))
        if a * a + b * b != c * c and f"{a} cm, {b} cm, {c} cm" != key:
            cand = f"{a} cm, {b} cm, {c} cm"
            if cand not in wrong:
                wrong.append(cand)
    return AuthoredItem(
        family="right_set", chapter=chapter, topic_slug="congruence",
        question_text="Which of these sets of side lengths forms a right-angled triangle?",
        key=key, distractors=wrong,
        explanation=f"{triple[0]}^2 + {triple[1]}^2 = {triple[0]**2 + triple[1]**2} = {triple[2]}^2, so the triangle is right-angled (converse of the Baudhayana-Pythagoras theorem).",
        difficulty="medium", skill_tags=["geometry", "pythagoras"],
        params={"triple": triple},
    )


def gen_complement(rng: random.Random, chapter: int) -> AuthoredItem:
    a = rng.randint(15, 75)
    key = f"{90 - a}°"
    return AuthoredItem(
        family="complement", chapter=chapter, topic_slug="spatial-reasoning",
        question_text=f"Two angles are complementary. One of them is {a}°. What is the other angle?",
        key=key,
        distractors=_unique_distractors(key, [f"{180 - a}°", f"{a}°", f"{90 - a + 5}°", f"{90 - a - 5}°", f"{90 - a + 10}°"]),
        explanation=f"Complementary angles add to 90°: 90 - {a} = {90 - a}°.",
        difficulty="easy", skill_tags=["geometry", "angles"], params={"a": a},
    )


def gen_supplement(rng: random.Random, chapter: int) -> AuthoredItem:
    a = rng.randint(25, 155)
    key = f"{180 - a}°"
    return AuthoredItem(
        family="supplement", chapter=chapter, topic_slug="spatial-reasoning",
        question_text=f"Two angles are supplementary. One of them is {a}°. What is the other angle?",
        key=key,
        distractors=_unique_distractors(key, [f"{a // 2}°", f"{a}°", f"{180 - a + 10}°", f"{180 - a - 10}°", f"{360 - a}°", f"{(180 - a) // 2}°"]),
        explanation=f"Supplementary angles add to 180°: 180 - {a} = {180 - a}°.",
        difficulty="easy", skill_tags=["geometry", "angles"], params={"a": a},
    )


def gen_exterior(rng: random.Random, chapter: int) -> AuthoredItem:
    a = rng.randint(25, 80)
    b = rng.randint(25, 80)
    key = f"{a + b}°"
    return AuthoredItem(
        family="exterior", chapter=chapter, topic_slug="spatial-reasoning",
        question_text=f"In a triangle, the exterior angle at one vertex is formed. The two interior angles opposite to it are {a}° and {b}°. What is the exterior angle?",
        key=key,
        distractors=_unique_distractors(key, [f"{180 - a - b}°", f"{abs(a - b)}°", f"{a}°", f"{b}°", f"{a + b - 10}°"]),
        explanation=f"An exterior angle equals the sum of the two opposite interior angles: {a} + {b} = {a + b}°.",
        difficulty="medium", skill_tags=["geometry", "angles"], params={"a": a, "b": b},
    )


def gen_quad_angle(rng: random.Random, chapter: int) -> AuthoredItem:
    a = rng.randint(50, 130)
    b = rng.randint(50, 130)
    c = rng.randint(40, min(130, 300 - a - b))
    d = 360 - a - b - c
    key = f"{d}°"
    return AuthoredItem(
        family="quad_angle", chapter=chapter, topic_slug="spatial-reasoning",
        question_text=f"Three angles of a quadrilateral are {a}°, {b}°, and {c}°. What is the fourth angle?",
        key=key,
        distractors=_unique_distractors(key, [f"{d + 10}°", f"{d - 10}°", f"{180 - a}°", f"{a + b + c - 180}°", f"{d + 20}°"]),
        explanation=f"Angles of a quadrilateral add to 360°: 360 - {a} - {b} - {c} = {d}°.",
        difficulty="medium", skill_tags=["geometry", "angles"], params={"a": a, "b": b, "c": c},
    )


_POLY_NAMES = {5: "pentagon", 6: "hexagon", 7: "heptagon", 8: "octagon", 9: "nonagon", 10: "decagon"}


def gen_polygon_sum(rng: random.Random, chapter: int) -> AuthoredItem:
    n = rng.choice(list(_POLY_NAMES))
    total = (n - 2) * 180
    key = f"{total}°"
    return AuthoredItem(
        family="polygon_sum", chapter=chapter, topic_slug="spatial-reasoning",
        question_text=f"What is the sum of the interior angles of a {_POLY_NAMES[n]} ({n} sides)?",
        key=key,
        distractors=_unique_distractors(key, [f"{(n - 1) * 180}°", f"{n * 180}°", f"{(n - 2) * 90}°", f"{(n - 3) * 180}°", f"{total + 90}°"]),
        explanation=f"Sum of interior angles = (n - 2) x 180 = ({n} - 2) x 180 = {total}°.",
        difficulty="medium", skill_tags=["geometry", "polygons"], params={"n": n},
    )


_SHAPES = [("square", 4), ("rectangle", 2), ("equilateral triangle", 3), ("rhombus", 2), ("isosceles triangle", 1)]


def gen_symmetry(rng: random.Random, chapter: int) -> AuthoredItem:
    name, count = rng.choice(_SHAPES)
    key = str(count)
    pool = [str(i) for i in range(0, 7)]
    return AuthoredItem(
        family="symmetry", chapter=chapter, topic_slug="symmetry",
        question_text=f"How many lines of symmetry does a {name} have?",
        key=key,
        distractors=_unique_distractors(key, [p for p in pool if p != key]),
        explanation=f"A {name} has {count} line{'s' if count != 1 else ''} of symmetry.",
        difficulty="easy", skill_tags=["geometry", "symmetry"], params={"shape": name},
    )


GENERATORS = {
    13: [
        (gen_linear_eq, verify_linear_eq, 6),
        (gen_substitute, verify_substitute, 6),
        (gen_square_sum, verify_square_sum, 5),
        (gen_expand_product, verify_expand_product, 5),
        (gen_like_terms, verify_like_terms, 6),
        (gen_nth_term, verify_nth_term, 5),
        (gen_consecutive, verify_consecutive, 5),
        (gen_word_equation, verify_word_equation, 6),
    ],
    11: [
        (gen_triangle_angle, verify_triangle_angle, 6),
        (gen_right_set, None, 5),
        (gen_complement, verify_complement, 5),
        (gen_supplement, verify_supplement, 5),
        (gen_exterior, verify_exterior, 5),
        (gen_quad_angle, verify_quad_angle, 6),
        (gen_polygon_sum, verify_polygon_sum, 5),
        (gen_symmetry, verify_symmetry, 5),
    ],
}


def _verify_right_set(item: AuthoredItem) -> bool:
    # every option must be tested: exactly the key satisfies a^2+b^2=c^2
    for opt in [item.key] + item.distractors:
        a, b, c = sorted(_nums(opt))
        is_right = abs(a * a + b * b - c * c) < 1e-9
        if (opt == item.key) != is_right:
            return False
    return True


VERIFIERS = {
    "linear_eq": verify_linear_eq,
    "substitute": verify_substitute,
    "square_sum": verify_square_sum,
    "expand_product": verify_expand_product,
    "like_terms": verify_like_terms,
    "nth_term": verify_nth_term,
    "consecutive": verify_consecutive,
    "word_equation": verify_word_equation,
    "triangle_angle": verify_triangle_angle,
    "complement": verify_complement,
    "supplement": verify_supplement,
    "exterior": verify_exterior,
    "quad_angle": verify_quad_angle,
    "polygon_sum": verify_polygon_sum,
    "symmetry": verify_symmetry,
}


def build_candidate(item: AuthoredItem, spec, rng: random.Random) -> dict:
    concept_id = uuid5(spec.id, f"concept:{item.topic_slug}")
    if item.qtype == "true_false":
        options = [{"id": "A", "text": "True"}, {"id": "B", "text": "False"}]
        correct = "A" if item.key == "True" else "B"
        answer_spec = {
            "type": "true_false",
            "correct_option_id": correct,
            "source_answer": item.key,
            "verification": f"code_verified:{item.family}",
            "answer_provenance": "evater_original",
        }
        render_config: dict = {}
    elif item.qtype == "numerical":
        options = []
        correct = None
        value = float(item.key)
        expected_value: float | int = int(value) if value.is_integer() else value
        unit = item.params.get("unit", "")
        answer_spec = {
            "type": "numerical",
            "expected_value": expected_value,
            "tolerance": item.params.get("tolerance", 0),
            "unit": unit,
            "source_answer": item.key,
            "verification": f"code_verified:{item.family}",
            "answer_provenance": "evater_original",
        }
        render_config = {"unit": unit} if unit else {}
    else:
        order = [item.key] + item.distractors
        rng.shuffle(order)
        letters = ["A", "B", "C", "D"]
        options = [{"id": letters[i], "text": order[i]} for i in range(4)]
        correct = letters[order.index(item.key)]
        answer_spec = {
            "type": "mcq_single",
            "source_answer": item.key,
            "verification": f"code_verified:{item.family}",
            "answer_provenance": "evater_original",
        }
        render_config = {}
    return {
        "chapter_id": str(spec.id),
        "concept_id": str(concept_id),
        "question_type": item.qtype,
        "question_text": item.question_text,
        "options": options,
        "correct_option_id": correct,
        "explanation": item.explanation,
        "hint": None,
        "difficulty": item.difficulty,
        "cognitive_level": "apply",
        "skill_tags": item.skill_tags + ["evater_original"],
        "misconception_tags": [],
        "question_style": "direct",
        "estimated_time_seconds": 90,
        "marks": 1,
        "source_url": SOURCE_URL,
        "source_locator": f"Evater-authored generator family {item.family}",
        "source_question_id": f"evater:class8-maths:ch{item.chapter}:{item.family}:{abs(hash(item.question_text)) % 99999:05d}",
        "source_license": "Evater original content",
        "source_attribution": ATTRIBUTION,
        "source_tier": "manual",
        "rework_status": "not_required",
        "question_family_key": f"evater-{'mcq' if item.qtype == 'mcq_single' else item.qtype}:{spec.id}:{item.family}",
        "variant_key": "evater_authored",
        "media": [],
        "option_media": {},
        "render_config": render_config,
        "answer_spec": answer_spec,
        "generation_spec": {
            "ingestion_method": "evater_authored_generator",
            "generator_family": item.family,
            "parameters": item.params,
            "verified_by": "independent_text_reparse",
        },
        "license_status": "review_required",
        "review_status": "review",
    }



# Topic-slug remap: some generators are shared across chapters, but each chapter's
# concepts table only contains its own slugs. Map to a valid concept per chapter.
SLUG_REMAP = {
    (4, "spatial-reasoning"): "angle-sum",
    (4, "symmetry"): "quadrilateral-properties",
    (5, "power-patterns"): "digits-puzzles",
    (5, "exponents"): "number-patterns",
    (6, "expressions"): "expansion",
    (6, "like-terms"): "simplification",
    (9, "congruence"): "converse",
    (9, "exponents"): "pythagoras-statement",
    (12, "patterns-generalisation"): "data-stories",
    (13, "identities"): "expressions",
}


def author_chapter(chapter: int, seed: int) -> tuple[list[dict], list[str]]:
    spec = next(c for c in CHAPTERS if c.sequence_number == chapter)
    rng = random.Random(seed)
    report: list[str] = []
    candidates: list[dict] = []
    seen_texts: set[str] = set()
    for gen, _ver, count in GENERATORS[chapter]:
        produced = 0
        attempts = 0
        while produced < count and attempts < count * 10:
            attempts += 1
            try:
                item = gen(rng, chapter)
                item.topic_slug = SLUG_REMAP.get((chapter, item.topic_slug), item.topic_slug)
            except Exception as exc:
                report.append(f"GEN-ERROR: {exc}")
                continue
            dedupe_key = item.question_text + "|" + item.key
            if dedupe_key in seen_texts:
                continue
            # independent verification: re-parse the rendered text
            try:
                if item.family in SET_VERIFIERS:
                    assert SET_VERIFIERS[item.family](item)
                else:
                    rederived = VERIFIERS[item.family](item.question_text)
                    assert rederived == item.key, f"{rederived} != {item.key}"
            except Exception as exc:  # drop, never stage
                report.append(f"DROPPED {item.family}: {exc} :: {item.question_text[:60]}")
                continue
            seen_texts.add(dedupe_key)
            candidates.append(build_candidate(item, spec, rng))
            produced += 1
        report.append(f"{item.family}: {produced}/{count}")
    return candidates, report


def main() -> int:
    parser = argparse.ArgumentParser(description="Author Evater-original Class 8 Maths MCQs.")
    parser.add_argument("--chapters", default="11,13")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    chapters = [int(p) for p in args.chapters.split(",")]
    for chapter in chapters:
        spec = next(c for c in CHAPTERS if c.sequence_number == chapter)
        candidates, report = author_chapter(chapter, args.seed + chapter)
        pack = {
            "chapter": chapter,
            "title": spec.title,
            "source_url": SOURCE_URL,
            "questions": candidates,
            "items": [],
        }
        path = args.out_dir / f"class8_maths_scraped_chapter_{chapter:02d}.json"
        path.write_text(json.dumps(pack, indent=2, ensure_ascii=False))
        print(f"ch{chapter} {spec.title}: {len(candidates)} authored")
        for line in report:
            print("  ", line)
    return 0


# ---------------------------------------------------------------------------
# Chapter 2 (Power Play), 3 (A Story of Numbers), 7 (Proportional Reasoning-1)
# ---------------------------------------------------------------------------

def v_power(text: str) -> str:
    m = re.search(r"What is (\d+)\^(\d+)\?", text)
    return str(int(m.group(1)) ** int(m.group(2)))


def gen_power(rng, chapter):
    a = rng.choice([2, 3, 4, 5, 6, 7, 10])
    n = rng.randint(2, 8 if a <= 3 else 4)
    key = str(a ** n)
    return AuthoredItem(
        family="power", chapter=chapter, topic_slug="exponents",
        question_text=f"What is {a}^{n}?",
        key=key,
        distractors=_unique_distractors(key, [str(a * n), str(a ** (n + 1)), str(a ** n + a), str(a ** (n - 1)), str(n ** a if n ** a != a ** n else a ** n + 1), str(a ** n - a)]),
        explanation=f"{a}^{n} means {a} multiplied by itself {n} times = {key}.",
        difficulty="easy", skill_tags=["exponents"], params={"a": a, "n": n},
    )


def v_power_product(text: str) -> str:
    m = re.search(r"Simplify: (\d+)\^(\d+) x \1\^\??(\d+)", text.replace(f"^?", "^"))
    m = re.search(r"Simplify: (\d+)\^(\d+) x (\d+)\^(\d+)\.", text)
    a1, m1, a2, m2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    assert a1 == a2
    return f"{a1}^{m1 + m2}"


def gen_power_product(rng, chapter):
    a = rng.choice([2, 3, 5, 7])
    m1 = rng.randint(2, 6)
    m2 = rng.randint(2, 6)
    key = f"{a}^{m1 + m2}"
    return AuthoredItem(
        family="power_product", chapter=chapter, topic_slug="laws-of-exponents",
        question_text=f"Simplify: {a}^{m1} x {a}^{m2}. Write the answer as a single power.",
        key=key,
        distractors=_unique_distractors(key, [f"{a}^{m1 * m2}", f"{a}^{abs(m1 - m2)}", f"{a * 2}^{m1 + m2}", f"{a}^{m1 + m2 + 1}", f"{a}^{m1 + m2 - 1}"]),
        explanation=f"Product law: {a}^{m1} x {a}^{m2} = {a}^({m1}+{m2}) = {key}.",
        difficulty="medium", skill_tags=["exponents", "laws"], params={"a": a, "m": m1, "n": m2},
    )


def v_power_quotient(text: str) -> str:
    m = re.search(r"Simplify: (\d+)\^(\d+) / (\d+)\^(\d+)\.", text)
    a1, m1, a2, m2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    assert a1 == a2 and m1 > m2
    return f"{a1}^{m1 - m2}"


def gen_power_quotient(rng, chapter):
    a = rng.choice([2, 3, 5, 7])
    m2 = rng.randint(1, 4)
    m1 = m2 + rng.randint(1, 5)
    key = f"{a}^{m1 - m2}"
    return AuthoredItem(
        family="power_quotient", chapter=chapter, topic_slug="laws-of-exponents",
        question_text=f"Simplify: {a}^{m1} / {a}^{m2}. Write the answer as a single power.",
        key=key,
        distractors=_unique_distractors(key, [f"{a}^{m1 + m2}", f"{a}^{m1 * m2}", f"{a}^1", f"{a}^{m1 - m2 + 1}", f"1"]),
        explanation=f"Quotient law: {a}^{m1} / {a}^{m2} = {a}^({m1}-{m2}) = {key}.",
        difficulty="medium", skill_tags=["exponents", "laws"], params={"a": a, "m": m1, "n": m2},
    )


def v_power_of_power(text: str) -> str:
    m = re.search(r"Simplify: \((\d+)\^(\d+)\)\^(\d+)\.", text)
    a, m1, n = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return f"{a}^{m1 * n}"


def gen_power_of_power(rng, chapter):
    a = rng.choice([2, 3, 5, 7])
    m1 = rng.randint(2, 5)
    n = rng.randint(2, 4)
    key = f"{a}^{m1 * n}"
    return AuthoredItem(
        family="power_of_power", chapter=chapter, topic_slug="laws-of-exponents",
        question_text=f"Simplify: ({a}^{m1})^{n}. Write the answer as a single power.",
        key=key,
        distractors=_unique_distractors(key, [f"{a}^{m1 + n}", f"{a}^{m1 ** n}", f"{a}^{abs(m1 - n)}", f"{a}^{m1 * n + 1}", f"{a}^{n}"]),
        explanation=f"Power law: ({a}^{m1})^{n} = {a}^({m1} x {n}) = {key}.",
        difficulty="medium", skill_tags=["exponents", "laws"], params={"a": a, "m": m1, "n": n},
    )


def v_zero_exponent(text: str) -> str:
    m = re.search(r"What is (\d+)\^0\?", text)
    assert int(m.group(1)) > 0
    return "1"


def gen_zero_exponent(rng, chapter):
    a = rng.randint(2, 99)
    return AuthoredItem(
        family="zero_exponent", chapter=chapter, topic_slug="zero-negative-exponents",
        question_text=f"What is {a}^0?",
        key="1",
        distractors=["0", str(a), f"1/{a}", f"{a}^1"],
        explanation=f"Any non-zero number raised to the power 0 is 1, so {a}^0 = 1.",
        difficulty="easy", skill_tags=["exponents"], params={"a": a},
    )


def v_negative_exponent(text: str) -> str:
    m = re.search(r"What is (\d+)\^\(-(\d+)\)\?", text)
    a, n = int(m.group(1)), int(m.group(2))
    return f"1/{a ** n}"


def gen_negative_exponent(rng, chapter):
    a = rng.choice([2, 3, 4, 5, 10])
    n = rng.randint(1, 4 if a <= 4 else 3)
    key = f"1/{a ** n}"
    return AuthoredItem(
        family="negative_exponent", chapter=chapter, topic_slug="zero-negative-exponents",
        question_text=f"What is {a}^(-{n})? Write the answer as a fraction.",
        key=key,
        distractors=_unique_distractors(key, [str(a ** n), f"1/{a * n}", f"-1/{a ** n}", f"1/{a ** (n + 1)}", str(a ** n - 1)]),
        explanation=f"{a}^(-{n}) = 1/{a}^{n} = 1/{a ** n}.",
        difficulty="medium", skill_tags=["exponents", "negative"], params={"a": a, "n": n},
    )


def v_sci(text: str) -> str:
    m = re.search(r"Write (\d+) in scientific notation", text)
    N = int(m.group(1))
    k = len(str(N)) - 1
    c = N / (10 ** k)
    cstr = f"{c:.1f}".rstrip("0").rstrip(".") if "." in f"{c:.2f}" else str(int(c))
    c = round(c, 9)
    cstr = str(int(c)) if c == int(c) else str(c)
    return f"{cstr} x 10^{k}"


def gen_sci(rng, chapter):
    c = rng.choice([round(x * 0.1, 1) for x in range(11, 100)])
    k = rng.randint(3, 7)
    N = int(round(c * (10 ** k)))
    cstr = str(int(c)) if c == int(c) else str(c)
    key = f"{cstr} x 10^{k}"
    c_up = round(c + 0.1, 1)
    c_dn = round(c - 0.1, 1)
    return AuthoredItem(
        family="sci", chapter=chapter, topic_slug="scientific-notation",
        question_text=f"Write {N} in scientific notation (standard form).",
        key=key,
        distractors=_unique_distractors(key, [f"{cstr} x 10^{k + 1}", f"{cstr} x 10^{k - 1}", f"{c_up} x 10^{k}", f"{c_dn} x 10^{k}", f"{cstr} x 10^{k + 2}"]),
        explanation=f"{N} = {cstr} x 10^{k} (move the decimal point {k} places).",
        difficulty="medium", skill_tags=["exponents", "scientific-notation"], params={"N": N},
    )


def v_last_digit(text: str) -> str:
    m = re.search(r"units digit of (\d+)\^(\d+)", text)
    a, n = int(m.group(1)), int(m.group(2))
    return str((a ** n) % 10)


def gen_last_digit(rng, chapter):
    a = rng.choice([2, 3, 7, 8, 9, 13, 17])
    n = rng.randint(11, 60)
    key = str((a ** n) % 10)
    pool = [str(d) for d in range(10)]
    rng.shuffle(pool)
    return AuthoredItem(
        family="last_digit", chapter=chapter, topic_slug="power-patterns",
        question_text=f"What is the units digit of {a}^{n}?",
        key=key,
        distractors=_unique_distractors(key, [d for d in pool if d != key]),
        explanation=f"The units digits of powers of {a} repeat in a cycle; {a}^{n} falls on the digit {key}.",
        difficulty="hard", skill_tags=["exponents", "patterns"], params={"a": a, "n": n},
    )


def gen_greatest_power(rng, chapter):
    # options verified as a set: exactly one maximum
    seen: set[int] = set()
    opts: list[str] = []
    while len(opts) < 4:
        a = rng.choice([2, 3, 4, 5, 6, 7, 9, 10])
        n = rng.randint(2, 7)
        v = a ** n
        if v not in seen and v < 100000:
            seen.add(v)
            opts.append(f"{a}^{n}")
    vals = {o: int(o.split("^")[0]) ** int(o.split("^")[1]) for o in opts}
    key = max(vals, key=vals.get)
    return AuthoredItem(
        family="greatest_power", chapter=chapter, topic_slug="comparing-powers",
        question_text="Which of these numbers is the greatest?",
        key=key, distractors=[o for o in opts if o != key],
        explanation=f"Evaluating each: " + ", ".join(f"{o} = {vals[o]}" for o in opts) + f". The greatest is {key}.",
        difficulty="medium", skill_tags=["exponents", "compare"], params={"options": opts},
    )


def _verify_greatest_power(item: AuthoredItem) -> bool:
    vals = {o: int(o.split("^")[0]) ** int(o.split("^")[1]) for o in [item.key] + item.distractors}
    if len(set(vals.values())) != 4:
        return False
    return max(vals, key=vals.get) == item.key


def v_expanded(text: str) -> str:
    m = re.search(r"expanded form of (\d+) using powers of 10", text)
    N = m.group(1)
    parts = []
    for i, d in enumerate(N):
        p = len(N) - 1 - i
        parts.append(f"{d} x 10^{p}")
    return " + ".join(parts)


def gen_expanded(rng, chapter):
    digits = rng.sample(range(1, 10), 4)
    N = int("".join(map(str, digits)))
    parts = [f"{d} x 10^{3 - i}" for i, d in enumerate(digits)]
    key = " + ".join(parts)
    wrongs = []
    for _ in range(6):
        perm = digits[:]
        rng.shuffle(perm)
        w = " + ".join(f"{d} x 10^{3 - i}" for i, d in enumerate(perm))
        if w != key and w not in wrongs:
            wrongs.append(w)
    return AuthoredItem(
        family="expanded", chapter=chapter, topic_slug="expanded-form",
        question_text=f"What is the expanded form of {N} using powers of 10?",
        key=key, distractors=wrongs[:3],
        explanation=f"{N} = " + key + ".",
        difficulty="medium", skill_tags=["place-value", "exponents"], params={"N": N},
    )


def v_place_value(text: str) -> str:
    m = re.search(r"In the number (\d+), what is the place value of the digit (\d+)\?", text)
    N, d = m.group(1), m.group(2)
    assert N.count(d) == 1
    pos = len(N) - 1 - N.index(d)
    return str(int(d) * (10 ** pos))


def gen_place_value(rng, chapter):
    digits = rng.sample(range(1, 10), rng.choice([4, 5]))
    N = "".join(map(str, digits))
    d = rng.choice(digits)
    pos = len(N) - 1 - N.index(str(d))
    key = str(d * (10 ** pos))
    return AuthoredItem(
        family="place_value", chapter=chapter, topic_slug="place-value",
        question_text=f"In the number {N}, what is the place value of the digit {d}?",
        key=key,
        distractors=_unique_distractors(key, [str(d), str(d * (10 ** (pos + 1))), str(d * (10 ** max(pos - 1, 0))), str(d * 10), str(int(N) - d * (10 ** pos)), str(d * (10 ** pos) + d), str(d * (10 ** pos) - d if d * (10 ** pos) > d else d * (10 ** (pos + 2)))]),
        explanation=f"The digit {d} is in the 10^{pos} place, so its place value is {d} x 10^{pos} = {key}.",
        difficulty="easy", skill_tags=["place-value"], params={"N": N, "d": d},
    )


def v_base_to_10(text: str) -> str:
    m = re.search(r"number (\d+) \(base (\d+)\) is equal to which base-10 number", text)
    digits, b = m.group(1), int(m.group(2))
    return str(int(digits, b))


def gen_base_to_10(rng, chapter):
    b = rng.choice([2, 3, 4, 5, 8])
    length = rng.randint(3, 4)
    digits = "".join(rng.choice("01234"[:b] if b < 5 else "01234567"[:b]) for _ in range(length)).lstrip("0")
    if len(digits) < 2:
        digits = "1" + digits
    key = str(int(digits, b))
    return AuthoredItem(
        family="base_to_10", chapter=chapter, topic_slug="base-systems",
        question_text=f"The number {digits} (base {b}) is equal to which base-10 number?",
        key=key,
        distractors=_unique_distractors(key, [str(int(digits)), str(int(key) + b), str(int(key) - b if int(key) > b else int(key) + 1), str(int(key) + 1), str(int(digits, b + 1) if max(int(x) for x in digits) < b + 1 else int(key) + 2)]),
        explanation=f"Multiply each digit by its place power of {b} and add: {digits} (base {b}) = {key}.",
        difficulty="hard", skill_tags=["base-systems"], params={"digits": digits, "b": b},
    )


_ROMAN_TABLE = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def _to_roman(n: int) -> str:
    out = ""
    for v, s in _ROMAN_TABLE:
        while n >= v:
            out += s
            n -= v
    return out


def _from_roman(s: str) -> int:
    total = 0
    i = 0
    vals = {"M": 1000, "D": 500, "C": 100, "L": 50, "X": 10, "V": 5, "I": 1}
    while i < len(s):
        if i + 1 < len(s) and vals[s[i]] < vals[s[i + 1]]:
            total -= vals[s[i]]
        else:
            total += vals[s[i]]
        i += 1
    return total


def v_roman(text: str) -> str:
    m = re.search(r"What is ([IVXLCDM]+) in Hindu-Arabic numerals\?", text)
    return str(_from_roman(m.group(1)))


def gen_roman(rng, chapter):
    n = rng.randint(4, 89)
    roman = _to_roman(n)
    key = str(n)
    return AuthoredItem(
        family="roman", chapter=chapter, topic_slug="numeral-conversion",
        question_text=f"What is {roman} in Hindu-Arabic numerals?",
        key=key,
        distractors=_unique_distractors(key, [str(n + 1), str(n - 1), str(n + 10), str(n - 10 if n > 10 else n + 5), str(_from_roman(roman.replace("I", "V", 1)) if "I" in roman else n + 2)]),
        explanation=f"{roman} = {key}.",
        difficulty="medium", skill_tags=["roman", "numerals"], params={"n": n},
    )


def v_count_range(text: str) -> str:
    m = re.search(r"from (\d+) to (\d+) \(both included\)", text)
    a, b = int(m.group(1)), int(m.group(2))
    return str(b - a + 1)


def gen_count_range(rng, chapter):
    a = rng.randint(11, 99)
    b = a + rng.randint(9, 99)
    key = str(b - a + 1)
    return AuthoredItem(
        family="count_range", chapter=chapter, topic_slug="counting",
        question_text=f"How many whole numbers are there from {a} to {b} (both included)?",
        key=key,
        distractors=_unique_distractors(key, [str(b - a), str(b - a + 2), str(b - a - 1), str(b - a + 10), str(b)]),
        explanation=f"Count = last - first + 1 = {b} - {a} + 1 = {key}.",
        difficulty="easy", skill_tags=["counting"], params={"a": a, "b": b},
    )


def v_dec_to_binary(text: str) -> str:
    m = re.search(r"What is (\d+) written in base 2\?", text)
    return bin(int(m.group(1)))[2:]


def gen_dec_to_binary(rng, chapter):
    n = rng.randint(5, 40)
    key = bin(n)[2:]
    return AuthoredItem(
        family="dec_to_binary", chapter=chapter, topic_slug="base-systems",
        question_text=f"What is {n} written in base 2?",
        key=key,
        distractors=_unique_distractors(key, [bin(n + 1)[2:], bin(n - 1)[2:], key[::-1], bin(n + 2)[2:], key + "0"]),
        explanation=f"Divide repeatedly by 2 and read the remainders upward: {n} = {key} (base 2).",
        difficulty="hard", skill_tags=["base-systems", "binary"], params={"n": n},
    )


_HISTORY_FACTS = {
    "Which ancient civilization used a base-60 (sexagesimal) number system?": "Babylonians",
    "How many distinct symbols (digits) does the Hindu-Arabic numeral system use?": "10",
    "Which digit's invention in India was crucial to the place-value system?": "0 (zero)",
    "In the Roman numeral system, which symbol represents 100?": "C",
    "In the Roman numeral system, what value does the symbol L represent?": "50",
    "Which ancient civilization used a base-20 (vigesimal) number system?": "Mayans",
}

_FACT_DISTRACTORS = {
    "Babylonians": ["Egyptians", "Romans", "Greeks"],
    "10": ["8", "12", "16"],
    "0 (zero)": ["1 (one)", "9 (nine)", "10 (ten)"],
    "C": ["L", "D", "M"],
    "50": ["500", "5", "100"],
    "Mayans": ["Babylonians", "Egyptians", "Chinese"],
}


def gen_history(rng, chapter):
    q = rng.choice(list(_HISTORY_FACTS))
    key = _HISTORY_FACTS[q]
    return AuthoredItem(
        family="history", chapter=chapter, topic_slug="number-systems-history",
        question_text=q, key=key, distractors=_FACT_DISTRACTORS[key][:3],
        explanation=f"Fact: {q[:-1]} - {key}.",
        difficulty="easy", skill_tags=["history", "number-systems"], params={"fact": q},
    )


def _verify_history(item: AuthoredItem) -> bool:
    return _HISTORY_FACTS.get(item.question_text) == item.key and item.key not in item.distractors


def v_simplify_ratio(text: str) -> str:
    m = re.search(r"ratio (\d+):(\d+) in its simplest form", text)
    a, b = int(m.group(1)), int(m.group(2))
    import math
    g = math.gcd(a, b)
    assert g > 1
    return f"{a // g}:{b // g}"


def gen_simplify_ratio(rng, chapter):
    import math
    p = rng.randint(1, 9)
    q = rng.randint(1, 9)
    while math.gcd(p, q) != 1 or p == q:
        p, q = rng.randint(1, 9), rng.randint(1, 9)
    k = rng.randint(2, 12)
    a, b = p * k, q * k
    key = f"{p}:{q}"
    return AuthoredItem(
        family="simplify_ratio", chapter=chapter, topic_slug="ratio",
        question_text=f"Express the ratio {a}:{b} in its simplest form.",
        key=key,
        distractors=_unique_distractors(key, [f"{a}:{b}", f"{q}:{p}", f"{p + 1}:{q + 1}", f"{p}:{q + 1}", f"{p * 2}:{q * 3}"]),
        explanation=f"Divide both terms by their HCF {k}: {a}:{b} = {key}.",
        difficulty="easy", skill_tags=["ratio"], params={"a": a, "b": b},
    )


def v_unitary(text: str) -> str:
    m = re.search(r"If (\d+) notebooks cost Rs(\d+), what is the cost of one notebook\?", text)
    n, c = int(m.group(1)), int(m.group(2))
    assert c % n == 0
    return f"Rs{c // n}"


def gen_unitary(rng, chapter):
    n = rng.randint(2, 12)
    price = rng.randint(3, 45)
    c = n * price
    key = f"Rs{price}"
    return AuthoredItem(
        family="unitary", chapter=chapter, topic_slug="unitary-method",
        question_text=f"If {n} notebooks cost Rs{c}, what is the cost of one notebook?",
        key=key,
        distractors=_unique_distractors(key, [f"Rs{price + 1}", f"Rs{price - 1}", f"Rs{c - n}", f"Rs{price * 2}", f"Rs{price + 2}"]),
        explanation=f"Cost of one = {c} / {n} = Rs{price}.",
        difficulty="easy", skill_tags=["unitary"], params={"n": n, "c": c},
    )


def v_unitary_extended(text: str) -> str:
    m = re.search(r"If (\d+) pencils cost Rs(\d+), how much will (\d+) pencils cost\?", text)
    n, c, k = int(m.group(1)), int(m.group(2)), int(m.group(3))
    assert c % n == 0 and (c // n) * k == c * k // n
    return f"Rs{(c // n) * k}"


def gen_unitary_extended(rng, chapter):
    n = rng.randint(2, 10)
    price = rng.randint(2, 25)
    c = n * price
    k = rng.randint(2, 15)
    key = f"Rs{price * k}"
    return AuthoredItem(
        family="unitary_extended", chapter=chapter, topic_slug="unitary-method",
        question_text=f"If {n} pencils cost Rs{c}, how much will {k} pencils cost?",
        key=key,
        distractors=_unique_distractors(key, [f"Rs{price * k + price}", f"Rs{price * (k + 1)}", f"Rs{c + k}", f"Rs{price * k - price}", f"Rs{c * k}"]),
        explanation=f"One pencil costs {c}/{n} = Rs{price}, so {k} pencils cost {price} x {k} = Rs{price * k}.",
        difficulty="easy", skill_tags=["unitary"], params={"n": n, "c": c, "k": k},
    )


def v_direct_distance(text: str) -> str:
    m = re.search(r"covers (\d+) km in (\d+) hours.*in (\d+) hours\?", text)
    d, h, k = int(m.group(1)), int(m.group(2)), int(m.group(3))
    assert d % h == 0
    return f"{(d // h) * k} km"


def gen_direct_distance(rng, chapter):
    h = rng.randint(2, 6)
    v = rng.randint(15, 70)
    d = h * v
    k = rng.randint(2, 9)
    key = f"{v * k} km"
    return AuthoredItem(
        family="direct_distance", chapter=chapter, topic_slug="direct-proportion",
        question_text=f"A bus covers {d} km in {h} hours at a constant speed. How far will it go in {k} hours?",
        key=key,
        distractors=_unique_distractors(key, [f"{v * (k + 1)} km", f"{d + k} km", f"{v * k - v} km", f"{v * h * k} km", f"{v * k + 5} km"]),
        explanation=f"Speed = {d}/{h} = {v} km per hour. In {k} hours: {v} x {k} = {v * k} km.",
        difficulty="medium", skill_tags=["proportion"], params={"d": d, "h": h, "k": k},
    )


def v_ratio_division(text: str) -> str:
    m = re.search(r"Rs(\d+) is shared between Asha and Bala in the ratio (\d+):(\d+)\. How much does Asha get\?", text)
    t, p, q = int(m.group(1)), int(m.group(2)), int(m.group(3))
    assert t % (p + q) == 0
    return f"Rs{t * p // (p + q)}"


def gen_ratio_division(rng, chapter):
    p = rng.randint(1, 9)
    q = rng.randint(1, 9)
    while p == q:
        q = rng.randint(1, 9)
    s = rng.randint(10, 60)
    t = (p + q) * s
    key = f"Rs{p * s}"
    return AuthoredItem(
        family="ratio_division", chapter=chapter, topic_slug="ratio-division",
        question_text=f"Rs{t} is shared between Asha and Bala in the ratio {p}:{q}. How much does Asha get?",
        key=key,
        distractors=_unique_distractors(key, [f"Rs{q * s}", f"Rs{t // 2}", f"Rs{p * s + s}", f"Rs{p * s - s if p > 1 else p * s + 2 * s}", f"Rs{t - p * s + s}"]),
        explanation=f"Total parts = {p}+{q} = {p + q}. Each part = {t}/{p + q} = Rs{s}. Asha gets {p} parts = Rs{p * s}.",
        difficulty="medium", skill_tags=["ratio"], params={"t": t, "p": p, "q": q},
    )


def v_scale_map(text: str) -> str:
    m = re.search(r"1 cm represents (\d+) km\. Two cities are (\d+) cm apart", text)
    s, d = int(m.group(1)), int(m.group(2))
    return f"{s * d} km"


def gen_scale_map(rng, chapter):
    s = rng.choice([2, 5, 10, 15, 20, 25, 50, 100])
    d = rng.randint(2, 15)
    key = f"{s * d} km"
    return AuthoredItem(
        family="scale_map", chapter=chapter, topic_slug="scale-maps",
        question_text=f"On a map, 1 cm represents {s} km. Two cities are {d} cm apart on the map. What is the actual distance between them?",
        key=key,
        distractors=_unique_distractors(key, [f"{s * d + s} km", f"{s * d - s} km", f"{s + d} km", f"{s * d * 10} km", f"{s * (d + 1)} km"]),
        explanation=f"Actual distance = {d} x {s} = {s * d} km.",
        difficulty="easy", skill_tags=["scale"], params={"s": s, "d": d},
    )


def v_recipe(text: str) -> str:
    m = re.search(r"recipe for (\d+) people uses (\d+) g of rice\. How much rice is needed for (\d+) people\?", text)
    n, g, m2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
    assert g % n == 0
    return f"{(g // n) * m2} g"


def gen_recipe(rng, chapter):
    n = rng.randint(2, 6)
    u = rng.choice([25, 50, 75, 100, 125, 150])
    g = n * u
    m2 = rng.randint(2, 12)
    while m2 == n:
        m2 = rng.randint(2, 12)
    key = f"{u * m2} g"
    return AuthoredItem(
        family="recipe", chapter=chapter, topic_slug="recipes-mixtures",
        question_text=f"A recipe for {n} people uses {g} g of rice. How much rice is needed for {m2} people?",
        key=key,
        distractors=_unique_distractors(key, [f"{u * m2 + u} g", f"{u * (m2 - 1)} g", f"{g + m2} g", f"{g * m2} g", f"{u * m2 - u} g"]),
        explanation=f"Per person = {g}/{n} = {u} g. For {m2} people: {u} x {m2} = {u * m2} g.",
        difficulty="easy", skill_tags=["proportion", "recipe"], params={"n": n, "g": g, "m": m2},
    )


def gen_equivalent_ratio(rng, chapter):
    import math
    p = rng.randint(1, 8)
    q = rng.randint(1, 8)
    while math.gcd(p, q) != 1 or p == q:
        p, q = rng.randint(1, 8), rng.randint(1, 8)
    k = rng.randint(2, 6)
    key = f"{p * k}:{q * k}"
    wrongs = []
    for kp, kq in [(p + 1, q + 1), (p * k + 1, q * k), (p * k, q * k + 1), (p + k, q + k), (p * 2, q * 3), (p * k, q * (k + 1))]:
        w = f"{kp}:{kq}"
        if kp * q != kq * p and w != key and w not in wrongs:
            wrongs.append(w)
        if len(wrongs) == 3:
            break
    return AuthoredItem(
        family="equivalent_ratio", chapter=chapter, topic_slug="ratio",
        question_text=f"Which ratio is equivalent to {p}:{q}?",
        key=key, distractors=wrongs,
        explanation=f"Multiply both terms by {k}: {p}:{q} = {key}.",
        difficulty="medium", skill_tags=["ratio"], params={"p": p, "q": q, "k": k},
    )


def _verify_equivalent_ratio(item: AuthoredItem) -> bool:
    m = re.search(r"equivalent to (\d+):(\d+)\?", item.question_text)
    p, q = int(m.group(1)), int(m.group(2))
    def parses(r):
        a, b = r.split(":")
        return int(a), int(b)
    kp, kq = parses(item.key)
    if kp * q != kq * p:
        return False
    return all((lambda ab: ab[0] * q != ab[1] * p)(parses(w)) for w in item.distractors)


def v_better_deal(text: str) -> str:
    m = re.search(r"Shop A sells (\d+) oranges for Rs(\d+) and Shop B sells (\d+) oranges for Rs(\d+)", text)
    n1, c1, n2, c2 = map(int, m.groups())
    if c1 * n2 == c2 * n1:
        return "Both cost the same per orange"
    return "Shop A" if c1 * n2 < c2 * n1 else "Shop B"


def gen_better_deal(rng, chapter):
    while True:
        n1 = rng.randint(2, 12)
        n2 = rng.randint(2, 12)
        c1 = rng.randint(10, 120)
        c2 = rng.randint(10, 120)
        if c1 * n2 != c2 * n1 and n1 != n2:
            break
    key = "Shop A" if c1 * n2 < c2 * n1 else "Shop B"
    return AuthoredItem(
        family="better_deal", chapter=chapter, topic_slug="comparing-ratios",
        question_text=f"Shop A sells {n1} oranges for Rs{c1} and Shop B sells {n2} oranges for Rs{c2}. Which shop offers the lower price per orange?",
        key=key,
        distractors=["Shop B" if key == "Shop A" else "Shop A", "Both cost the same per orange", "Cannot be decided"],
        explanation=f"Per orange: A = {c1}/{n1} = Rs{c1 / n1:.2f}, B = {c2}/{n2} = Rs{c2 / n2:.2f}. Lower is {key}.",
        difficulty="medium", skill_tags=["ratio", "compare"], params={"n1": n1, "c1": c1, "n2": n2, "c2": c2},
    )


VERIFIERS.update({
    "power": v_power,
    "power_product": v_power_product,
    "power_quotient": v_power_quotient,
    "power_of_power": v_power_of_power,
    "zero_exponent": v_zero_exponent,
    "negative_exponent": v_negative_exponent,
    "sci": v_sci,
    "last_digit": v_last_digit,
    "expanded": v_expanded,
    "place_value": v_place_value,
    "base_to_10": v_base_to_10,
    "roman": v_roman,
    "count_range": v_count_range,
    "dec_to_binary": v_dec_to_binary,
    "simplify_ratio": v_simplify_ratio,
    "unitary": v_unitary,
    "unitary_extended": v_unitary_extended,
    "direct_distance": v_direct_distance,
    "ratio_division": v_ratio_division,
    "scale_map": v_scale_map,
    "recipe": v_recipe,
    "better_deal": v_better_deal,
})

SET_VERIFIERS = {
    "right_set": _verify_right_set,
    "greatest_power": _verify_greatest_power,
    "equivalent_ratio": _verify_equivalent_ratio,
    "history": _verify_history,
}

GENERATORS.update({
    2: [
        (gen_power, v_power, 6),
        (gen_power_product, v_power_product, 6),
        (gen_power_quotient, v_power_quotient, 6),
        (gen_power_of_power, v_power_of_power, 6),
        (gen_zero_exponent, v_zero_exponent, 4),
        (gen_negative_exponent, v_negative_exponent, 6),
        (gen_sci, v_sci, 6),
        (gen_last_digit, v_last_digit, 5),
        (gen_greatest_power, None, 5),
    ],
    3: [
        (gen_expanded, v_expanded, 6),
        (gen_place_value, v_place_value, 6),
        (gen_base_to_10, v_base_to_10, 6),
        (gen_roman, v_roman, 6),
        (gen_count_range, v_count_range, 5),
        (gen_dec_to_binary, v_dec_to_binary, 6),
        (gen_history, None, 6),
    ],
    7: [
        (gen_simplify_ratio, v_simplify_ratio, 6),
        (gen_unitary, v_unitary, 6),
        (gen_unitary_extended, v_unitary_extended, 6),
        (gen_direct_distance, v_direct_distance, 6),
        (gen_ratio_division, v_ratio_division, 6),
        (gen_scale_map, v_scale_map, 6),
        (gen_recipe, v_recipe, 5),
        (gen_equivalent_ratio, None, 5),
        (gen_better_deal, v_better_deal, 6),
    ],
})


# ---------------------------------------------------------------------------
# Chapter 1 (Squares/Cubes), 4 (Quadrilaterals), 5 (Number Play) families
# ---------------------------------------------------------------------------

import math as _math


def v_square_eval(text):
    m = re.search(r"What is (\d+)\^2\?", text)
    return str(int(m.group(1)) ** 2)


def gen_square_eval(rng, chapter):
    n = rng.randint(11, 35)
    key = str(n * n)
    return AuthoredItem(
        family="square_eval", chapter=chapter, topic_slug="perfect-squares",
        question_text=f"What is {n}^2?",
        key=key,
        distractors=_unique_distractors(key, [str(n * 2), str((n + 1) ** 2), str((n - 1) ** 2), str(n * n + n), str(n * n - n), str(n * 10)]),
        explanation=f"{n}^2 = {n} x {n} = {key}.",
        difficulty="easy", skill_tags=["squares"], params={"n": n},
    )


def v_sqrt_perfect(text):
    m = re.search(r"square root of (\d+)\?", text)
    N = int(m.group(1))
    r = _math.isqrt(N)
    assert r * r == N
    return str(r)


def gen_sqrt_perfect(rng, chapter):
    n = rng.randint(11, 40)
    key = str(n)
    return AuthoredItem(
        family="sqrt_perfect", chapter=chapter, topic_slug="square-roots",
        question_text=f"What is the square root of {n * n}?",
        key=key,
        distractors=_unique_distractors(key, [str(n + 1), str(n - 1), str(n * 2), str(n + 2), str(n * n // 2)]),
        explanation=f"{n} x {n} = {n * n}, so the square root is {n}.",
        difficulty="easy", skill_tags=["square-roots"], params={"n": n},
    )


def v_cube_eval(text):
    m = re.search(r"What is (\d+)\^3\?", text)
    return str(int(m.group(1)) ** 3)


def gen_cube_eval(rng, chapter):
    n = rng.randint(3, 12)
    key = str(n ** 3)
    return AuthoredItem(
        family="cube_eval", chapter=chapter, topic_slug="perfect-cubes",
        question_text=f"What is {n}^3?",
        key=key,
        distractors=_unique_distractors(key, [str(n * 3), str(n ** 2), str((n + 1) ** 3), str(n ** 3 + n), str(n ** 3 - n)]),
        explanation=f"{n}^3 = {n} x {n} x {n} = {key}.",
        difficulty="easy", skill_tags=["cubes"], params={"n": n},
    )


def v_cuberoot_perfect(text):
    m = re.search(r"cube root of (\d+)\?", text)
    N = int(m.group(1))
    r = round(N ** (1 / 3))
    assert r ** 3 == N
    return str(r)


def gen_cuberoot_perfect(rng, chapter):
    n = rng.randint(3, 15)
    key = str(n)
    return AuthoredItem(
        family="cuberoot_perfect", chapter=chapter, topic_slug="cube-roots",
        question_text=f"What is the cube root of {n ** 3}?",
        key=key,
        distractors=_unique_distractors(key, [str(n + 1), str(n - 1), str(n * 3), str(n + 2), str(n ** 2)]),
        explanation=f"{n} x {n} x {n} = {n ** 3}, so the cube root is {n}.",
        difficulty="easy", skill_tags=["cube-roots"], params={"n": n},
    )


def v_odd_sum(text):
    m = re.search(r"sum of the first (\d+) odd numbers", text)
    k = int(m.group(1))
    return str(k * k)


def gen_odd_sum(rng, chapter):
    k = rng.randint(4, 20)
    key = str(k * k)
    return AuthoredItem(
        family="odd_sum", chapter=chapter, topic_slug="square-patterns",
        question_text=f"What is the sum of the first {k} odd numbers (1 + 3 + 5 + ...)?",
        key=key,
        distractors=_unique_distractors(key, [str(k * k + k), str(k * k - k), str((k + 1) ** 2), str(2 * k), str(k * (k + 1))]),
        explanation=f"The first {k} odd numbers sum to {k}^2 = {key}.",
        difficulty="medium", skill_tags=["squares", "patterns"], params={"k": k},
    )


def v_square_area(text):
    m = re.search(r"square plot is (\d+) square metres", text)
    N = int(m.group(1))
    r = _math.isqrt(N)
    assert r * r == N
    return f"{r} m"


def gen_square_area(rng, chapter):
    n = rng.randint(5, 30)
    key = f"{n} m"
    return AuthoredItem(
        family="square_area", chapter=chapter, topic_slug="area-side",
        question_text=f"The area of a square plot is {n * n} square metres. What is the length of one side?",
        key=key,
        distractors=_unique_distractors(key, [f"{n + 1} m", f"{n - 1} m", f"{n * 4} m", f"{n * n // 4} m", f"{n + 2} m"]),
        explanation=f"Side = square root of area = square root of {n * n} = {n} m.",
        difficulty="easy", skill_tags=["squares", "area"], params={"n": n},
    )


def v_consec_squares(text):
    m = re.search(r"difference between (\d+)\^2 and (\d+)\^2", text)
    a, b = int(m.group(1)), int(m.group(2))
    return str(a * a - b * b)


def gen_consec_squares(rng, chapter):
    b = rng.randint(10, 30)
    a = b + 1
    key = str(a * a - b * b)
    return AuthoredItem(
        family="consec_squares", chapter=chapter, topic_slug="square-patterns",
        question_text=f"What is the difference between {a}^2 and {b}^2?",
        key=key,
        distractors=_unique_distractors(key, ["1", str(a + b + 2), str(a * b), str(2 * a), str(a + b - 2)]),
        explanation=f"{a}^2 - {b}^2 = ({a}+{b})({a}-{b}) = {a + b} x 1 = {key}.",
        difficulty="medium", skill_tags=["squares", "patterns"], params={"a": a, "b": b},
    )


def v_pgram_opp(text):
    m = re.search(r"parallelogram, one angle is (\d+)", text)
    a = int(m.group(1))
    return f"{180 - a}°"


def gen_pgram_opp(rng, chapter):
    a = rng.randint(30, 150)
    key = f"{180 - a}°"
    return AuthoredItem(
        family="pgram_opp", chapter=chapter, topic_slug="parallelogram",
        question_text=f"In a parallelogram, one angle is {a}°. What is the angle adjacent to it (on the same side)?",
        key=key,
        distractors=_unique_distractors(key, [f"{a}°", f"{90 - a // 2}°", f"{360 - a}°", f"{180 - a + 10}°", f"{180 - a - 10}°"]),
        explanation=f"Adjacent angles of a parallelogram add to 180°: 180 - {a} = {180 - a}°.",
        difficulty="medium", skill_tags=["quadrilaterals"], params={"a": a},
    )


def v_rect_diag(text):
    m = re.search(r"rectangle has length (\d+) cm and width (\d+) cm.*diagonal", text)
    l, w = int(m.group(1)), int(m.group(2))
    d2 = l * l + w * w
    d = _math.isqrt(d2)
    assert d * d == d2
    return f"{d} cm"


def gen_rect_diag(rng, chapter):
    l, w, d = rng.choice([(6, 8, 10), (9, 12, 15), (5, 12, 13), (8, 15, 17), (12, 16, 20), (7, 24, 25), (3, 4, 5), (10, 24, 26)])
    key = f"{d} cm"
    return AuthoredItem(
        family="rect_diag", chapter=chapter, topic_slug="rectangle-square",
        question_text=f"A rectangle has length {l} cm and width {w} cm. What is the length of its diagonal?",
        key=key,
        distractors=_unique_distractors(key, [f"{d + 1} cm", f"{d - 1} cm", f"{l + w} cm", f"{abs(l - w)} cm", f"{d + 2} cm"]),
        explanation=f"Diagonal = square root of ({l}^2 + {w}^2) = square root of {l * l + w * w} = {d} cm.",
        difficulty="medium", skill_tags=["rectangles", "pythagoras"], params={"l": l, "w": w},
    )


def v_rhombus_perimeter(text):
    m = re.search(r"side (\d+) cm.*perimeter", text)
    s = int(m.group(1))
    return f"{4 * s} cm"


def gen_rhombus_perimeter(rng, chapter):
    s = rng.randint(3, 25)
    key = f"{4 * s} cm"
    return AuthoredItem(
        family="rhombus_perimeter", chapter=chapter, topic_slug="rhombus",
        question_text=f"A rhombus has side {s} cm. What is its perimeter?",
        key=key,
        distractors=_unique_distractors(key, [f"{2 * s} cm", f"{s * s} cm", f"{4 * s + 4} cm", f"{3 * s} cm", f"{4 * s - 4} cm"]),
        explanation=f"All 4 sides of a rhombus are equal: perimeter = 4 x {s} = {4 * s} cm.",
        difficulty="easy", skill_tags=["rhombus", "perimeter"], params={"s": s},
    )


def gen_quad_facts(rng, chapter):
    facts = [
        ("How many diagonals does a quadrilateral have?", "2", ["1", "4", "3"],
         "A quadrilateral has 2 diagonals, each joining opposite vertices."),
        ("In a parallelogram, what can you say about the opposite sides?", "They are equal and parallel",
         ["They are perpendicular", "They bisect the angles", "They are always unequal"],
         "Opposite sides of a parallelogram are equal in length and parallel."),
        ("What is the sum of the angles of any quadrilateral?", "360°", ["180°", "270°", "540°"],
         "Every quadrilateral's four angles add to 360°."),
        ("A quadrilateral with all sides equal and all angles 90° is called a?", "Square",
         ["Rectangle", "Rhombus", "Parallelogram"],
         "All sides equal plus all angles 90° makes it a square."),
        ("Which quadrilateral has exactly one pair of parallel sides?", "Trapezium",
         ["Parallelogram", "Rhombus", "Rectangle"],
         "A trapezium has exactly one pair of parallel sides."),
        ("In a rhombus, the diagonals always?", "Bisect each other at right angles",
         ["Are equal in length", "Are parallel to each other", "Bisect the angles into 45° each"],
         "The diagonals of a rhombus bisect each other at 90°."),
        ("How many sides does a kite have equal in pairs?", "2 pairs of adjacent equal sides",
         ["All 4 sides equal", "Only opposite sides equal", "No equal sides"],
         "A kite has two pairs of adjacent equal sides."),
    ]
    q, key, distractors, expl = rng.choice(facts)
    return AuthoredItem(
        family="quad_facts", chapter=chapter, topic_slug="quadrilateral-properties",
        question_text=q, key=key, distractors=distractors,
        explanation=expl, difficulty="easy", skill_tags=["quadrilaterals"], params={"fact": q},
    )


_QUAD_FACT_ANSWERS = {
    "How many diagonals does a quadrilateral have?": "2",
    "In a parallelogram, what can you say about the opposite sides?": "They are equal and parallel",
    "What is the sum of the angles of any quadrilateral?": "360°",
    "A quadrilateral with all sides equal and all angles 90° is called a?": "Square",
    "Which quadrilateral has exactly one pair of parallel sides?": "Trapezium",
    "In a rhombus, the diagonals always?": "Bisect each other at right angles",
    "How many sides does a kite have equal in pairs?": "2 pairs of adjacent equal sides",
}


def _verify_quad_facts(item):
    return _QUAD_FACT_ANSWERS.get(item.question_text) == item.key and item.key not in item.distractors


def v_hcf(text):
    m = re.search(r"HCF of (\d+) and (\d+)", text)
    a, b = int(m.group(1)), int(m.group(2))
    return str(_math.gcd(a, b))


def gen_hcf(rng, chapter):
    g = rng.randint(3, 15)
    a = g * rng.randint(2, 9)
    b = g * rng.randint(2, 9)
    while a == b or _math.gcd(a, b) != g:
        a = g * rng.randint(2, 9)
        b = g * rng.randint(2, 9)
    key = str(g)
    return AuthoredItem(
        family="hcf", chapter=chapter, topic_slug="hcf-lcm",
        question_text=f"What is the HCF of {a} and {b}?",
        key=key,
        distractors=_unique_distractors(key, [str(a * b // g), str(g * 2), str(g + 1), str(g - 1 if g > 1 else g + 3), str(a)]),
        explanation=f"{a} = {g} x {a // g} and {b} = {g} x {b // g}; no larger common factor, so HCF = {g}.",
        difficulty="medium", skill_tags=["hcf"], params={"a": a, "b": b},
    )


def v_lcm(text):
    m = re.search(r"LCM of (\d+) and (\d+)", text)
    a, b = int(m.group(1)), int(m.group(2))
    return str(a * b // _math.gcd(a, b))


def gen_lcm(rng, chapter):
    a = rng.randint(2, 15)
    b = rng.randint(2, 15)
    while a == b:
        b = rng.randint(2, 15)
    key = str(a * b // _math.gcd(a, b))
    return AuthoredItem(
        family="lcm", chapter=chapter, topic_slug="hcf-lcm",
        question_text=f"What is the LCM of {a} and {b}?",
        key=key,
        distractors=_unique_distractors(key, [str(a * b), str(_math.gcd(a, b)), str(a + b), str(int(key) + a), str(int(key) - a if int(key) > a else int(key) + 2 * a)]),
        explanation=f"LCM x HCF = {a} x {b} = {a * b}; HCF is {_math.gcd(a, b)}, so LCM = {key}.",
        difficulty="medium", skill_tags=["lcm"], params={"a": a, "b": b},
    )


def v_divisibility(text):
    nums = _nums(text)
    units, d = int(nums[1]), int(nums[-1])
    for x in range(10):
        if (400 + 10 * x + units) % d == 0:
            return str(x)
    raise AssertionError("none")


def gen_divisibility(rng, chapter):
    d = rng.choice([3, 9, 3, 9])
    units = rng.randint(0, 9)
    target_total = 4 + units
    add = (d - target_total % d) % d
    x = add
    key = str(x)
    return AuthoredItem(
        family="divisibility", chapter=chapter, topic_slug="divisibility",
        question_text=f"What is the smallest digit that can fill the blank in 4_{units} so that the 3-digit number 4_{units} is divisible by {d}? (The blank is the tens digit.)",
        key=key,
        distractors=_unique_distractors(key, [str((x + 1) % 10), str((x + 2) % 10), str((x + 3) % 10), str((x + d) % 10), str((x + 5) % 10)]),
        explanation=f"Divisible by {d}: digit sum 4 + x + {units} must be a multiple of {d}. Smallest x = {key}.",
        difficulty="hard", skill_tags=["divisibility"], params={"units": units, "d": d},
    )


def v_prime_check(text):
    m = re.search(r"smallest prime number greater than (\d+)", text)
    n = int(m.group(1))
    def is_prime(k):
        return k > 1 and all(k % i for i in range(2, int(k ** 0.5) + 1))
    k = n + 1
    while not is_prime(k):
        k += 1
    return str(k)


def gen_prime_check(rng, chapter):
    n = rng.randint(10, 60)
    def is_prime(k):
        return k > 1 and all(k % i for i in range(2, int(k ** 0.5) + 1))
    k = n + 1
    while not is_prime(k):
        k += 1
    key = str(k)
    return AuthoredItem(
        family="prime_check", chapter=chapter, topic_slug="primes",
        question_text=f"What is the smallest prime number greater than {n}?",
        key=key,
        distractors=_unique_distractors(key, [str(k + 1), str(k + 2), str(n + 2), str(n + 1), str(k - 2)]),
        explanation=f"Test each number after {n} for factors; the first prime is {key}.",
        difficulty="medium", skill_tags=["primes"], params={"n": n},
    )


def v_remainder(text):
    m = re.search(r"remainder when (\d+) is divided by (\d+)", text)
    a, b = int(m.group(1)), int(m.group(2))
    return str(a % b)


def gen_remainder(rng, chapter):
    b = rng.randint(3, 12)
    a = rng.randint(b * 3, b * 30)
    r = a % b
    if r == 0:
        a += 1
        r = 1
    key = str(r)
    return AuthoredItem(
        family="remainder", chapter=chapter, topic_slug="remainders",
        question_text=f"What is the remainder when {a} is divided by {b}?",
        key=key,
        distractors=_unique_distractors(key, [str(r + 1), str(b - r), str(a // b), str(r - 1 if r > 0 else r + 2), str(r + 2)]),
        explanation=f"{a} = {b} x {a // b} + {r}, so the remainder is {r}.",
        difficulty="easy", skill_tags=["remainders"], params={"a": a, "b": b},
    )


def v_distributive(text):
    m = re.search(r"(\d+) x (\d+) \+ (\d+) x (\d+)\?", text) or re.search(r"(\d+) x (\d+) - (\d+) x (\d+)\?", text)
    a, b, c, d = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return str(a * (b + d)) if "+" in text.split("What is")[1] else str(a * (b - d))


def gen_distributive(rng, chapter):
    a = rng.randint(3, 12)
    d = rng.randint(2, 9)
    b = rng.randint(2, 9)
    while b == d:
        b = rng.randint(2, 9)
    key = str(a * (b + d))
    return AuthoredItem(
        family="distributive", chapter=chapter, topic_slug="distributive-property",
        question_text=f"What is {a} x {b} + {a} x {d}?",
        key=key,
        distractors=_unique_distractors(key, [str(a * b + d), str(a * (b - d) if b > d else a * (d - b)), str(a * b * d), str(a * (b + d) + a), str(a * (b + d) - a)]),
        explanation=f"{a} x {b} + {a} x {d} = {a} x ({b}+{d}) = {a} x {b + d} = {key}.",
        difficulty="medium", skill_tags=["distributive"], params={"a": a, "b": b, "d": d},
    )


def v_mental_98(text):
    m = re.search(r"what is (\d+) x (\d+)\?", text, re.IGNORECASE)
    a, b = int(m.group(1)), int(m.group(2))
    return str(a * b)


def gen_mental_98(rng, chapter):
    base = rng.choice([98, 99, 101, 102, 97, 103])
    n = rng.randint(3, 12)
    key = str(base * n)
    return AuthoredItem(
        family="mental_98", chapter=chapter, topic_slug="mental-math",
        question_text=f"Use the distributive property to compute quickly: what is {base} x {n}?",
        key=key,
        distractors=_unique_distractors(key, [str(base * n + n), str(base * n - n), str(base * n + 10), str((base + 1) * n), str((base - 1) * n), str(base * n + base), str(base * n - base), str(base * (n + 1)), str(base * (n - 1)), str(base * n + 2 * n)]),
        explanation=f"{base} x {n} = (100 {'-' if base < 100 else '+'} {abs(base - 100)}) x {n} = {100 * n} {'-' if base < 100 else '+'} {abs(base - 100) * n} = {key}.",
        difficulty="medium", skill_tags=["distributive", "mental-math"], params={"base": base, "n": n},
    )


def v_factorise(text):
    m = re.search(r"common factor from (\d+)x \+ (\d+)", text)
    a, b = int(m.group(1)), int(m.group(2))
    g = _math.gcd(a, b)
    assert g > 1
    return f"{g}(x + {b // g})" if a // g == 1 else f"{g}({a // g}x + {b // g})"


def gen_factorise(rng, chapter):
    g = rng.randint(2, 9)
    p = rng.choice([1, 2, 3, 4])
    q = rng.randint(2, 9)
    while _math.gcd(p, q) != 1:
        q = rng.randint(2, 9)
    a, b = g * p, g * q
    key = f"{g}(x + {q})" if p == 1 else f"{g}({p}x + {q})"
    return AuthoredItem(
        family="factorise", chapter=chapter, topic_slug="factorisation",
        question_text=f"Factorise by taking out the common factor from {a}x + {b}.",
        key=key,
        distractors=_unique_distractors(key, [f"{a}(x + {b})", f"{g}({p}x + {q + 1})" if p > 1 else f"{g}(x + {q + 1})", f"{g + 1}({p}x + {q})" if p > 1 else f"{g + 1}(x + {q})", f"{b}({a}x + 1)"]),
        explanation=f"HCF of {a} and {b} is {g}: {a}x + {b} = {key}.",
        difficulty="medium", skill_tags=["factorisation"], params={"a": a, "b": b},
    )


def v_identity_eval(text):
    m = re.search(r"\((\d+) \+ (\d+)\)\^2 - \((\d+) - (\d+)\)\^2", text)
    a, b, c, d = map(int, m.groups())
    return str((a + b) ** 2 - (c - d) ** 2)


def gen_identity_eval(rng, chapter):
    a = rng.randint(3, 15)
    b = rng.randint(2, 12)
    c = rng.randint(5, 15)
    d = rng.randint(1, c - 1)
    v1, v2 = (a + b) ** 2, (c - d) ** 2
    while v1 <= v2:
        a += 3
        v1 = (a + b) ** 2
    key = str(v1 - v2)
    return AuthoredItem(
        family="identity_eval", chapter=chapter, topic_slug="identities",
        question_text=f"Using identities, what is ({a} + {b})^2 - ({c} - {d})^2?",
        key=key,
        distractors=_unique_distractors(key, [str(v1 + v2), str(v1 - v2 + 1), str(v1 - v2 - 1), str(v1), str(v2)]),
        explanation=f"({a}+{b})^2 = {v1} and ({c}-{d})^2 = {v2}; difference = {key}.",
        difficulty="hard", skill_tags=["identities"], params={"a": a, "b": b, "c": c, "d": d},
    )


def v_pyth_leg(text):
    m = re.search(r"hypotenuse (\d+) cm and one side (\d+) cm", text)
    c, a = int(m.group(1)), int(m.group(2))
    b2 = c * c - a * a
    b = _math.isqrt(b2)
    assert b * b == b2
    return f"{b} cm"


def gen_pyth_leg(rng, chapter):
    a, b, c = rng.choice([(3, 4, 5), (6, 8, 10), (5, 12, 13), (9, 12, 15), (8, 15, 17), (12, 16, 20), (7, 24, 25), (20, 21, 29)])
    key = f"{b} cm"
    return AuthoredItem(
        family="pyth_leg", chapter=chapter, topic_slug="finding-sides",
        question_text=f"A right-angled triangle has hypotenuse {c} cm and one side {a} cm. What is the other side?",
        key=key,
        distractors=_unique_distractors(key, [f"{b + 1} cm", f"{b - 1} cm", f"{c - a} cm", f"{c + a} cm", f"{b + 2} cm"]),
        explanation=f"Other side = square root of ({c}^2 - {a}^2) = square root of {c * c - a * a} = {b} cm.",
        difficulty="medium", skill_tags=["pythagoras"], params={"a": a, "b": b, "c": c},
    )


def v_pyth_ladder(text):
    m = re.search(r"ladder (\d+) m long leans.*base is (\d+) m", text)
    c, a = int(m.group(1)), int(m.group(2))
    b2 = c * c - a * a
    b = _math.isqrt(b2)
    assert b * b == b2
    return f"{b} m"


def gen_pyth_ladder(rng, chapter):
    a, b, c = rng.choice([(3, 4, 5), (6, 8, 10), (5, 12, 13), (9, 12, 15), (12, 16, 20)])
    key = f"{b} m"
    return AuthoredItem(
        family="pyth_ladder", chapter=chapter, topic_slug="applications",
        question_text=f"A ladder {c} m long leans against a wall. Its base is {a} m from the wall. How high up the wall does it reach?",
        key=key,
        distractors=_unique_distractors(key, [f"{b + 1} m", f"{b - 1} m", f"{c - a} m", f"{c} m", f"{b + 2} m"]),
        explanation=f"Height = square root of ({c}^2 - {a}^2) = square root of {c * c - a * a} = {b} m.",
        difficulty="medium", skill_tags=["pythagoras", "applications"], params={"a": a, "b": b, "c": c},
    )


def v_triple_scale(text):
    m = re.search(r"3, 4, 5 are multiplied by (\d+)", text)
    k = int(m.group(1))
    return f"{3 * k}, {4 * k}, {5 * k}"


def gen_triple_scale(rng, chapter):
    k = rng.randint(2, 9)
    key = f"{3 * k}, {4 * k}, {5 * k}"
    return AuthoredItem(
        family="triple_scale", chapter=chapter, topic_slug="pythagorean-triples",
        question_text=f"The numbers 3, 4, 5 are multiplied by {k}. The resulting triple is?",
        key=key,
        distractors=_unique_distractors(key, [f"{3 * k}, {4 * k}, {5 * k + k}", f"{3 + k}, {4 + k}, {5 + k}", f"{3 * k}, {4 * k}, {5 * k - k}", f"{3 * k + 1}, {4 * k + 1}, {5 * k + 1}", f"{4 * k}, {3 * k}, {5 * k}"]),
        explanation=f"Multiply each by {k}: {3 * k}, {4 * k}, {5 * k} - and (3k)^2 + (4k)^2 = (5k)^2 still holds.",
        difficulty="easy", skill_tags=["triples"], params={"k": k},
    )


def gen_converse_check(rng, chapter):
    a, b, c = rng.choice([(3, 4, 5), (6, 8, 10), (5, 12, 13), (8, 15, 17), (9, 12, 15)])
    while True:
        x, y, z = sorted(rng.randint(2, 26) for _ in range(3))
        if x * x + y * y != z * z:
            break
    right = rng.random() < 0.5
    if right:
        sides = (a, b, c)
        key = "Yes, it is right-angled"
    else:
        sides = (x, y, z)
        key = "No, it is not right-angled"
    p, q, r = sides
    return AuthoredItem(
        family="converse_check", chapter=chapter, topic_slug="converse",
        question_text=f"A triangle has sides {p} cm, {q} cm, {r} cm. Is it right-angled?",
        key=key,
        distractors=["No, it is not right-angled" if right else "Yes, it is right-angled", "Only if the largest side is the base", "Cannot be decided from the sides"],
        explanation=f"{p}^2 + {q}^2 = {p * p + q * q} and {r}^2 = {r * r}. {'Equal' if p * p + q * q == r * r else 'Not equal'}, so {'it is' if p * p + q * q == r * r else 'it is not'} right-angled.",
        difficulty="medium", skill_tags=["converse", "pythagoras"], params={"sides": sides},
    )


def v_coord_x(text):
    m = re.search(r"point \((-?\d+), (-?\d+)\)", text)
    return m.group(1)


def gen_coord_x(rng, chapter):
    x = rng.randint(-9, 9)
    y = rng.randint(-9, 9)
    while x == 0 or y == 0:
        x = rng.randint(-9, 9)
        y = rng.randint(-9, 9)
    key = str(x)
    return AuthoredItem(
        family="coord_x", chapter=chapter, topic_slug="coordinates",
        question_text=f"What is the x-coordinate of the point ({x}, {y})?",
        key=key,
        distractors=_unique_distractors(key, [str(y), str(-x), str(x + 1), str(abs(x)), str(x - 1)]),
        explanation=f"In (x, y) the first number is the x-coordinate: {x}.",
        difficulty="easy", skill_tags=["coordinates"], params={"x": x, "y": y},
    )


def v_coord_quadrant(text):
    m = re.search(r"quadrant does the point \((-?\d+), (-?\d+)\)", text)
    x, y = int(m.group(1)), int(m.group(2))
    if x > 0 and y > 0:
        return "First"
    if x < 0 and y > 0:
        return "Second"
    if x < 0 and y < 0:
        return "Third"
    return "Fourth"


def gen_coord_quadrant(rng, chapter):
    x = rng.choice([i for i in range(-9, 10) if i != 0])
    y = rng.choice([i for i in range(-9, 10) if i != 0])
    key = v_coord_quadrant(f"quadrant does the point ({x}, {y})")
    return AuthoredItem(
        family="coord_quadrant", chapter=chapter, topic_slug="plotting-points",
        question_text=f"In which quadrant does the point ({x}, {y}) lie?",
        key=key,
        distractors=[q for q in ["First", "Second", "Third", "Fourth"] if q != key],
        explanation=f"x = {x} ({'positive' if x > 0 else 'negative'}), y = {y} ({'positive' if y > 0 else 'negative'}): {key} quadrant.",
        difficulty="medium", skill_tags=["coordinates", "quadrants"], params={"x": x, "y": y},
    )


def v_axis_distance(text):
    m = re.search(r"points \((-?\d+), (\d+)\) and \((-?\d+), \2\)", text)
    x1, y, x2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return f"{abs(x2 - x1)} units"


def gen_axis_distance(rng, chapter):
    y = rng.randint(1, 9)
    x1 = rng.randint(-8, 4)
    x2 = x1 + rng.randint(2, 10)
    key = f"{x2 - x1} units"
    return AuthoredItem(
        family="axis_distance", chapter=chapter, topic_slug="coordinates",
        question_text=f"What is the distance between the points ({x1}, {y}) and ({x2}, {y})?",
        key=key,
        distractors=_unique_distractors(key, [f"{x2 - x1 + 1} units", f"{x2 - x1 - 1} units", f"{x2 + x1} units", f"{2 * (x2 - x1)} units", f"{y} units"]),
        explanation=f"Same y-coordinate, so distance = |{x2} - {x1}| = {x2 - x1} units.",
        difficulty="medium", skill_tags=["coordinates", "distance"], params={"x1": x1, "x2": x2, "y": y},
    )


def v_graph_read(text):
    m = re.search(r"t = (\d+), its value is (\d+)\. What is its value at t = (\d+)\?", text)
    t1, v1, t2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return str(v1 + (t2 - t1))


def gen_graph_read(rng, chapter):
    t1 = rng.randint(1, 4)
    t2 = t1 + rng.randint(1, 5)
    v1 = rng.randint(2, 10)
    v2 = v1 + (t2 - t1)
    key = str(v2)
    return AuthoredItem(
        family="graph_read", chapter=chapter, topic_slug="line-graphs",
        question_text=f"A line graph rises by 1 unit for every 1 unit of time. At t = {t1}, its value is {v1}. What is its value at t = {t2}?",
        key=key,
        distractors=_unique_distractors(key, [str(v2 + 1), str(v2 - 1), str(v1), str(v1 + t2), str(v2 + 2)]),
        explanation=f"From t = {t1} to t = {t2} is {t2 - t1} steps: {v1} + {t2 - t1} = {v2}.",
        difficulty="easy", skill_tags=["graphs"], params={"t1": t1, "v1": v1, "t2": t2},
    )


def v_area_rect(text):
    m = re.search(r"rectangle (?:is|with length) ?(\d+)? ?c?m? ?(?:by|and width) (\d+)?", text)
    nums = _nums(text)
    l, w = int(nums[0]), int(nums[1])
    return f"{l * w} cm^2"


def gen_area_rect(rng, chapter):
    l = rng.randint(3, 20)
    w = rng.randint(2, 15)
    key = f"{l * w} cm^2"
    return AuthoredItem(
        family="area_rect", chapter=chapter, topic_slug="area-rectangle",
        question_text=f"A rectangle is {l} cm long and {w} cm wide. What is its area?",
        key=key,
        distractors=_unique_distractors(key, [f"{2 * (l + w)} cm^2", f"{l * w + l} cm^2", f"{l * w - w} cm^2", f"{l + w} cm^2", f"{(l + 1) * w} cm^2"]),
        explanation=f"Area = length x width = {l} x {w} = {l * w} cm^2.",
        difficulty="easy", skill_tags=["area"], params={"l": l, "w": w},
    )


def v_area_triangle(text):
    m = re.search(r"base (\d+) cm and height (\d+) cm", text)
    b, h = int(m.group(1)), int(m.group(2))
    assert (b * h) % 2 == 0
    return f"{b * h // 2} cm^2"


def gen_area_triangle(rng, chapter):
    b = rng.randint(4, 24)
    h = rng.randint(2, 16)
    if (b * h) % 2:
        h += 1
    key = f"{b * h // 2} cm^2"
    return AuthoredItem(
        family="area_triangle", chapter=chapter, topic_slug="area-triangle",
        question_text=f"A triangle has base {b} cm and height {h} cm. What is its area?",
        key=key,
        distractors=_unique_distractors(key, [f"{b * h} cm^2", f"{b + h} cm^2", f"{b * h // 2 + b} cm^2", f"{b * h // 4 if (b * h) % 4 == 0 else b * h // 2 + 2} cm^2", f"{b * h // 2 - 2} cm^2"]),
        explanation=f"Area = (1/2) x base x height = (1/2) x {b} x {h} = {b * h // 2} cm^2.",
        difficulty="easy", skill_tags=["area", "triangles"], params={"b": b, "h": h},
    )


def v_area_pgram(text):
    m = re.search(r"parallelogram has base (\d+) cm and (?:vertical )?height (\d+) cm", text)
    b, h = int(m.group(1)), int(m.group(2))
    return f"{b * h} cm^2"


def gen_area_pgram(rng, chapter):
    b = rng.randint(4, 20)
    h = rng.randint(3, 15)
    key = f"{b * h} cm^2"
    return AuthoredItem(
        family="area_pgram", chapter=chapter, topic_slug="area-parallelogram",
        question_text=f"A parallelogram has base {b} cm and vertical height {h} cm. What is its area?",
        key=key,
        distractors=_unique_distractors(key, [f"{b * h // 2} cm^2", f"{2 * (b + h)} cm^2", f"{b * h + b} cm^2", f"{b + h} cm^2", f"{b * h - h} cm^2"]),
        explanation=f"Area of a parallelogram = base x height = {b} x {h} = {b * h} cm^2.",
        difficulty="easy", skill_tags=["area", "parallelogram"], params={"b": b, "h": h},
    )


def v_area_trapezium(text):
    m = re.search(r"parallel sides (\d+) cm and (\d+) cm,.*(\d+) cm apart", text)
    a, b, h = int(m.group(1)), int(m.group(2)), int(m.group(3))
    assert ((a + b) * h) % 2 == 0
    return f"{(a + b) * h // 2} cm^2"


def gen_area_trapezium(rng, chapter):
    a = rng.randint(3, 15)
    b = rng.randint(3, 15)
    while a == b or (a + b) % 2:
        b = rng.randint(3, 15)
    h = rng.choice([2, 4, 6, 8, 10, 3, 5])
    key = f"{(a + b) * h // 2} cm^2"
    return AuthoredItem(
        family="area_trapezium", chapter=chapter, topic_slug="area-trapezium",
        question_text=f"A trapezium has parallel sides {a} cm and {b} cm, and they are {h} cm apart. What is its area?",
        key=key,
        distractors=_unique_distractors(key, [f"{(a + b) * h} cm^2", f"{a * b} cm^2", f"{(a + b + h)} cm^2", f"{(a + b) * h // 2 + a} cm^2", f"{(a + b) * h // 4} cm^2"]),
        explanation=f"Area = (1/2) x (sum of parallel sides) x height = (1/2) x ({a}+{b}) x {h} = {(a + b) * h // 2} cm^2.",
        difficulty="medium", skill_tags=["area", "trapezium"], params={"a": a, "b": b, "h": h},
    )


def v_area_composite(text):
    nums = _nums(text)
    s, w = int(nums[0]), int(nums[2])
    return f"{s * s + (s * w) // 2} cm^2"


def gen_area_composite(rng, chapter):
    s = rng.choice([4, 6, 8, 10, 12])
    w = rng.choice([2, 4, 6, 8])
    key = f"{s * s + (s * w) // 2} cm^2"
    return AuthoredItem(
        family="area_composite", chapter=chapter, topic_slug="composite-shapes",
        question_text=f"A shape is made of a square of side {s} cm with a triangle on top. The triangle has base {s} cm (the top of the square) and height {w} cm. What is the total area?",
        key=key,
        distractors=_unique_distractors(key, [f"{s * s + s * w} cm^2", f"{s * s} cm^2", f"{(s * w) // 2} cm^2", f"{s * s + w} cm^2", f"{(s + w) ** 2} cm^2"]),
        explanation=f"Square = {s}^2 = {s * s}; triangle = (1/2) x {s} x {w} = {(s * w) // 2}. Total = {s * s + (s * w) // 2} cm^2.",
        difficulty="medium", skill_tags=["area", "composite"], params={"s": s, "w": w},
    )


def v_perim_rect(text):
    nums = _nums(text)
    l, w = int(nums[0]), int(nums[1])
    return f"{2 * (l + w)} cm"


def gen_perim_rect(rng, chapter):
    l = rng.randint(5, 25)
    w = rng.randint(3, 18)
    key = f"{2 * (l + w)} cm"
    return AuthoredItem(
        family="perim_rect", chapter=chapter, topic_slug="perimeter-area",
        question_text=f"A rectangle has length {l} cm and width {w} cm. What is its perimeter?",
        key=key,
        distractors=_unique_distractors(key, [f"{l * w} cm", f"{l + w} cm", f"{2 * (l + w) + 2} cm", f"{2 * (l + w) - 2} cm", f"{4 * l} cm"]),
        explanation=f"Perimeter = 2 x (length + width) = 2 x ({l}+{w}) = {2 * (l + w)} cm.",
        difficulty="easy", skill_tags=["perimeter"], params={"l": l, "w": w},
    )


def v_area_convert(text):
    m = re.search(r"(\d+) m\^2 in square centimetres", text)
    n = int(m.group(1))
    return f"{n * 10000} cm^2"


def gen_area_convert(rng, chapter):
    n = rng.randint(2, 15)
    key = f"{n * 10000} cm^2"
    return AuthoredItem(
        family="area_convert", chapter=chapter, topic_slug="units-conversion",
        question_text=f"What is {n} m^2 in square centimetres?",
        key=key,
        distractors=_unique_distractors(key, [f"{n * 100} cm^2", f"{n * 1000} cm^2", f"{n * 100000} cm^2", f"{n * 10000 + 1000} cm^2", f"{n * 10} cm^2"]),
        explanation=f"1 m^2 = 100 cm x 100 cm = 10000 cm^2, so {n} m^2 = {n * 10000} cm^2.",
        difficulty="medium", skill_tags=["units", "area"], params={"n": n},
    )


def v_area_context(text):
    nums = _nums(text)
    l, w, c = int(nums[0]), int(nums[1]), int(nums[2])
    return f"Rs{l * w * c}"


def gen_area_context(rng, chapter):
    l = rng.randint(3, 12)
    w = rng.randint(3, 10)
    c = rng.choice([50, 100, 150, 200, 250])
    key = f"Rs{l * w * c}"
    return AuthoredItem(
        family="area_context", chapter=chapter, topic_slug="area-in-context",
        question_text=f"A floor {l} m long and {w} m wide is to be tiled. Tiles cost Rs{c} per square metre. What is the total cost of the tiles?",
        key=key,
        distractors=_unique_distractors(key, [f"Rs{l * w * c + c}", f"Rs{(l + w) * c}", f"Rs{2 * (l + w) * c}", f"Rs{l * w * c - c}", f"Rs{l * w * c * 2}"]),
        explanation=f"Area = {l} x {w} = {l * w} m^2. Cost = {l * w} x Rs{c} = Rs{l * w * c}.",
        difficulty="medium", skill_tags=["area", "context"], params={"l": l, "w": w, "c": c},
    )


VERIFIERS.update({
    "square_eval": v_square_eval,
    "sqrt_perfect": v_sqrt_perfect,
    "cube_eval": v_cube_eval,
    "cuberoot_perfect": v_cuberoot_perfect,
    "odd_sum": v_odd_sum,
    "square_area": v_square_area,
    "consec_squares": v_consec_squares,
    "pgram_opp": v_pgram_opp,
    "rect_diag": v_rect_diag,
    "rhombus_perimeter": v_rhombus_perimeter,
    "hcf": v_hcf,
    "lcm": v_lcm,
    "divisibility": v_divisibility,
    "prime_check": v_prime_check,
    "remainder": v_remainder,
    "distributive": v_distributive,
    "mental_98": v_mental_98,
    "factorise": v_factorise,
    "identity_eval": v_identity_eval,
    "pyth_leg": v_pyth_leg,
    "pyth_ladder": v_pyth_ladder,
    "triple_scale": v_triple_scale,
    "coord_x": v_coord_x,
    "coord_quadrant": v_coord_quadrant,
    "axis_distance": v_axis_distance,
    "graph_read": v_graph_read,
    "area_rect": v_area_rect,
    "area_triangle": v_area_triangle,
    "area_pgram": v_area_pgram,
    "area_trapezium": v_area_trapezium,
    "area_composite": v_area_composite,
    "perim_rect": v_perim_rect,
    "area_convert": v_area_convert,
    "area_context": v_area_context,
})

SET_VERIFIERS.update({
    "quad_facts": _verify_quad_facts,
})

GENERATORS.update({
    1: [
        (gen_square_eval, v_square_eval, 7),
        (gen_sqrt_perfect, v_sqrt_perfect, 7),
        (gen_cube_eval, v_cube_eval, 6),
        (gen_cuberoot_perfect, v_cuberoot_perfect, 6),
        (gen_odd_sum, v_odd_sum, 5),
        (gen_square_area, v_square_area, 5),
        (gen_consec_squares, v_consec_squares, 5),
    ],
    4: [
        (gen_quad_angle, verify_quad_angle, 6),
        (gen_pgram_opp, v_pgram_opp, 7),
        (gen_rect_diag, v_rect_diag, 6),
        (gen_rhombus_perimeter, v_rhombus_perimeter, 6),
        (gen_quad_facts, None, 7),
        (gen_symmetry, verify_symmetry, 5),
        (gen_polygon_sum, verify_polygon_sum, 5),
    ],
    5: [
        (gen_hcf, v_hcf, 7),
        (gen_lcm, v_lcm, 7),
        (gen_divisibility, v_divisibility, 6),
        (gen_prime_check, v_prime_check, 7),
        (gen_remainder, v_remainder, 7),
        (gen_last_digit, v_last_digit, 4),
        (gen_power, v_power, 4),
    ],
    6: [
        (gen_distributive, v_distributive, 8),
        (gen_mental_98, v_mental_98, 7),
        (gen_factorise, v_factorise, 7),
        (gen_identity_eval, v_identity_eval, 6),
        (gen_expand_product, verify_expand_product, 5),
        (gen_like_terms, verify_like_terms, 5),
        (gen_square_sum, verify_square_sum, 4),
    ],
    9: [
        (gen_pyth_leg, v_pyth_leg, 8),
        (gen_pyth_ladder, v_pyth_ladder, 6),
        (gen_triple_scale, v_triple_scale, 6),
        (gen_converse_check, None, 8),
        (gen_right_set, None, 6),
        (gen_power, v_power, 4),
    ],
    12: [
        (gen_coord_x, v_coord_x, 8),
        (gen_coord_quadrant, v_coord_quadrant, 8),
        (gen_axis_distance, v_axis_distance, 8),
        (gen_graph_read, v_graph_read, 7),
        (gen_nth_term, verify_nth_term, 5),
    ],
    14: [
        (gen_area_rect, v_area_rect, 6),
        (gen_area_triangle, v_area_triangle, 6),
        (gen_area_pgram, v_area_pgram, 6),
        (gen_area_trapezium, v_area_trapezium, 6),
        (gen_area_composite, v_area_composite, 5),
        (gen_perim_rect, v_perim_rect, 6),
        (gen_area_convert, v_area_convert, 4),
        (gen_area_context, v_area_context, 6),
    ],
})


def _verify_converse_check(item):
    m = re.search(r"sides (\d+) cm, (\d+) cm, (\d+) cm", item.question_text)
    a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
    is_right = a * a + b * b == c * c
    expected = "Yes, it is right-angled" if is_right else "No, it is not right-angled"
    return item.key == expected and expected not in item.distractors


SET_VERIFIERS["converse_check"] = _verify_converse_check


# ---------------- Chapter 8: Fractions in Disguise ----------------

def v_percent_of(text):
    m = re.search(r"What is (\d+)% of (\d+)\?", text)
    p, n = int(m.group(1)), int(m.group(2))
    assert (p * n) % 100 == 0
    return str(p * n // 100)


def gen_percent_of(rng, chapter):
    while True:
        p = rng.choice([5, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 75, 80])
        n = rng.choice([40, 60, 80, 120, 150, 200, 240, 250, 300, 400, 500, 600, 800])
        if (p * n) % 100 == 0:
            break
    key = str(p * n // 100)
    return AuthoredItem(
        family="percent_of", chapter=chapter, topic_slug="percent-of",
        question_text=f"What is {p}% of {n}?",
        key=key,
        distractors=_unique_distractors(key, [str(p * n // 100 + p), str(p * n // 100 - p), str(p * n // 10), str(p * n // 100 * 2), str(p * n // 100 + 10), str(p * n // 100 // 2 if p * n // 200 > 0 else p * n // 100 + 5), str(p * n // 100 + n // 10)]),
        explanation=f"{p}% of {n} = ({p}/100) x {n} = {key}.",
        difficulty="easy", skill_tags=["percentage"], params={"p": p, "n": n},
    )


def v_frac_to_percent(text):
    m = re.search(r"Write (\d+)/(\d+) as a percentage", text)
    a, b = int(m.group(1)), int(m.group(2))
    assert (a * 100) % b == 0
    return f"{a * 100 // b}%"


def gen_frac_to_percent(rng, chapter):
    while True:
        b = rng.choice([2, 4, 5, 8, 10, 20, 25, 50])
        a = rng.randint(1, b - 1)
        if (a * 100) % b == 0:
            break
    key = f"{a * 100 // b}%"
    return AuthoredItem(
        family="frac_to_percent", chapter=chapter, topic_slug="percent-conversion",
        question_text=f"Write {a}/{b} as a percentage.",
        key=key,
        distractors=_unique_distractors(key, [f"{a * 100 // b + 5}%", f"{a * 100 // b - 5}%", f"{a * 100 // b * 2}%", f"{(b - a) * 100 // b}%", f"{a * 10}%", f"{a * 100 // b + 10}%", f"{a * 100 // b // 2 if a * 100 // b > 20 else a * 100 // b + 25}%"]),
        explanation=f"{a}/{b} = ({a} x 100)/{b} % = {key}.",
        difficulty="easy", skill_tags=["percentage", "fractions"], params={"a": a, "b": b},
    )


_DEC_FRAC = [("0.5", "1/2"), ("0.25", "1/4"), ("0.75", "3/4"), ("0.2", "1/5"), ("0.4", "2/5"), ("0.6", "3/5"), ("0.8", "4/5"), ("0.125", "1/8"), ("0.375", "3/8"), ("0.05", "1/20"), ("0.35", "7/20"), ("0.45", "9/20"), ("0.15", "3/20"), ("0.625", "5/8"), ("0.875", "7/8"), ("0.1", "1/10"), ("0.3", "3/10"), ("0.7", "7/10"), ("0.9", "9/10")]


def gen_decimal_to_fraction(rng, chapter):
    dec, frac = rng.choice(_DEC_FRAC)
    num, den = frac.split("/")
    return AuthoredItem(
        family="decimal_to_fraction", chapter=chapter, topic_slug="fraction-decimal",
        question_text=f"Write {dec} as a fraction in its simplest form.",
        key=frac,
        distractors=_unique_distractors(frac, [f"{num}/{int(den) + 1}", f"{int(num) + 1}/{den}", f"{den}/{num}", f"{int(num) * 2}/{den}", f"{num}/{int(den) * 2}", f"{int(num) + 2}/{int(den) + 4}", f"1/{int(den) // 2 if int(den) % 2 == 0 else int(den) + 1}"]),
        explanation=f"{dec} = {num}/{den} after dividing numerator and denominator by their common factor.",
        difficulty="easy", skill_tags=["decimals", "fractions"], params={"dec": dec},
    )


def v_decimal_to_fraction(text):
    m = re.search(r"Write (0\.\d+) as a fraction", text)
    dec = m.group(1)
    return dict(_DEC_FRAC)[dec]


def _pct_change(rng, direction):
    while True:
        n = rng.choice([200, 240, 250, 300, 400, 500, 600, 800, 1000, 1200, 1500])
        p = rng.choice([5, 10, 12, 15, 20, 25, 30, 40, 50])
        if (n * p) % 100 == 0:
            break
    delta = n * p // 100
    return n, p, n + delta if direction == "increases" else n - delta, delta


def v_percent_change(text):
    m = re.search(r"Rs (\d+).*?(increases|decreases) by (\d+)%", text)
    n, direction, p = int(m.group(1)), m.group(2), int(m.group(3))
    delta = n * p // 100
    return f"Rs {n + delta}" if direction == "increases" else f"Rs {n - delta}"


def gen_percent_increase(rng, chapter):
    n, p, new, delta = _pct_change(rng, "increases")
    key = f"Rs {new}"
    return AuthoredItem(
        family="percent_increase", chapter=chapter, topic_slug="increase-decrease",
        question_text=f"The price of a bag is Rs {n}. It increases by {p}%. What is the new price?",
        key=key,
        distractors=_unique_distractors(key, [f"Rs {n - delta}", f"Rs {n + p}", f"Rs {new + delta}", f"Rs {n + 2 * delta}", f"Rs {new - p}", f"Rs {n + delta // 2 if delta > 1 else new + 5}", f"Rs {delta}"]),
        explanation=f"Increase = {p}% of {n} = {delta}. New price = {n} + {delta} = Rs {new}.",
        difficulty="medium", skill_tags=["percentage", "increase"], params={"n": n, "p": p},
    )


def gen_percent_decrease(rng, chapter):
    n, p, new, delta = _pct_change(rng, "decreases")
    key = f"Rs {new}"
    return AuthoredItem(
        family="percent_decrease", chapter=chapter, topic_slug="increase-decrease",
        question_text=f"A jacket costs Rs {n}. Its price decreases by {p}% in a sale. What is the sale price?",
        key=key,
        distractors=_unique_distractors(key, [f"Rs {n + delta}", f"Rs {n - p}", f"Rs {new - delta}", f"Rs {n - 2 * delta}", f"Rs {new + p}", f"Rs {delta}", f"Rs {n - delta // 2 if delta > 1 else new - 5}"]),
        explanation=f"Discount = {p}% of {n} = {delta}. Sale price = {n} - {delta} = Rs {new}.",
        difficulty="medium", skill_tags=["percentage", "decrease"], params={"n": n, "p": p},
    )


def _profit_pick(rng, want_profit):
    while True:
        c = rng.choice([100, 200, 250, 400, 500, 800, 1000, 160, 320, 640])
        pct = rng.choice([5, 10, 20, 25, 50])
        if (c * pct) % 100 == 0:
            break
    s = c + c * pct // 100 if want_profit else c - c * pct // 100
    return c, s, pct


def v_profit_percent(text):
    m = re.search(r"Rs (\d+) and sells it for Rs (\d+)", text)
    c, s = int(m.group(1)), int(m.group(2))
    if "profit" in text:
        assert s > c and (s - c) * 100 % c == 0
        return f"{(s - c) * 100 // c}%"
    assert c > s and (c - s) * 100 % c == 0
    return f"{(c - s) * 100 // c}%"


def gen_profit_percent(rng, chapter):
    c, s, pct = _profit_pick(rng, True)
    key = f"{pct}%"
    return AuthoredItem(
        family="profit_percent", chapter=chapter, topic_slug="profit-loss",
        question_text=f"A shopkeeper buys a toy for Rs {c} and sells it for Rs {s}. What is the profit percentage?",
        key=key,
        distractors=_unique_distractors(key, [f"{pct + 5}%", f"{pct - 5 if pct > 5 else pct + 10}%", f"{pct * 2}%", f"{(s - c) * 100 // s}%", f"{pct + 10}%", f"Rs {s - c}", f"{pct // 2 if pct % 2 == 0 and pct > 2 else pct + 15}%"]),
        explanation=f"Profit = {s} - {c} = {s - c}. Profit% = ({s - c}/{c}) x 100 = {key}.",
        difficulty="medium", skill_tags=["profit-loss"], params={"c": c, "s": s},
    )


def gen_loss_percent(rng, chapter):
    c, s, pct = _profit_pick(rng, False)
    key = f"{pct}%"
    return AuthoredItem(
        family="loss_percent", chapter=chapter, topic_slug="profit-loss",
        question_text=f"A trader buys a chair for Rs {c} and sells it for Rs {s}. What is the loss percentage?",
        key=key,
        distractors=_unique_distractors(key, [f"{pct + 5}%", f"{pct - 5 if pct > 5 else pct + 10}%", f"{pct * 2}%", f"{(c - s) * 100 // c + 5}%", f"{pct + 10}%", f"Rs {c - s}", f"{pct // 2 if pct % 2 == 0 and pct > 2 else pct + 15}%"]),
        explanation=f"Loss = {c} - {s} = {c - s}. Loss% = ({c - s}/{c}) x 100 = {key}.",
        difficulty="medium", skill_tags=["profit-loss"], params={"c": c, "s": s},
    )


def v_simple_interest(text):
    m = re.search(r"Rs (\d+) at (\d+)% per year for (\d+) year", text)
    p, r, t = int(m.group(1)), int(m.group(2)), int(m.group(3))
    assert (p * r * t) % 100 == 0
    return f"Rs {p * r * t // 100}"


def gen_simple_interest(rng, chapter):
    while True:
        p = rng.choice([1000, 2000, 2500, 4000, 5000, 8000, 10000, 1200])
        r = rng.choice([3, 4, 5, 6, 8, 10, 12])
        t = rng.choice([1, 2, 3, 4, 5])
        if (p * r * t) % 100 == 0:
            break
    si = p * r * t // 100
    key = f"Rs {si}"
    return AuthoredItem(
        family="simple_interest", chapter=chapter, topic_slug="interest",
        question_text=f"Find the simple interest on Rs {p} at {r}% per year for {t} years.",
        key=key,
        distractors=_unique_distractors(key, [f"Rs {si + 100}", f"Rs {si - 100 if si > 100 else si + 50}", f"Rs {si * 2}", f"Rs {p * r // 100}", f"Rs {si // 2 if si % 2 == 0 else si + 25}", f"Rs {si + p // 10}", f"Rs {si + 50}"]),
        explanation=f"SI = (P x R x T)/100 = ({p} x {r} x {t})/100 = Rs {si}.",
        difficulty="medium", skill_tags=["interest"], params={"p": p, "r": r, "t": t},
    )


def v_gst_price(text):
    m = re.search(r"Rs (\d+) plus (\d+)% GST", text)
    n, g = int(m.group(1)), int(m.group(2))
    assert (n * g) % 100 == 0
    return f"Rs {n + n * g // 100}"


def gen_gst_price(rng, chapter):
    while True:
        n = rng.choice([200, 400, 500, 1000, 1500, 2000, 2500, 800])
        g = rng.choice([5, 12, 18, 28])
        if (n * g) % 100 == 0:
            break
    total = n + n * g // 100
    key = f"Rs {total}"
    return AuthoredItem(
        family="gst_price", chapter=chapter, topic_slug="tax",
        question_text=f"A phone case is priced at Rs {n} plus {g}% GST. What is the final price a customer pays?",
        key=key,
        distractors=_unique_distractors(key, [f"Rs {n}", f"Rs {n + g}", f"Rs {total + n // 10}", f"Rs {total - n * g // 200 if n * g // 200 > 0 else total + 5}", f"Rs {n + 2 * n * g // 100}", f"Rs {n * g // 100}", f"Rs {total + g}"]),
        explanation=f"GST = {g}% of {n} = {n * g // 100}. Final price = {n} + {n * g // 100} = Rs {total}.",
        difficulty="medium", skill_tags=["tax", "gst"], params={"n": n, "g": g},
    )


# ---------------- Chapter 10: Proportional Reasoning-2 ----------------

def v_inverse_prop(text):
    m = re.search(r"(\d+) workers can finish a job in (\d+) days\D*(\d+) workers", text)
    x, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    assert (x * d) % y == 0
    return f"{x * d // y} days"


def gen_inverse_prop(rng, chapter):
    while True:
        x = rng.choice([4, 6, 8, 10, 12, 15])
        d = rng.choice([6, 8, 9, 10, 12, 15, 18, 20])
        y = rng.choice([2, 3, 4, 5, 6, 8, 9, 24, 30])
        if y != x and (x * d) % y == 0:
            break
    key = f"{x * d // y} days"
    return AuthoredItem(
        family="inverse_prop", chapter=chapter, topic_slug="inverse-proportion",
        question_text=f"{x} workers can finish a job in {d} days. Working at the same rate, how many days will {y} workers take?",
        key=key,
        distractors=_unique_distractors(key, [f"{x * d // y + 2} days", f"{x * d // y - 2 if x * d // y > 2 else x * d // y + 4} days", f"{d} days", f"{x * d // y * 2} days", f"{y * d // x if (y * d) % x == 0 and y * d // x != x * d // y else x * d // y + 1} days", f"{x * d // y + 1} days", f"{x * d // y + 3} days"]),
        explanation=f"Total work = {x} x {d} = {x * d} worker-days. For {y} workers: {x * d}/{y} = {key}.",
        difficulty="medium", skill_tags=["inverse-proportion"], params={"x": x, "d": d, "y": y},
    )


def v_speed_time(text):
    m = re.search(r"(\d+) km/h for (\d+) hours", text)
    return f"{int(m.group(1)) * int(m.group(2))} km"


def gen_speed_time(rng, chapter):
    s = rng.choice([40, 45, 50, 60, 65, 70, 80, 90])
    t = rng.choice([2, 3, 4, 5, 6])
    key = f"{s * t} km"
    return AuthoredItem(
        family="speed_time", chapter=chapter, topic_slug="speed-distance",
        question_text=f"A bus travels at {s} km/h for {t} hours. How far does it go?",
        key=key,
        distractors=_unique_distractors(key, [f"{s * t + s} km", f"{s * t - s} km", f"{s + t} km", f"{s * t * 2} km", f"{s * (t + 1)} km" if s * (t + 1) != s * t + s else f"{s * t + 2 * s} km", f"{s * t + 10} km", f"{s * t // 2} km"]),
        explanation=f"Distance = speed x time = {s} x {t} = {key}.",
        difficulty="easy", skill_tags=["speed", "distance"], params={"s": s, "t": t},
    )


def v_time_for_distance(text):
    m = re.search(r"cover (\d+) km at a speed of (\d+) km/h", text)
    d, s = int(m.group(1)), int(m.group(2))
    assert d % s == 0
    return f"{d // s} hours"


def gen_time_for_distance(rng, chapter):
    s = rng.choice([40, 50, 60, 70, 80])
    t = rng.choice([2, 3, 4, 5, 6])
    d = s * t
    key = f"{t} hours"
    return AuthoredItem(
        family="time_for_distance", chapter=chapter, topic_slug="speed-distance",
        question_text=f"A train must cover {d} km at a speed of {s} km/h. How long will the journey take?",
        key=key,
        distractors=_unique_distractors(key, [f"{t + 1} hours", f"{t - 1 if t > 1 else t + 2} hours", f"{t * 2} hours", f"{d // (s + 10) if d % (s + 10) == 0 and d // (s + 10) != t else t + 2} hours", f"{t + 3} hours", f"{s // 10} hours", f"{d} hours"]),
        explanation=f"Time = distance / speed = {d}/{s} = {key}.",
        difficulty="easy", skill_tags=["speed", "time"], params={"d": d, "s": s},
    )


_WORK_PAIRS = [(4, 4), (6, 3), (10, 15), (12, 6), (8, 8), (5, 20), (10, 10), (3, 6), (12, 4), (20, 5), (6, 12), (15, 10), (9, 18), (18, 9), (14, 14), (16, 16)]


def v_work_time(text):
    m = re.search(r"in (\d+) days.*same job in (\d+) days", text)
    a, b = int(m.group(1)), int(m.group(2))
    assert (a * b) % (a + b) == 0
    return f"{a * b // (a + b)} days"


def gen_work_time(rng, chapter):
    a, b = rng.choice(_WORK_PAIRS)
    t = a * b // (a + b)
    key = f"{t} days"
    return AuthoredItem(
        family="work_time", chapter=chapter, topic_slug="work-time",
        question_text=f"Asha can do a job in {a} days. Bina can do the same job in {b} days. Working together, how many days will they take?",
        key=key,
        distractors=_unique_distractors(key, [f"{t + 1} days", f"{t - 1 if t > 1 else t + 2} days", f"{(a + b) // 2} days", f"{t * 2} days", f"{min(a, b)} days", f"{t + 2} days", f"{a + b} days"]),
        explanation=f"Together they do 1/{a} + 1/{b} = {a + b}/{a * b} of the job per day, so they need {a * b}/{a + b} = {key}.",
        difficulty="hard", skill_tags=["work-time"], params={"a": a, "b": b},
    )


def v_unitary(text):
    m = re.search(r"(\d+) notebooks cost Rs (\d+)\.*.*cost of (\d+) notebooks", text)
    n, c, m2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
    assert c % n == 0
    return f"Rs {c // n * m2}"


def gen_unitary(rng, chapter):
    n = rng.choice([4, 5, 6, 8, 10, 12])
    u = rng.choice([5, 8, 10, 12, 15, 20, 25, 30])
    c = n * u
    while True:
        m2 = rng.randint(2, 15)
        if m2 != n:
            break
    key = f"Rs {m2 * u}"
    return AuthoredItem(
        family="unitary_cost", chapter=chapter, topic_slug="rates",
        question_text=f"{n} notebooks cost Rs {c}. What is the cost of {m2} notebooks at the same rate?",
        key=key,
        distractors=_unique_distractors(key, [f"Rs {m2 * u + u}", f"Rs {m2 * u - u}", f"Rs {c}", f"Rs {(m2 + 1) * u if (m2 + 1) * u != m2 * u + u else m2 * u + 2 * u}", f"Rs {m2 * c // n + 10}", f"Rs {m2 * u * 2}", f"Rs {u}"]),
        explanation=f"One notebook costs {c}/{n} = Rs {u}. So {m2} notebooks cost {m2} x {u} = {key}.",
        difficulty="easy", skill_tags=["unitary", "rates"], params={"n": n, "u": u, "m": m2},
    )


def v_speed_convert(text):
    m = re.search(r"Convert (\d+) km/h to m/s", text)
    k = int(m.group(1))
    assert (k * 5) % 18 == 0
    return f"{k * 5 // 18} m/s"


def gen_speed_convert(rng, chapter):
    k = rng.choice([18, 36, 54, 72, 90, 108, 126, 144, 198, 216])
    key = f"{k * 5 // 18} m/s"
    return AuthoredItem(
        family="speed_convert", chapter=chapter, topic_slug="speed-distance",
        question_text=f"Convert {k} km/h to m/s.",
        key=key,
        distractors=_unique_distractors(key, [f"{k * 5 // 18 + 5} m/s", f"{k * 5 // 18 - 5 if k * 5 // 18 > 5 else k * 5 // 18 + 10} m/s", f"{k // 10} m/s", f"{k * 2 // 9} m/s", f"{k * 5 // 18 * 2} m/s", f"{k * 18 // 5} m/s", f"{k * 5 // 18 + 1} m/s"]),
        explanation=f"km/h to m/s: multiply by 5/18. {k} x 5/18 = {key}.",
        difficulty="medium", skill_tags=["speed", "conversion"], params={"k": k},
    )


def v_percent_in_prop(text):
    m = re.search(r"group of (\d+) students, (\d+)%", text)
    t, p = int(m.group(1)), int(m.group(2))
    assert (t * p) % 100 == 0
    return str(t * p // 100)


def gen_percent_in_prop(rng, chapter):
    while True:
        t = rng.choice([20, 25, 40, 50, 60, 80, 120, 200, 240])
        p = rng.choice([10, 20, 25, 30, 40, 50, 60, 75, 80])
        if (t * p) % 100 == 0:
            break
    key = str(t * p // 100)
    return AuthoredItem(
        family="percent_in_prop", chapter=chapter, topic_slug="percent-in-proportion",
        question_text=f"In a group of {t} students, {p}% walk to school. How many students walk to school?",
        key=key,
        distractors=_unique_distractors(key, [str(t * p // 100 + 2), str(t * p // 100 - 2 if t * p // 100 > 2 else t * p // 100 + 4), str(t - t * p // 100), str(t * p // 100 * 2), str(p), str(t * p // 100 + 5), str(t * p // 100 // 2 if t * p // 100 > 10 else t * p // 100 + 1)]),
        explanation=f"{p}% of {t} = ({p}/100) x {t} = {key}.",
        difficulty="easy", skill_tags=["percentage", "proportion"], params={"t": t, "p": p},
    )


def v_fuel_rate(text):
    m = re.search(r"(\d+) litres of petrol per 100 km.*for a (\d+) km trip", text)
    r, d = int(m.group(1)), int(m.group(2))
    assert (r * d) % 100 == 0
    return f"{r * d // 100} litres"


def gen_fuel_rate(rng, chapter):
    while True:
        r = rng.choice([4, 5, 6, 8, 10, 12])
        d = rng.choice([150, 200, 250, 300, 350, 400, 450, 500, 750])
        if (r * d) % 100 == 0:
            break
    key = f"{r * d // 100} litres"
    return AuthoredItem(
        family="fuel_rate", chapter=chapter, topic_slug="compound-problems",
        question_text=f"A car uses {r} litres of petrol per 100 km. How much petrol does it need for a {d} km trip?",
        key=key,
        distractors=_unique_distractors(key, [f"{r * d // 100 + 2} litres", f"{r * d // 100 - 2 if r * d // 100 > 2 else r * d // 100 + 4} litres", f"{r * d // 100 * 2} litres", f"{r} litres", f"{r * d // 100 + 5} litres", f"{d // 100} litres", f"{r * d // 100 + 1} litres"]),
        explanation=f"Petrol needed = ({r}/100) x {d} = {key}.",
        difficulty="medium", skill_tags=["rates", "multi-step"], params={"r": r, "d": d},
    )


_PROP_SCENARIOS = [
    ("More workers are hired to build a wall, and the wall gets finished sooner.", "Inverse proportion"),
    ("A car drives faster, and the journey takes less time.", "Inverse proportion"),
    ("You buy more notebooks, and the total cost goes up.", "Direct proportion"),
    ("More taps are opened to fill a tank, and it fills faster.", "Inverse proportion"),
    ("You use more flour, and you can bake more loaves of bread.", "Direct proportion"),
    ("A scooter uses more petrol, and it travels more kilometres.", "Direct proportion"),
    ("Fewer people share a pizza, and each person gets a bigger slice.", "Inverse proportion"),
    ("A printer prints more pages, and it uses more ink.", "Direct proportion"),
    ("You walk faster, and you reach school in less time.", "Inverse proportion"),
    ("More hours of work bring in more wages at a fixed hourly rate.", "Direct proportion"),
]


def gen_prop_classify(rng, chapter):
    scenario, kind = rng.choice(_PROP_SCENARIOS)
    other = "Direct proportion" if kind == "Inverse proportion" else "Inverse proportion"
    return AuthoredItem(
        family="prop_classify", chapter=chapter, topic_slug="direct-inverse-mix",
        question_text=f"{scenario} What kind of proportion is this?",
        key=kind,
        distractors=[other, "Both direct and inverse", "No proportion at all"],
        explanation=f"As one quantity increases, the other {'decreases' if kind == 'Inverse proportion' else 'increases'} - that is {kind.lower()}.",
        difficulty="easy", skill_tags=["direct-inverse"], params={"kind": kind},
    )


def _verify_prop_classify(item):
    expected = dict((s, k) for s, k in _PROP_SCENARIOS)
    scen = item.question_text.split(" What kind of proportion")[0]
    return expected.get(scen) == item.key and item.key not in item.distractors


VERIFIERS.update({
    "percent_of": v_percent_of,
    "frac_to_percent": v_frac_to_percent,
    "decimal_to_fraction": v_decimal_to_fraction,
    "percent_increase": v_percent_change,
    "percent_decrease": v_percent_change,
    "profit_percent": v_profit_percent,
    "loss_percent": v_profit_percent,
    "simple_interest": v_simple_interest,
    "gst_price": v_gst_price,
    "inverse_prop": v_inverse_prop,
    "speed_time": v_speed_time,
    "time_for_distance": v_time_for_distance,
    "work_time": v_work_time,
    "unitary_cost": v_unitary,
    "speed_convert": v_speed_convert,
    "percent_in_prop": v_percent_in_prop,
    "fuel_rate": v_fuel_rate,
})

SET_VERIFIERS["prop_classify"] = _verify_prop_classify

GENERATORS.update({
    8: [
        (gen_percent_of, v_percent_of, 6),
        (gen_frac_to_percent, v_frac_to_percent, 5),
        (gen_decimal_to_fraction, v_decimal_to_fraction, 5),
        (gen_percent_increase, v_percent_change, 5),
        (gen_percent_decrease, v_percent_change, 4),
        (gen_profit_percent, v_profit_percent, 5),
        (gen_loss_percent, v_profit_percent, 4),
        (gen_simple_interest, v_simple_interest, 4),
        (gen_gst_price, v_gst_price, 4),
    ],
    10: [
        (gen_inverse_prop, v_inverse_prop, 6),
        (gen_speed_time, v_speed_time, 5),
        (gen_time_for_distance, v_time_for_distance, 5),
        (gen_work_time, v_work_time, 5),
        (gen_unitary, v_unitary, 5),
        (gen_speed_convert, v_speed_convert, 4),
        (gen_percent_in_prop, v_percent_in_prop, 5),
        (gen_fuel_rate, v_fuel_rate, 5),
        (gen_prop_classify, None, 5),
    ],
})


# ---------------------------------------------------------------------------
# True/false and numerical families (Oct 21 user approval: type mix for maths).
# Every key is re-derived from the rendered text by an independent verifier.
# ---------------------------------------------------------------------------

def _tf(item_family, chapter, topic_slug, text, truth, explanation, difficulty, tags, params):
    return AuthoredItem(
        family=item_family, chapter=chapter, topic_slug=topic_slug,
        question_text=text,
        key="True" if truth else "False",
        distractors=[],
        explanation=explanation,
        difficulty=difficulty, skill_tags=tags, params=params, qtype="true_false",
    )


def v_tf_perfect_square(text):
    m = re.search(r"(\d+) is a perfect square", text)
    n = int(m.group(1))
    r = _math.isqrt(n)
    return "True" if r * r == n else "False"


def gen_tf_perfect_square(rng, chapter):
    if rng.random() < 0.5:
        r = rng.randint(4, 30)
        n = r * r
    else:
        while True:
            n = rng.randint(10, 900)
            r = _math.isqrt(n)
            if r * r != n:
                break
    truth = _math.isqrt(n) ** 2 == n
    return _tf("tf_perfect_square", chapter, "perfect-squares",
               f"State whether the following statement is true or false: {n} is a perfect square.",
               truth,
               f"{_math.isqrt(n)}^2 = {_math.isqrt(n) ** 2}, so {n} {'is' if truth else 'is not'} a perfect square.",
               "easy", ["squares", "true-false"], {"n": n})


def v_num_square_root(text):
    m = re.search(r"square plot is (\d+) m\^?2", text)
    area = int(m.group(1))
    r = _math.isqrt(area)
    assert r * r == area
    return str(r)


def gen_num_square_root(rng, chapter):
    r = rng.randint(6, 40)
    area = r * r
    return AuthoredItem(
        family="num_square_root", chapter=chapter, topic_slug="square-roots",
        question_text=f"The area of a square plot is {area} m^2. What is the length of one side, in metres?",
        key=str(r), distractors=[],
        explanation=f"Side = square root of {area} = {r} m, because {r}^2 = {area}.",
        difficulty="easy", skill_tags=["square-roots", "numerical"], params={"area": area, "unit": "m"},
        qtype="numerical",
    )


def v_tf_exponent_eq(text):
    m = re.search(r"(\d+)\^(\d+) x \1\^(\d+) = (\d+)\^(\d+)", text)
    a, m1, m2, b, n = (int(g) for g in m.groups())
    return "True" if (a ** m1) * (a ** m2) == b ** n else "False"


def gen_tf_exponent_eq(rng, chapter):
    a = rng.choice([2, 3, 5])
    m1, m2 = rng.randint(2, 4), rng.randint(2, 4)
    if rng.random() < 0.5:
        b, n = a, m1 + m2
    else:
        b, n = rng.choice([a * 2, a, a]), rng.choice([m1 + m2, m1 * m2, m1 + m2 + 1])
        if b == a and n == m1 + m2:
            n += 1
    truth = (a ** m1) * (a ** m2) == b ** n
    return _tf("tf_exponent_eq", chapter, "laws-of-exponents",
               f"True or false: {a}^{m1} x {a}^{m2} = {b}^{n}.",
               truth,
               f"{a}^{m1} x {a}^{m2} = {a ** m1 * a ** m2}, and {b}^{n} = {b ** n}. These are {'equal' if truth else 'not equal'}.",
               "medium", ["exponents", "true-false"], {"a": a, "m1": m1, "m2": m2, "b": b, "n": n})


def v_num_exponent(text):
    m = re.search(r"value of (\d+)\^(\d+)\?", text)
    return str(int(m.group(1)) ** int(m.group(2)))


def gen_num_exponent(rng, chapter):
    a = rng.choice([2, 3, 4, 5])
    n = rng.randint(3, 6 if a == 2 else 4)
    return AuthoredItem(
        family="num_exponent", chapter=chapter, topic_slug="exponents",
        question_text=f"What is the value of {a}^{n}?",
        key=str(a ** n), distractors=[],
        explanation=f"{a}^{n} = {a} multiplied by itself {n} times = {a ** n}.",
        difficulty="easy", skill_tags=["exponents", "numerical"], params={"a": a, "n": n},
        qtype="numerical",
    )


def v_tf_ratio(text):
    m = re.search(r"ratio (\d+):(\d+) is (?:the same as|equivalent to) (\d+):(\d+)", text)
    a, b, c, d = (int(g) for g in m.groups())
    return "True" if a * d == b * c else "False"


def gen_tf_ratio(rng, chapter):
    a = rng.randint(2, 9)
    b = rng.randint(2, 9)
    g = _math.gcd(a, b)
    sa, sb = a // g, b // g
    if rng.random() < 0.5:
        k = rng.choice([2, 3, 4, 5])
        c, d = sa * k, sb * k
    else:
        c, d = sa + rng.randint(1, 3), sb * rng.choice([2, 3])
        if c * sb == d * sa:
            c += 1
    truth = a * d == b * c
    return _tf("tf_ratio", chapter, "ratio",
               f"True or false: the ratio {a}:{b} is equivalent to {c}:{d}.",
               truth,
               f"{a}:{b} simplifies to {sa}:{sb}. {c}:{d} {'also gives' if truth else 'does not give'} {sa}:{sb}.",
               "medium", ["ratios", "true-false"], {"a": a, "b": b, "c": c, "d": d})


def v_num_unitary(text):
    m = re.search(r"(\d+) kg of rice costs Rs (\d+)\.*.*cost of (\d+) kg", text)
    n, c, m2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
    assert c % n == 0
    return str(c // n * m2)


def gen_num_unitary(rng, chapter):
    n = rng.choice([2, 4, 5, 8, 10])
    u = rng.choice([12, 15, 20, 25, 30, 40, 45, 60])
    c = n * u
    while True:
        m2 = rng.randint(2, 12)
        if m2 != n:
            break
    return AuthoredItem(
        family="num_unitary", chapter=chapter, topic_slug="unitary-method",
        question_text=f"{n} kg of rice costs Rs {c}. What is the cost of {m2} kg of rice at the same rate, in rupees?",
        key=str(m2 * u), distractors=[],
        explanation=f"1 kg costs {c}/{n} = Rs {u}. {m2} kg costs {m2} x {u} = Rs {m2 * u}.",
        difficulty="easy", skill_tags=["unitary", "numerical"], params={"n": n, "u": u, "m": m2, "unit": "Rs"},
        qtype="numerical",
    )


def v_tf_pyth_triple(text):
    m = re.search(r"sides (\d+) cm, (\d+) cm, and (\d+) cm is right-angled", text)
    a, b, c = (int(g) for g in m.groups())
    x, y, z = sorted([a, b, c])
    return "True" if x * x + y * y == z * z else "False"


def gen_tf_pyth_triple(rng, chapter):
    if rng.random() < 0.5:
        a, b, c = rng.choice([(3, 4, 5), (6, 8, 10), (5, 12, 13), (9, 12, 15), (8, 15, 17), (7, 24, 25), (12, 16, 20)])
    else:
        while True:
            a, b, c = sorted(rng.randint(3, 26) for _ in range(3))
            if a * a + b * b != c * c:
                break
    truth = a * a + b * b == c * c
    return _tf("tf_pyth_triple", chapter, "converse",
               f"True or false: a triangle with sides {a} cm, {b} cm, and {c} cm is right-angled.",
               truth,
               f"{a}^2 + {b}^2 = {a * a + b * b} and {c}^2 = {c * c}. They are {'equal' if truth else 'not equal'}, so the triangle {'is' if truth else 'is not'} right-angled.",
               "medium", ["pythagoras", "true-false"], {"a": a, "b": b, "c": c})


def v_num_pyth_hyp(text):
    m = re.search(r"triangle are (\d+) cm and (\d+) cm", text)
    a, b = int(m.group(1)), int(m.group(2))
    h = _math.isqrt(a * a + b * b)
    assert h * h == a * a + b * b
    return str(h)


def gen_num_pyth_hyp(rng, chapter):
    a, b, c = rng.choice([(3, 4, 5), (6, 8, 10), (5, 12, 13), (9, 12, 15), (8, 15, 17), (7, 24, 25), (12, 16, 20), (10, 24, 26), (20, 21, 29)])
    return AuthoredItem(
        family="num_pyth_hyp", chapter=chapter, topic_slug="finding-sides",
        question_text=f"The two legs of a right-angled triangle are {a} cm and {b} cm. What is the length of the hypotenuse, in cm?",
        key=str(c), distractors=[],
        explanation=f"Hypotenuse^2 = {a}^2 + {b}^2 = {a * a + b * b} = {c}^2, so the hypotenuse is {c} cm.",
        difficulty="medium", skill_tags=["pythagoras", "numerical"], params={"a": a, "b": b, "unit": "cm"},
        qtype="numerical",
    )


def v_tf_identity(text):
    m = re.search(r"\(a \+ b\)\^2 = a\^2 \+ b\^2 for all", text)
    if m:
        return "False"  # missing the 2ab term
    m = re.search(r"\(a \+ b\)\^2 = a\^2 \+ 2ab \+ b\^2 for all", text)
    if m:
        return "True"
    m = re.search(r"\(a - b\)\^2 = a\^2 - 2ab \+ b\^2 for all", text)
    if m:
        return "True"
    m = re.search(r"\(a - b\)\^2 = a\^2 - b\^2 for all", text)
    if m:
        return "False"
    m = re.search(r"\(a \+ b\)\(a - b\) = a\^2 - b\^2 for all", text)
    if m:
        return "True"
    m = re.search(r"\(a \+ b\)\(a - b\) = a\^2 \+ b\^2 for all", text)
    if m:
        return "False"
    raise ValueError("unknown identity statement")


_IDENTITY_STATEMENTS = [
    ("(a + b)^2 = a^2 + 2ab + b^2 for all numbers a and b.", True, "The expansion of (a + b)^2 includes the middle term 2ab."),
    ("(a + b)^2 = a^2 + b^2 for all numbers a and b.", False, "The middle term 2ab is missing; for example a = 1, b = 1 gives 4 on the left but 2 on the right."),
    ("(a - b)^2 = a^2 - 2ab + b^2 for all numbers a and b.", True, "This is the standard expansion of (a - b)^2."),
    ("(a - b)^2 = a^2 - b^2 for all numbers a and b.", False, "The middle term -2ab is missing; for example a = 2, b = 1 gives 1 on the left but 3 on the right."),
    ("(a + b)(a - b) = a^2 - b^2 for all numbers a and b.", True, "This is the difference-of-squares identity."),
    ("(a + b)(a - b) = a^2 + b^2 for all numbers a and b.", False, "The cross terms cancel, leaving a^2 - b^2, not a^2 + b^2."),
]


def gen_tf_identity(rng, chapter):
    stmt, truth, why = rng.choice(_IDENTITY_STATEMENTS)
    return _tf("tf_identity", chapter, "identities",
               f"True or false: {stmt}",
               truth, why, "medium", ["identities", "true-false"], {"stmt": stmt})


def v_num_algebra_eval(text):
    m = re.search(r"If x = (-?\d+), what is the value of (\d+)x\^2 \+ (\d+)\?", text)
    if m:
        x, a, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return str(a * x * x + b)
    m = re.search(r"If x = (-?\d+), what is the value of (\d+)x \+ (\d+)\?", text)
    x, a, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return str(a * x + b)


def gen_num_algebra_eval(rng, chapter):
    x = rng.randint(2, 9)
    if rng.random() < 0.5:
        a = rng.randint(2, 5)
        b = rng.randint(1, 20)
        return AuthoredItem(
            family="num_algebra_eval", chapter=chapter, topic_slug="expressions",
            question_text=f"If x = {x}, what is the value of {a}x^2 + {b}?",
            key=str(a * x * x + b), distractors=[],
            explanation=f"{a} x {x}^2 + {b} = {a} x {x * x} + {b} = {a * x * x} + {b} = {a * x * x + b}.",
            difficulty="medium", skill_tags=["algebra", "numerical"], params={"x": x, "a": a, "b": b},
            qtype="numerical",
        )
    a = rng.randint(2, 12)
    b = rng.randint(1, 30)
    return AuthoredItem(
        family="num_algebra_eval", chapter=chapter, topic_slug="expressions",
        question_text=f"If x = {x}, what is the value of {a}x + {b}?",
        key=str(a * x + b), distractors=[],
        explanation=f"{a} x {x} + {b} = {a * x} + {b} = {a * x + b}.",
        difficulty="easy", skill_tags=["algebra", "numerical"], params={"x": x, "a": a, "b": b},
        qtype="numerical",
    )


def v_tf_area(text):
    m = re.search(r"base (\d+) cm and height (\d+) cm has an area of (\d+) cm\^?2", text)
    b, h, claimed = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if "triangle" in text:
        actual = b * h // 2 if (b * h) % 2 == 0 else -1
    else:
        actual = b * h
    return "True" if actual == claimed else "False"


def gen_tf_area(rng, chapter):
    b = rng.choice([4, 6, 8, 10, 12])
    h = rng.choice([3, 4, 5, 6, 7, 8])
    shape = rng.choice(["parallelogram", "triangle"])
    if (b * h) % 2 and shape == "triangle":
        h += 1
    actual = b * h // 2 if shape == "triangle" else b * h
    if rng.random() < 0.5:
        claimed = actual
    else:
        claimed = rng.choice([actual + b, actual - b if actual > b else actual + 2, b * h, actual * 2])
        if claimed == actual:
            claimed += 2
    truth = claimed == actual
    return _tf("tf_area", chapter, "area-parallelogram" if shape == "parallelogram" else "area-triangle",
               f"True or false: a {shape} with base {b} cm and height {h} cm has an area of {claimed} cm^2.",
               truth,
               f"Area of the {shape} = {'(1/2) x ' if shape == 'triangle' else ''}{b} x {h} = {actual} cm^2, so the claim is {'correct' if truth else 'wrong'}.",
               "medium", ["area", "true-false"], {"b": b, "h": h, "claimed": claimed})


def v_num_area_triangle(text):
    m = re.search(r"base (\d+) cm and height (\d+) cm", text)
    b, h = int(m.group(1)), int(m.group(2))
    assert (b * h) % 2 == 0
    return str(b * h // 2)


def gen_num_area_triangle(rng, chapter):
    b = rng.choice([4, 6, 8, 10, 12, 14])
    h = rng.choice([3, 5, 6, 7, 8, 9])
    if (b * h) % 2:
        b += 2
    return AuthoredItem(
        family="num_area_triangle", chapter=chapter, topic_slug="area-triangle",
        question_text=f"What is the area of a triangle with base {b} cm and height {h} cm, in cm^2?",
        key=str(b * h // 2), distractors=[],
        explanation=f"Area = (1/2) x base x height = (1/2) x {b} x {h} = {b * h // 2} cm^2.",
        difficulty="easy", skill_tags=["area", "numerical"], params={"b": b, "h": h, "unit": "cm^2"},
        qtype="numerical",
    )


VERIFIERS.update({
    "tf_perfect_square": v_tf_perfect_square,
    "num_square_root": v_num_square_root,
    "tf_exponent_eq": v_tf_exponent_eq,
    "num_exponent": v_num_exponent,
    "tf_ratio": v_tf_ratio,
    "num_unitary": v_num_unitary,
    "tf_pyth_triple": v_tf_pyth_triple,
    "num_pyth_hyp": v_num_pyth_hyp,
    "tf_identity": v_tf_identity,
    "num_algebra_eval": v_num_algebra_eval,
    "tf_area": v_tf_area,
    "num_area_triangle": v_num_area_triangle,
})

GENERATORS[1].extend([
    (gen_tf_perfect_square, v_tf_perfect_square, 5),
    (gen_num_square_root, v_num_square_root, 4),
])
GENERATORS[2].extend([
    (gen_tf_exponent_eq, v_tf_exponent_eq, 5),
    (gen_num_exponent, v_num_exponent, 4),
])
GENERATORS[7].extend([
    (gen_tf_ratio, v_tf_ratio, 5),
    (gen_num_unitary, v_num_unitary, 4),
])
GENERATORS[9].extend([
    (gen_tf_pyth_triple, v_tf_pyth_triple, 5),
    (gen_num_pyth_hyp, v_num_pyth_hyp, 4),
])
GENERATORS[13].extend([
    (gen_tf_identity, v_tf_identity, 5),
    (gen_num_algebra_eval, v_num_algebra_eval, 4),
])
GENERATORS[14].extend([
    (gen_tf_area, v_tf_area, 5),
    (gen_num_area_triangle, v_num_area_triangle, 4),
])


if __name__ == "__main__":
    raise SystemExit(main())
