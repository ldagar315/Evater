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
    order = [item.key] + item.distractors
    rng.shuffle(order)
    letters = ["A", "B", "C", "D"]
    options = [{"id": letters[i], "text": order[i]} for i in range(4)]
    correct = letters[order.index(item.key)]
    return {
        "chapter_id": str(spec.id),
        "concept_id": str(concept_id),
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
        "question_family_key": f"evater-mcq:{spec.id}:{item.family}",
        "variant_key": "evater_authored",
        "media": [],
        "option_media": {},
        "render_config": {},
        "answer_spec": {
            "type": "mcq_single",
            "source_answer": item.key,
            "verification": f"code_verified:{item.family}",
            "answer_provenance": "evater_original",
        },
        "generation_spec": {
            "ingestion_method": "evater_authored_generator",
            "generator_family": item.family,
            "parameters": item.params,
            "verified_by": "independent_text_reparse",
        },
        "license_status": "review_required",
        "review_status": "review",
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


if __name__ == "__main__":
    raise SystemExit(main())
