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
            item = gen(rng, chapter)
            dedupe_key = item.question_text + "|" + item.key
            if dedupe_key in seen_texts:
                continue
            # independent verification: re-parse the rendered text
            try:
                if item.family == "right_set":
                    ok = _verify_right_set(item)
                    assert ok
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


if __name__ == "__main__":
    raise SystemExit(main())
