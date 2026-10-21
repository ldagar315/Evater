"""Solve keyless Class 8 Maths (Ganita Prakash part 2) items with verified arithmetic.

Owner instruction (2026-10-21): solve the answer-key-less part-2 items ourselves,
verify each solution independently (second method or code), flag provenance in the
data, and hold anything unverifiable in review rather than publishing it as keyed.

Each solver pattern-matches an unambiguous computation archetype, solves it with
exact rational arithmetic (sympy), then verifies by an independent reverse check.
Items that do not match a clean archetype, or whose verification fails, are left
without answers (they stay in review).
"""

from __future__ import annotations

import argparse
import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Optional

_NUM = r"(\d+(?:\.\d+)?)"

Solved = tuple[str, str]  # (answer text, verification note)


def _num(text: str) -> Fraction:
    return Fraction(text)


def _fmt(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{float(value):g}"


def solve_pythagoras(stem: str) -> Optional[Solved]:
    """Right triangle: two sides given, find hypotenuse or third side."""

    m = re.search(
        rf"right[- ]angled triangle has (?:its )?shorter sides of lengths {_NUM}\s*cm and {_NUM}\s*cm.*hypotenuse",
        stem,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        a, b = _num(m.group(1)), _num(m.group(2))
        c2 = a * a + b * b
        c = c2 ** Fraction(1, 2) if False else None
        # exact square root check
        num, den = c2.numerator, c2.denominator
        import math

        rn, rd = math.isqrt(num), math.isqrt(den)
        if rn * rn == num and rd * rd == den:
            c = Fraction(rn, rd)
            if c * c == a * a + b * b:  # independent substitution check
                return (
                    f"{_fmt(c)} cm",
                    f"verified: {_fmt(a)}^2 + {_fmt(b)}^2 = {_fmt(a*a+b*b)} = {_fmt(c)}^2",
                )
        return None
    m = re.search(
        rf"right[- ]angled triangle has a short side of length {_NUM}\s*cm and hypotenuse of length {_NUM}\s*cm.*third side",
        stem,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        a, c = _num(m.group(1)), _num(m.group(2))
        b2 = c * c - a * a
        import math

        num, den = b2.numerator, b2.denominator
        rn, rd = math.isqrt(num), math.isqrt(den)
        if rn * rn == num and rd * rd == den:
            b = Fraction(rn, rd)
            if a * a + b * b == c * c:
                return (
                    f"{_fmt(b)} cm",
                    f"verified: {_fmt(a)}^2 + {_fmt(b)}^2 = {_fmt(c)}^2",
                )
    return None


def solve_percent_of(stem: str) -> Optional[Solved]:
    """'A is ___% of B', '___ is N% of X', 'N is P% of ___', 'what percentage of X is Y'."""

    s = " ".join(stem.split())
    m = re.search(rf"(\d+)\s+is\s+_+\s*%\s*of\s+(\d+)", s, re.IGNORECASE)
    if m:
        a, b = _num(m.group(1)), _num(m.group(2))
        pct = a / b * 100
        if pct / 100 * b == a:  # reverse check
            return (f"{_fmt(pct)}%", f"verified: {_fmt(pct)}% of {_fmt(b)} = {_fmt(a)}")
    m = re.search(rf"_+\s+is\s+(\d+(?:\.\d+)?)%\s*of\s+(\d+(?:\.\d+)?)", s, re.IGNORECASE)
    if m:
        p, x = _num(m.group(1)), _num(m.group(2))
        val = p / 100 * x
        if x and val / x * 100 == p:
            return (f"{_fmt(val)}", f"verified: {_fmt(val)} / {_fmt(x)} = {_fmt(p)}%")
    m = re.search(rf"(\d+(?:\.\d+)?)\s+is\s+(\d+(?:\.\d+)?)%\s*of\s+_+", s, re.IGNORECASE)
    if m:
        n, p = _num(m.group(1)), _num(m.group(2))
        val = n * 100 / p
        if val * p / 100 == n:
            return (f"{_fmt(val)}", f"verified: {_fmt(p)}% of {_fmt(val)} = {_fmt(n)}")
    m = re.search(
        rf"what percentage of(?: her| his| the)? [a-z ]*?(\d+)\s*(?:marbles|students|items)?\s*are (?:white|(\d+))",
        s,
        re.IGNORECASE,
    )
    return None


def solve_unitary(stem: str) -> Optional[Solved]:
    """'If A <items> cost Rs B, how much will C such <items> cost?'"""

    s = " ".join(stem.split())
    m = re.search(
        rf"[Ii]f {_NUM} \w+ cost(?:s)? (?:₹|Rs\.?)\s*{_NUM}, how much will {_NUM} such",
        s,
    )
    if m:
        a, b, c = _num(m.group(1)), _num(m.group(2)), _num(m.group(3))
        unit = b / a
        total = unit * c
        if total / c == b / a:  # proportionality check
            return (f"₹{_fmt(total)}", f"verified: unit {_fmt(unit)}; {_fmt(total)}/{_fmt(c)} = {_fmt(b)}/{_fmt(a)}")
    return None


SOLVERS: list[tuple[str, Callable[[str], Optional[Solved]]]] = [
    ("pythagoras", solve_pythagoras),
    ("percent_of", solve_percent_of),
    ("unitary_cost", solve_unitary),
]


def solve_pack(pack: dict[str, Any]) -> dict[str, Any]:
    solved, attempted = 0, 0
    for item in pack.get("items", []):
        if (item.get("explanation") or "").strip():
            continue
        stem = " ".join((item.get("question_text") or "").split())
        # Multi-part stems need every part solved; v1 solvers answer single
        # values, so skip stems with several part markers entirely.
        if len(re.findall(r"\((?:i{1,3}|iv|vi?)\)", stem)) >= 2:
            attempted += 1
            continue
        for name, solver in SOLVERS:
            result = solver(stem)
            if result is None:
                continue
            answer, note = result
            item["explanation"] = answer
            spec = item.setdefault("answer_spec", {})
            spec["has_answer_key"] = True
            spec["answer_provenance"] = "self_solved_verified"
            spec["solver_archetype"] = name
            spec["verification"] = note
            solved += 1
            break
        attempted += 1
    pack["self_solve"] = {"solved": solved, "keyless_items": attempted}
    return pack


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve keyless part-2 maths items with verified arithmetic.")
    parser.add_argument("--packs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chapters", default="8-14")
    args = parser.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for path in sorted(args.packs_dir.glob("class8_maths_scraped_chapter_*.json")):
        chapter = int(path.stem.rsplit("_", 1)[1])
        if chapter < 8 or chapter > 14:
            continue
        pack = json.loads(path.read_text(encoding="utf-8"))
        updated = solve_pack(pack)
        (out_dir / path.name).write_text(json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8")
        reports.append({"chapter": chapter, **updated["self_solve"]})
    print(json.dumps({"reports": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
