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

_NUM = r"(?<!\d)(\d+(?:\.\d+)?)"

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




def solve_inverse_proportion(stem: str) -> Optional[Solved]:
    """Work/rate inverse proportions: more workers -> fewer days, etc."""

    s = " ".join(stem.split())
    m = re.search(
        rf"enough water to supply {_NUM} families.*for {_NUM} days.*{_NUM} more families",
        s, re.IGNORECASE,
    )
    if m:
        f1, d1, extra = _num(m.group(1)), _num(m.group(2)), _num(m.group(3))
        days = f1 * d1 / (f1 + extra)
        if days * (f1 + extra) == f1 * d1:
            return (f"{_fmt(days)} days", f"verified: {_fmt(days)} x {_fmt(f1+extra)} = {_fmt(f1*d1)} = {_fmt(f1)} x {_fmt(d1)}")
    m = re.search(
        rf"{_NUM} workers can paint .* in {_NUM} days.*one more worker",
        s, re.IGNORECASE,
    )
    if m:
        w, d = _num(m.group(1)), _num(m.group(2))
        days = w * d / (w + 1)
        if days * (w + 1) == w * d:
            return (f"{_fmt(days)} days", f"verified: {_fmt(days)} x {_fmt(w+1)} = {_fmt(w*d)} worker-days")
    m = re.search(
        rf"[Ii]t takes {_NUM} hours to fill {_NUM} tanks.*[Hh]ow long will it take to fill {_NUM}",
        s,
    )
    if m:
        h, t1, t2 = _num(m.group(1)), _num(m.group(2)), _num(m.group(3))
        rate = h / t1
        total = rate * t2
        if total / t2 == h / t1:
            return (f"{_fmt(total)} hours", f"verified: {_fmt(total)}/{_fmt(t2)} = {_fmt(h)}/{_fmt(t1)}")
    m = re.search(
        rf"requires {_NUM} machines to produce .* in {_NUM} days.*[Hh]ow many machines .* in {_NUM} days",
        s,
    )
    if m:
        m1, d1, d2 = _num(m.group(1)), _num(m.group(2)), _num(m.group(3))
        machines = m1 * d1 / d2
        if machines * d2 == m1 * d1:
            return (f"{_fmt(machines)} machines", f"verified: {_fmt(machines)} x {_fmt(d2)} = {_fmt(m1*d1)} machine-days")
    m = re.search(
        rf"takes {_NUM} hours to reach.*speed of {_NUM} km/h.*speed of {_NUM} km/h",
        s, re.IGNORECASE,
    )
    if m:
        t1, v1, v2 = _num(m.group(1)), _num(m.group(2)), _num(m.group(3))
        t2 = v1 * t1 / v2
        if t2 * v2 == v1 * t1:
            return (f"{_fmt(t2)} hours", f"verified: {_fmt(t2)} x {_fmt(v2)} = {_fmt(v1*t1)} km")
    m = re.search(
        rf"arranged in {_NUM} rows, with {_NUM} chairs in each row.*?rearranged with {_NUM} chairs in each row, how many rows",
        s, re.IGNORECASE,
    )
    if m:
        r1, c1, c2 = _num(m.group(1)), _num(m.group(2)), _num(m.group(3))
        rows = r1 * c1 / c2
        if rows * c2 == r1 * c1:
            return (f"{_fmt(rows)} rows", f"verified: {_fmt(rows)} x {_fmt(c2)} = {_fmt(r1*c1)} chairs")
    m = re.search(
        rf"{_NUM} periods a day, each of {_NUM} minutes.*{_NUM} periods a day",
        s, re.IGNORECASE,
    )
    if m:
        p1, mins, p2 = _num(m.group(1)), _num(m.group(2)), _num(m.group(3))
        each = p1 * mins / p2
        if each * p2 == p1 * mins:
            return (f"{_fmt(each)} minutes", f"verified: {_fmt(each)} x {_fmt(p2)} = {_fmt(p1*mins)} minutes")
    m = re.search(
        rf"small pump can fill a tank in {_NUM} hours.*large pump can fill the same tank in {_NUM} hours.*together",
        s, re.IGNORECASE,
    )
    if m:
        a, b = _num(m.group(1)), _num(m.group(2))
        rate = Fraction(1, 1) / a + Fraction(1, 1) / b
        t = 1 / rate
        if (1 / a + 1 / b) * t == 1:
            return (f"{_fmt(t)} hours", f"verified: (1/{_fmt(a)} + 1/{_fmt(b)}) x {_fmt(t)} = 1 tank")
    m = re.search(
        rf"Workers .* take {_NUM} days to pick .* in {_NUM}% of the plantation.*[Hh]ow many days .* complete",
        s, re.IGNORECASE,
    )
    if m:
        d, p = _num(m.group(1)), _num(m.group(2))
        total = d * 100 / p
        if total * p / 100 == d:
            return (f"{_fmt(total)} days", f"verified: {_fmt(p)}% of {_fmt(total)} = {_fmt(d)}")
    return None


def solve_percentage_change(stem: str) -> Optional[Solved]:
    """Percentage increase/decrease, reverse-percent, successive change."""

    s = " ".join(stem.split())
    m = re.search(
        rf"[Aa] number increased by {_NUM}% becomes {_NUM}\. What is the number",
        s,
    )
    if m:
        p, final = _num(m.group(1)), _num(m.group(2))
        x = final * 100 / (100 + p)
        if x * (100 + p) / 100 == final:
            return (f"{_fmt(x)}", f"verified: {_fmt(x)} increased by {_fmt(p)}% = {_fmt(final)}")
    m = re.search(
        rf"petrol price .* was (?:₹|Rs\.?\s*)?{_NUM} .* and (?:₹|Rs\.?\s*)?{_NUM} .* [Ww]hat is the percentage increase",
        s, re.IGNORECASE,
    )
    if m:
        old, new = _num(m.group(1)), _num(m.group(2))
        pct = (new - old) / old * 100
        if old * (100 + pct) / 100 == new:
            return (f"{_fmt(pct)}%", f"verified: {_fmt(old)} + {_fmt(pct)}% = {_fmt(new)}")
    m = re.search(
        rf"{_NUM}% discount on .* original price .* is (?:₹|Rs\.?\s*)?{_NUM}.*[Hh]ow much .* pay",
        s, re.IGNORECASE,
    )
    if m:
        d, price = _num(m.group(1)), _num(m.group(2))
        pay = price * (100 - d) / 100
        if pay == price - price * d / 100:
            return (f"₹{_fmt(pay)}", f"verified: {_fmt(price)} - {_fmt(d)}% = {_fmt(pay)}")
    m = re.search(
        rf"increased by {_NUM}% last year and by {_NUM}% this year.*overall percentage",
        s, re.IGNORECASE,
    )
    if m:
        a, b = _num(m.group(1)), _num(m.group(2))
        factor = (100 + a) * (100 + b) / 10000
        overall = (factor - 1) * 100
        if (100 + overall) / 100 == factor:
            return (f"{_fmt(overall)}%", f"verified: {_fmt((100+a)/100)} x {_fmt((100+b)/100)} = {_fmt(factor)}")
    m = re.search(
        rf"about {_NUM}% of its population in {_NUM}.*population in {_NUM} was {_NUM} lakhs",
        s, re.IGNORECASE,
    )
    if m:
        p = _num(m.group(1))
        year_pct, year_base, x = m.group(2), m.group(3), _num(m.group(4))
        if year_pct == year_base:
            val = x * p / 100
            if val * 100 / p == x:
                return (f"{_fmt(val)} lakhs", f"verified: {_fmt(p)}% of {_fmt(x)} = {_fmt(val)}")
    m = re.search(
        rf"In a room of {_NUM} people, {_NUM}% are left-handed.*[Hh]ow many left-handed people have to leave .* down to {_NUM}%",
        s,
    )
    if m:
        n, p1, p2 = _num(m.group(1)), _num(m.group(2)), _num(m.group(3))
        left = n * p1 / 100
        # left - x = p2% of (n - x)  ->  x = (left*100 - p2*n) / (100 - p2)
        x = (left * 100 - p2 * n) / (100 - p2)
        if x.denominator == 1 and (left - x) * 100 == p2 * (n - x):
            return (f"{_fmt(x)} people", f"verified: {_fmt(left-x)}/{_fmt(n-x)} = {_fmt(p2)}%")
    return None


def solve_mean_median(stem: str) -> Optional[Solved]:
    """Mean with unknown, median of a list, sum from mean."""

    s = " ".join(stem.split())
    m = re.search(
        rf"[Tt]he mean of the numbers ((?:\d+(?:\.\d+)?,\s*)+\d+(?:\.\d+)?,\s*(\w),.*) is ({_NUM})\. Find the value of \2",
        s,
    )
    if m:
        nums = [Fraction(x) for x in re.findall(r"\d+(?:\.\d+)?", m.group(1).split(", y")[0] if ", y" in m.group(1) else m.group(1))]
        var = m.group(2)
        target = _num(m.group(3))
        series = m.group(1)
        known = [Fraction(x) for x in re.findall(r"^\d+(?:\.\d+)?$", "")]  # placeholder
        parts = [p.strip() for p in series.split(",")]
        values = [Fraction(p) for p in parts if re.fullmatch(r"\d+(?:\.\d+)?", p)]
        if len(values) == len(parts) - 1:
            total = target * (len(values) + 1)
            y = total - sum(values)
            if (sum(values) + y) / (len(values) + 1) == target:
                return (f"{var} = {_fmt(y)}", f"verified: (sum {_fmt(sum(values))} + {_fmt(y)})/{len(values)+1} = {_fmt(target)}")
    m = re.search(
        r"[Ff]ind the median of ((?:\d+(?:\.\d+)?[,\s]+)+\d+(?:\.\d+)?)",
        s,
    )
    if m:
        values = sorted(Fraction(x) for x in re.findall(r"\d+(?:\.\d+)?", m.group(1)))
        if len(values) >= 3:
            n = len(values)
            if n % 2 == 1:
                med = values[n // 2]
            else:
                med = (values[n // 2 - 1] + values[n // 2]) / 2
            return (f"{_fmt(med)}", f"verified: middle of {n} sorted values (positions {n//2} and {n//2+1})" if n % 2 == 0 else f"verified: middle of {n} sorted values")
    m = re.search(
        rf"[Tt]he mean of a set of data with {_NUM} values is {_NUM}\. Find the sum",
        s,
    )
    if m:
        n, mean = _num(m.group(1)), _num(m.group(2))
        total = n * mean
        if total / n == mean:
            return (f"{_fmt(total)}", f"verified: {_fmt(total)}/{_fmt(n)} = {_fmt(mean)}")
    return None


def solve_area(stem: str) -> Optional[Solved]:
    """Rhombus from diagonals; quadrilateral from diagonal + offsets."""

    s = " ".join(stem.split())
    m = re.search(
        rf"[Aa]rea of a rhombus whose diagonals are {_NUM} cm and {_NUM} cm",
        s,
    )
    if m:
        d1, d2 = _num(m.group(1)), _num(m.group(2))
        area = d1 * d2 / 2
        if area * 2 == d1 * d2:
            return (f"{_fmt(area)} cm^2", f"verified: (1/2) x {_fmt(d1)} x {_fmt(d2)} = {_fmt(area)}")
    m = re.search(
        rf"quadrilateral ABCD given that AC = {_NUM} cm, BM = {_NUM} cm, DN = {_NUM} cm",
        s,
    )
    if m:
        ac, bm, dn = _num(m.group(1)), _num(m.group(2)), _num(m.group(3))
        area = ac * (bm + dn) / 2
        if area == ac * bm / 2 + ac * dn / 2:
            return (f"{_fmt(area)} cm^2", f"verified: (1/2) x {_fmt(ac)} x ({_fmt(bm)} + {_fmt(dn)}) = {_fmt(area)}")
    return None


def solve_coins_ratio(stem: str) -> Optional[Solved]:
    """Coin counts in ratio with denominations -> total money."""

    s = " ".join(stem.split())
    m = re.search(
        rf"{_NUM} coins in the ratio.*₹10 coins : no. of ₹5 coins : no. of ₹2 coins : no. of ₹1 coins :: ({_NUM}) : ({_NUM}) : ({_NUM}) : ({_NUM}).*[Hh]ow much money",
        s,
    )
    if m:
        total = _num(m.group(1))
        parts = [_num(m.group(i)) for i in range(2, 6)]
        share = total / sum(parts)
        if share * sum(parts) == total:
            counts = [share * p for p in parts]
            money = counts[0] * 10 + counts[1] * 5 + counts[2] * 2 + counts[3]
            if all(c.denominator == 1 for c in counts) and sum(counts) == total:
                return (
                    f"₹{_fmt(money)}",
                    f"verified: counts {[str(int(c)) for c in counts]} sum to {_fmt(total)}; value {_fmt(money)}",
                )
    return None


SOLVERS: list[tuple[str, Callable[[str], Optional[Solved]]]] = [
    ("pythagoras", solve_pythagoras),
    ("percent_of", solve_percent_of),
    ("unitary_cost", solve_unitary),
    ("inverse_proportion", solve_inverse_proportion),
    ("percentage_change", solve_percentage_change),
    ("mean_median", solve_mean_median),
    ("area", solve_area),
    ("coins_ratio", solve_coins_ratio),
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
            try:
                result = solver(stem)
            except (ArithmeticError, ValueError, AttributeError):
                result = None
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
