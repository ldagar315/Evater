"""Concept topic seeds for the Ganita Prakash Grade 8 Mathematics chapters.

Used by the maths extractor for deterministic concept assignment and by the
seed script to create the per-chapter concept rows. One-line learning outcomes
keep the manifest inspectable; keywords drive the keyword-scoring assignment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MathsTopicSeed:
    slug: str
    title: str
    definition: str
    keywords: tuple[str, ...]


def topic(slug: str, title: str, definition: str, *keywords: str) -> MathsTopicSeed:
    return MathsTopicSeed(slug=slug, title=title, definition=definition, keywords=tuple(keywords))


TOPICS: dict[int, tuple[MathsTopicSeed, ...]] = {
    1: (
        topic("perfect-squares", "Perfect squares", "Recognise and form perfect squares and their properties.", "square", "perfect square", "square number"),
        topic("square-roots", "Square roots", "Find square roots by inspection, patterns, and factorisation.", "square root", "root"),
        topic("perfect-cubes", "Perfect cubes", "Recognise and form perfect cubes and their properties.", "cube", "perfect cube"),
        topic("cube-roots", "Cube roots", "Find cube roots by estimation and prime factorisation.", "cube root"),
        topic("square-patterns", "Patterns in squares", "Use patterns between consecutive squares and odd-number sums.", "pattern", "consecutive", "odd numbers"),
        topic("digit-properties", "Unit digits of squares and cubes", "Reason about possible unit digits of squares and cubes.", "unit digit", "last digit", "units place"),
        topic("estimation", "Estimating roots", "Estimate square and cube roots of larger numbers.", "estimate", "guess", "nearest"),
        topic("area-side", "Squares, area, and side length", "Relate the area of a square to its side length and back.", "area", "side", "cloth", "tile"),
    ),
    2: (
        topic("exponents", "Exponents and powers", "Read and write numbers in exponential form.", "power", "exponent", "base", "raised"),
        topic("laws-of-exponents", "Laws of exponents", "Apply the product, quotient, and power laws of exponents.", "law", "multiply", "divide", "product of powers"),
        topic("zero-negative-exponents", "Zero and negative exponents", "Interpret zero and negative integer exponents.", "negative", "zero exponent", "reciprocal"),
        topic("scientific-notation", "Scientific notation", "Express large and small numbers in standard form.", "standard form", "scientific notation", "10^", "powers of 10"),
        topic("comparing-powers", "Comparing powers", "Compare and order numbers written as powers.", "compare", "greater", "larger", "smaller"),
        topic("large-numbers", "Large and small numbers", "Reason about the scale of very large and very small quantities.", "large number", "small number", "scale", "approximation"),
        topic("base-conversion", "Changing bases", "Rewrite powers with a common base to simplify expressions.", "same base", "common base", "simplify"),
        topic("power-patterns", "Patterns in powers", "Explore cyclicity and patterns in the digits of powers.", "pattern", "cyclic", "last digit"),
    ),
    3: (
        topic("place-value", "Place value", "Explain place value and the role of position in numerals.", "place value", "digit", "position"),
        topic("number-systems-history", "History of number systems", "Describe historical numeration systems across cultures.", "roman", "egyptian", "babylonian", "history", "ancient"),
        topic("hindu-arabic", "Hindu-Arabic numerals", "Describe the Hindu-Arabic system and the role of zero.", "hindu", "arabic", "zero", "indian"),
        topic("base-systems", "Number bases", "Represent numbers in bases other than ten.", "base", "binary", "base 2", "base 5"),
        topic("expanded-form", "Expanded form", "Write numerals in expanded form using powers of ten.", "expanded", "expand"),
        topic("comparing-numbers", "Comparing and ordering numbers", "Compare, order, and estimate whole numbers.", "compare", "order", "ascending", "descending"),
        topic("numeral-conversion", "Converting between systems", "Convert numerals between representations.", "convert", "representation"),
        topic("counting", "Counting principles", "Count systematically using place value structure.", "count", "how many numbers"),
    ),
    4: (
        topic("quadrilateral-properties", "Properties of quadrilaterals", "Identify sides, angles, and diagonals of quadrilaterals.", "quadrilateral", "four sides"),
        topic("angle-sum", "Angle sum of quadrilaterals", "Use the 360-degree angle sum of a quadrilateral.", "angle sum", "360"),
        topic("parallelogram", "Parallelograms", "State and apply properties of parallelograms.", "parallelogram", "opposite sides"),
        topic("rectangle-square", "Rectangles and squares", "Distinguish rectangles and squares by their properties.", "rectangle", "square"),
        topic("rhombus", "Rhombuses and kites", "Distinguish rhombuses and kites by their properties.", "rhombus", "kite"),
        topic("trapezium", "Trapeziums", "Identify trapeziums and their properties.", "trapezium"),
        topic("diagonals", "Diagonals of quadrilaterals", "Reason about diagonal properties of special quadrilaterals.", "diagonal", "bisect", "perpendicular"),
        topic("construction", "Constructing quadrilaterals", "Construct quadrilaterals from given measurements.", "construct", "construction", "draw"),
    ),
    5: (
        topic("factors-multiples", "Factors and multiples", "Find factors and multiples of numbers.", "factor", "multiple"),
        topic("divisibility", "Divisibility rules", "Apply divisibility tests for small numbers.", "divisible", "divisibility", "test"),
        topic("primes", "Prime and composite numbers", "Identify prime and composite numbers.", "prime", "composite"),
        topic("prime-factorisation", "Prime factorisation", "Express numbers as products of primes.", "prime factorisation", "factor tree"),
        topic("hcf-lcm", "HCF and LCM", "Compute highest common factors and lowest common multiples.", "hcf", "lcm", "highest common", "lowest common"),
        topic("number-patterns", "Number patterns and games", "Explore patterns, puzzles, and games with numbers.", "pattern", "game", "puzzle", "magic"),
        topic("digits-puzzles", "Digit puzzles", "Solve puzzles about digits of numbers.", "digit", "puzzle", "cryptarithm", "letters"),
        topic("remainders", "Remainders", "Reason with remainders in division.", "remainder", "divides"),
    ),
    6: (
        topic("distributive-property", "The distributive property", "Apply multiplication distributed over addition and subtraction.", "distributive", "distribute"),
        topic("identities", "Algebraic identities", "Use standard identities like (a+b)^2 to expand and factor.", "identity", "(a+b)", "a^2", "expand"),
        topic("expansion", "Expanding products", "Expand products of binomials.", "expand", "product", "binomial", "(x+"),
        topic("factorisation", "Factorising expressions", "Factorise algebraic expressions using common factors and identities.", "factorise", "factorization", "common factor"),
        topic("mental-math", "Mental multiplication", "Use distribution for quick mental calculation.", "mental", "quickly", "without pen", "98", "102"),
        topic("area-models", "Area models of multiplication", "Interpret products using area models.", "area model", "rectangle area"),
        topic("number-tricks", "Number tricks", "Explain number tricks using algebra.", "trick", "think of a number"),
        topic("simplification", "Simplifying expressions", "Simplify algebraic expressions using distributive rules.", "simplify", "combine like terms"),
    ),
    7: (
        topic("ratio", "Ratio", "Express and simplify ratios between quantities.", "ratio"),
        topic("proportion", "Proportion", "Identify and use proportional relationships.", "proportion", "proportional"),
        topic("unitary-method", "Unitary method", "Solve problems by finding the value of one unit.", "unitary", "unit", "per unit"),
        topic("direct-proportion", "Direct proportion", "Solve direct proportion problems.", "direct", "varies directly"),
        topic("scale-maps", "Scale and maps", "Use scale in maps and drawings.", "scale", "map"),
        topic("recipes-mixtures", "Recipes and mixtures", "Scale recipes and mixtures using ratio.", "recipe", "mixture", "ingredient"),
        topic("ratio-division", "Dividing in a given ratio", "Divide a quantity in a given ratio.", "divide", "share", "ratio"),
        topic("comparing-ratios", "Comparing ratios", "Compare ratios to decide which is greater or better.", "compare", "better deal", "which ratio"),
    ),
    8: (
        topic("percentage", "Percentage", "Convert fractions and decimals to percentages.", "percent", "percentage", "%"),
        topic("percent-conversion", "Converting between forms", "Convert among fractions, decimals, and percentages.", "convert", "decimal", "fraction"),
        topic("percent-of", "Percentage of a quantity", "Find a percentage of a quantity.", "percent of", "% of"),
        topic("increase-decrease", "Percentage increase and decrease", "Compute percentage increase and decrease.", "increase", "decrease", "change"),
        topic("profit-loss", "Profit and loss", "Solve profit, loss, and discount problems.", "profit", "loss", "discount", "selling price", "cost price"),
        topic("interest", "Simple interest", "Compute simple interest.", "interest", "principal", "rate", "time"),
        topic("tax", "Taxes and GST", "Compute taxes including GST.", "tax", "gst"),
        topic("fraction-decimal", "Fractions and decimals", "Convert between fractions and decimals.", "fraction", "decimal", "terminating", "recurring"),
    ),
    9: (
        topic("pythagoras-statement", "The Pythagoras theorem", "State the Pythagoras theorem for right triangles.", "pythagoras", "right triangle", "hypotenuse"),
        topic("baudhayana", "Baudhayana and history", "Describe the historical Baudhayana statement of the theorem.", "baudhayana", "sulba", "history"),
        topic("finding-sides", "Finding unknown sides", "Compute unknown sides of right triangles.", "find the", "hypotenuse", "leg", "side"),
        topic("pythagorean-triples", "Pythagorean triples", "Recognise and generate Pythagorean triples.", "triple", "3, 4, 5"),
        topic("converse", "Converse of the theorem", "Use the converse to test for right angles.", "converse", "right angled", "test"),
        topic("applications", "Applications of the theorem", "Apply the theorem to ladders, distances, and diagonals.", "ladder", "distance", "diagonal", "pole"),
        topic("proof", "Proofs of the theorem", "Explain a proof of the Pythagoras theorem.", "proof", "prove", "dissection"),
        topic("distance-plane", "Distance between points", "Find distances between points using the theorem.", "distance between", "points", "coordinate"),
    ),
    10: (
        topic("inverse-proportion", "Inverse proportion", "Identify and solve inverse proportion problems.", "inverse", "inversely"),
        topic("rates", "Rates", "Work with rates such as speed and work rates.", "rate", "speed", "per hour"),
        topic("work-time", "Work and time", "Solve work-and-time problems.", "work", "days", "workers", "complete"),
        topic("mixtures", "Mixtures and alligation", "Solve mixture problems using proportion.", "mixture", "mixed", "solution"),
        topic("percent-in-proportion", "Percentages in proportion", "Combine percentage reasoning with proportion.", "percent", "proportion"),
        topic("speed-distance", "Speed, distance, and time", "Relate speed, distance, and time.", "speed", "distance", "time", "km/h"),
        topic("direct-inverse-mix", "Direct vs inverse", "Distinguish direct from inverse variation in context.", "direct", "inverse", "varies"),
        topic("compound-problems", "Multi-step proportion problems", "Solve chained proportion problems.", "chain", "multi-step", "then"),
    ),
    11: (
        topic("constructions-basic", "Basic constructions", "Construct bisectors, perpendiculars, and angles.", "construct", "bisector", "perpendicular", "compass"),
        topic("transformations", "Transformations", "Identify reflection, rotation, and translation.", "reflection", "rotation", "translation", "transformation"),
        topic("symmetry", "Symmetry", "Identify lines of symmetry and rotational symmetry.", "symmetry", "symmetric", "line of symmetry"),
        topic("tessellation", "Tessellations", "Recognise and create tessellations.", "tessellation", "tile", "tiling"),
        topic("nets", "Nets of solids", "Identify nets of cubes and other solids.", "net", "cube", "solid", "fold"),
        topic("angles-construction", "Constructing angles", "Construct standard angles with compass and ruler.", "angle", "60", "90", "45"),
        topic("congruence", "Congruent figures", "Identify congruent figures and their properties.", "congruent", "congruence"),
        topic("spatial-reasoning", "Spatial reasoning", "Reason about views and arrangements of shapes.", "view", "top view", "arrangement"),
    ),
    12: (
        topic("plotting-points", "Plotting points", "Plot points on a coordinate grid.", "plot", "point", "coordinate", "grid"),
        topic("reading-graphs", "Reading graphs", "Read values and trends from graphs.", "graph", "read", "trend"),
        topic("line-graphs", "Line graphs", "Draw and interpret line graphs.", "line graph"),
        topic("bar-pie", "Bar graphs and pie charts", "Draw and interpret bar graphs and pie charts.", "bar graph", "pie chart", "histogram"),
        topic("coordinates", "Coordinates", "Use x and y coordinates to locate points.", "x-coordinate", "y-coordinate", "origin", "axis"),
        topic("data-stories", "Stories from data", "Interpret graphs as stories of change.", "story", "interpret", "data"),
        topic("scatter", "Scatter plots", "Plot and read simple scatter data.", "scatter"),
        topic("tables-to-graphs", "Tables to graphs", "Convert data tables into graphs.", "table", "data", "plot"),
    ),
    13: (
        topic("expressions", "Algebraic expressions", "Form and simplify algebraic expressions.", "expression", "term", "coefficient"),
        topic("simple-equations", "Simple equations", "Solve simple linear equations.", "equation", "solve", "linear"),
        topic("patterns-generalisation", "Patterns and generalisation", "Generalise patterns using algebra.", "pattern", "generalise", "nth"),
        topic("variables", "Variables and constants", "Use variables to represent unknowns.", "variable", "constant", "unknown"),
        topic("substitution", "Substitution", "Evaluate expressions by substituting values.", "substitute", "evaluate", "value of"),
        topic("forming-equations", "Forming equations", "Translate word problems into equations.", "word problem", "form an equation", "translate"),
        topic("like-terms", "Like and unlike terms", "Combine like terms.", "like terms", "combine"),
        topic("algebra-puzzles", "Algebra puzzles and games", "Solve puzzles using algebraic thinking.", "puzzle", "game", "magic"),
    ),
    14: (
        topic("area-rectangle", "Area of rectangles and squares", "Compute areas of rectangles and squares.", "rectangle", "square", "area"),
        topic("area-parallelogram", "Area of parallelograms", "Derive and use the parallelogram area formula.", "parallelogram", "base", "height"),
        topic("area-triangle", "Area of triangles", "Derive and use the triangle area formula.", "triangle", "half"),
        topic("area-trapezium", "Area of trapeziums", "Derive and use the trapezium area formula.", "trapezium"),
        topic("composite-shapes", "Composite shapes", "Find areas of composite figures by decomposition.", "composite", "combined", "shaded", "figure"),
        topic("units-conversion", "Units of area", "Convert between square units.", "cm^2", "m^2", "square units", "convert", "hectare"),
        topic("perimeter-area", "Perimeter vs area", "Distinguish and relate perimeter and area.", "perimeter", "boundary", "fence"),
        topic("area-in-context", "Area in context", "Apply area to real situations like flooring and fields.", "floor", "tile", "field", "carpet", "cost"),
    ),
}
