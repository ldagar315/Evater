# Class 8 Maths: Number Play - Watch, Understand and Practise

Pick one topic. Use the video where shown, read the short explanation, then try the question before checking your answer. These seven topics are a starting point; use your textbook for the rest of the chapter.

## 1. Multiples and algebraic descriptions

### Think it through

A multiple of 7 can be written as 7a, where a is an integer. For any two integers a and b, 7a + 7b = 7(a + b). The bracket is still an integer, so the sum is a multiple of 7. For example, 14 + 35 = 49 follows this rule, but the letters explain why it works for every pair, including negative multiples.

Do not reverse the claim. A sum can be a multiple of 7 even if neither addend is: 2 + 5 = 7. When numbers are described as a multiple plus an extra amount, group the multiples and the extras separately. Check what the extras add to before deciding whether the whole sum is divisible by 7.

### Try it yourself

For integers p and q, must (7p + 3) + (7q + 4) be a multiple of 7? Show the factor of 7. Are the two addends themselves multiples of 7? Use p = 2 and q = 3 to check your reasoning with numbers.

**Check:** The sum is 7p + 7q + 7 = 7(p + q + 1), so it is always a multiple of 7. Neither addend is a multiple of 7: they leave remainders 3 and 4. With p = 2 and q = 3, the addends are 17 and 25 and the sum is 42. This also shows why a multiple sum does not force each addend to be a multiple.

## 2. Divisibility tests

### Watch: Divisibility Shortcuts: 1, 2, and 3

Professor Cunningham

[Open the video](https://www.youtube.com/watch?v=WC2puADmICM)

### Understand it

In decimal notation, 10 = 9 + 1, 100 = 99 + 1 and 1,000 = 999 + 1. Each place value is one more than a multiple of 9. For 738, write 7 × 100 + 3 × 10 + 8 = 7 × 99 + 3 × 9 + (7 + 3 + 8). The first two terms are multiples of 9. The remaining digit sum is 18, so 738 is divisible by 9. The same reasoning also works for divisibility by 3, since multiples of 9 are multiples of 3.

For divisibility by 4, use a different place-value fact: every complete hundred is divisible by 4, so only the last two digits matter. A number divisible by 9 need not be divisible by 4. When both conditions are requested, check both rather than treating one test as enough.

### Try it yourself

In the four-digit number 58d2, d is one decimal digit from 0 to 9. Find every possible value of d that makes the number divisible by both 9 and 4. Explain each test, then check by division.

**Check:** The digit sum is 5 + 8 + d + 2 = 15 + d. It ranges from 15 to 24, so the only possible multiple of 9 is 18, giving d = 3. The last two digits are then 32, divisible by 4. Thus the unique number is 5,832; 5,832 ÷ 9 = 648 and 5,832 ÷ 4 = 1,458. Both conditions hold.

## 3. Combined divisibility conditions

### Watch: Least common multiple

Khan Academy

[Open the video](https://www.youtube.com/watch?v=znmPfDfsir8)

### Understand it

To satisfy two divisibility conditions, a number must contain all the prime factors required by each divisor. For 6 = 2 × 3 and 8 = 2³, the least common multiple needs three factors of 2 and one of 3: 2³ × 3 = 24. It is the smallest positive number divisible by both. We do not need four factors of 2, because the three already meet the requirement for 6 as well as 8.

Use the highest required power of each prime, not the product of the two divisors by default. Multiplying the divisors gives a common multiple, but it can count shared factors twice and miss a smaller one. Any number divisible by both divisors is divisible by their least common multiple.

### Try it yourself

A number is divisible by both 12 and 18. Must it be divisible by 216? Find the smallest positive number satisfying the two given conditions and use prime factors to explain your answer.

**Check:** No. Since 12 = 2² × 3 and 18 = 2 × 3², their least common multiple is 2² × 3² = 36. The number 36 is divisible by both 12 and 18 but not by 216, so it disproves the claim. Every number satisfying both conditions is a multiple of 36, not necessarily a multiple of 216. The product 12 × 18 = 216 is a common multiple, but it is not the least one.

## 4. Digits in disguise

### Think it through

In a letter-digit puzzle, the same letter keeps the same digit, different letters use different digits, and the first digit of a multi-digit number cannot be zero. For AB + BA = 99, write AB as 10A + B, not A multiplied by B. The sum is 11(A + B), so A + B = 9. More than one pair can work; the equation alone does not force a unique answer.

For example, A = 2, B = 7 gives 27 + 72 = 99; A = 4, B = 5 gives 45 + 54 = 99. Neither letter can be zero here because both AB and BA must be two-digit numbers. An extra condition can narrow the choices, but do not assume it leaves exactly one. First turn the digit notation into place values, then test every stated condition.

### Try it yourself

Use the same digit rules for AB + BA = 121. This time A is even and A < B. Find every assignment, not just one. A learner says these extra conditions must make the answer unique. Is that justified?

**Check:** 11(A + B) = 121 gives A + B = 11. Both digits are nonzero, different and at most 9. With A even and smaller than B, the only possibilities are A = 2, B = 9 and A = 4, B = 7: 29 + 92 = 121 and 47 + 74 = 121. Trying A = 6 or 8 makes B smaller than A; A = 0 is forbidden and would require B = 11 anyway. There are two assignments, so the uniqueness claim fails.

## 5. Consecutive numbers and parity

### Think it through

Four consecutive integers contain two odd and two even numbers. Their sum is even. Algebra makes the scope clear: n + (n + 1) + (n + 2) + (n + 3) = 4n + 6, which is twice 2n + 3. Replacing a +n term with -n changes the result by -2n, an even number, so changing plus and minus signs does not change that parity. You can explain every sign choice without calculating each one.

For example, 23 - 24 + 25 - 26 is even before it is evaluated; its value is -2. Negative integers can be even too. The parity argument does not say every sign choice has the same value. Nor does a result about four consecutive integers automatically apply to a different number of terms.

### Try it yourself

Take five consecutive whole numbers n, n + 1, n + 2, n + 3 and n + 4. Put either a plus or a minus sign before each term. A learner says every resulting total must be even, as it was for four consecutive numbers. Decide in terms of n without listing all sign choices. Could any choice for 7, 8, 9, 10 and 11 give zero?

**Check:** With all plus signs the sum is 5n + 10, which has the same parity as n. Flipping any sign changes the total by an even number, so every choice stays even when n is even and odd when n is odd. The learner's always-even rule is false. For 7 through 11, n = 7 is odd, so every signed total is odd. Zero is even, so no sign choice can give zero.

## 6. Remainders as algebra

### Think it through

A whole number that leaves remainder 5 when divided by 7 can be written 7a + 5, where a is a whole number. Another with remainder 4 is 7b + 4. Their sum is 7(a + b) + 9 = 7(a + b + 1) + 2. The remainder is 2, not 9: a remainder after division by 7 must be smaller than 7. For example, 47 + 25 = 72 = 7 × 10 + 2.

The letters hold the unknown complete groups of 7. You do not need to know those groups to reason about the leftover amount. For a product, expand the expressions: every term containing a factor of 7 makes complete groups. Reduce what remains to a number from 0 to 6. Adding remainders and multiplying remainders are different operations; neither permits a final remainder of 7 or more.

### Try it yourself

A whole number N leaves remainder 6 when divided by 7. Without knowing N, decide whether N + 1 is divisible by 7 and find the remainder of N². A learner says the square must leave remainder 36. Explain the correction and check one possible value of N.

**Check:** Write N = 7k + 6. Then N + 1 = 7(k + 1), so it is divisible by 7. Also N² = 49k² + 84k + 36 = 7(7k² + 12k + 5) + 1. The remainder is 1, not 36: the 36 contains five complete groups of 7 and one left over. For N = 13, N + 1 = 14 and N² = 169 = 7 × 24 + 1. N itself is not divisible by 7; its given remainder is 6.

## 7. Digital roots: a check, not a proof

### Think it through

For a positive whole number, repeatedly add its decimal digits until one digit remains. This is its digital root. A multiple of 9 has digital root 9, not 0. To check a product, multiply the two factors' digital roots and reduce that result to one digit too. Compare it with the proposed answer's digital root.

For example, 46 and 28 each have digital root 1, so their product should have digital root 1. A proposed answer of 1,297 passes that test, but multiplication gives 1,288 instead. Both answers have digital root 1 because they differ by 9. Digital roots can expose some arithmetic mistakes, but two different numbers can have the same digital root. Passing the check does not prove a calculation is right.

### Try it yourself

Two learners give 897 and 898 as answers to 37 × 24. Which answer can you reject using digital roots alone? Does the other answer become certified correct? Multiply to settle the result and explain any error the root check misses.

**Check:** 37 has digital root 1 and 24 has digital root 6, so the product must have digital root 6. For 897, 8 + 9 + 7 = 24 and 2 + 4 = 6, so that answer passes. For 898, the digit sum is 25 and its digital root is 7, so it can be rejected. Actual multiplication gives 37 × 24 = 888, whose digital root is also 6. Thus both proposed answers are wrong: 897 passes because it differs from 888 by 9. Passing is not the same as proving.

## Your five-minute revision

Choose one idea you can now explain and one you still find tricky. Draw a diagram or work an example from memory. Compare with the lesson and fix the step you missed, then retry it tomorrow.

[Try an Evater sample paper](/papers/) and choose questions that match what you have practised.
