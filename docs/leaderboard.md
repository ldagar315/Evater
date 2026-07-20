# Global seasonal student league

The leaderboard is a simple global league for verified students. A verified
student is currently represented by a `student_enrollments` row with
`role = 'student'`. The backend owns all league writes; the browser only reads
the API response.

## Season

- Seasons last 14 days and use stable UTC boundaries.
- Trophy points carry forward when a season ends; the student's league tier carries forward too.
- Promotion and demotion are resolved only at the season boundary.
- One eligible student is promoted from each league.
- The bottom student is demoted only when the league has at least three students.
- Bronze III cannot demote and Diamond I cannot promote.

## Practice scoring

Only completed normal practice sessions score league points. Question values
are integer-scaled so penalties never produce decimals:

| Difficulty | Correct | Wrong | Unanswered |
| --- | ---: | ---: | ---: |
| Easy | +4 | -1 | -2 |
| Medium | +8 | -2 | -4 |
| Hard | +12 | -3 | -6 |

The raw practice score has a floor of zero. A question can contribute league
points only once per student per season; reattempts still support mastery but
do not allow leaderboard farming.

Higher leagues apply a multiplier to the raw practice score. The result is
floored and any positive practice retains a minimum award of one point:

- Bronze: 100%
- Silver: 90%
- Gold: 80%
- Platinum: 70%
- Diamond: 60%

## League ladder

The ladder runs from Bronze III through Bronze I, then Silver, Gold, Platinum,
and Diamond. `III` is the lowest tier and `I` is the highest within each metal.
Promotion requires the league's configured season threshold and first place in
that league. Tie-breakers are total league points, correct answers, completed
practices, then a stable user identifier.

Configured promotion thresholds live in `backend/app/leaderboard.py` so they
can be tuned after real traffic without changing the scoring math.

## Inactivity

The first full season without a completed practice is protected: the student
cannot be penalized or demoted. Starting with the next consecutive inactive
season, the carried trophy balance loses 10% once per season. This reduction is
capped at 50% of the trophy balance held when the inactive streak began; once
that floor is reached, both the reduction and demotion stop. Completing a
practice resets the inactive-season streak.
