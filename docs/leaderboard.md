# Student leaderboard MVP

The first leaderboard release is intentionally limited to authenticated learners who have an explicit row in `public.student_enrollments`.

## Scope

- `classroom`: learners with the same `classroom_id`.
- `school`: learners with the same `school_id`.

The API resolves the caller's membership on the server. Free-form `Users.school` and `Users.grade` values are not used to authorize or filter a leaderboard.

## Score

The first release uses a simple, explainable practice score:

- 25 points for each completed practice test.
- 10 points for each correct answer in a completed practice test.

Rank ties are broken by correct answers, completed tests, privacy-safe display name, and finally the user ID for deterministic ordering. The default period is the last seven days; all-time scores are also available.

## Local stage

The leaderboard migration seeds two schools and four classrooms. Assign a local auth user to `student_enrollments` before testing the ranking. The API returns a clear membership state when an account has not been assigned to a classroom.
