from pathlib import Path


MIGRATION = Path(__file__).parents[1] / "supabase" / "migrations" / "20260715120000_question_bank_mvp.sql"


def test_question_bank_migration_keeps_answer_key_out_of_authenticated_grant():
    sql = MIGRATION.read_text(encoding="utf-8")
    grant_start = sql.index("grant select (", sql.index("-- Column-level access"))
    granted_columns = sql[grant_start : sql.index(") on table", grant_start)]

    assert "question_text" in granted_columns
    assert "options_json" in granted_columns
    assert "correct_option_id" not in granted_columns
    assert "alter table public.question_bank enable row level security;" in sql
