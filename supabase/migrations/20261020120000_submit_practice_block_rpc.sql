-- =====================================================================
-- SPEC - NOT YET APPLIED. Reviewed-by-owner required before applying.
--
-- One-RPC Finish: submit_practice_block() performs an entire
-- block submit (validate, score, attempts, performance cache, mastery,
-- completion, seasonal leaderboard) in a single Postgres transaction,
-- replacing ~20 sequential backend->Supabase HTTPS round trips with one
-- RPC call from the Modal container.
--
-- Scope and known boundaries (read before applying):
--  1. Scoring is ported for single-select MCQ only (selected_option_id =
--     correct_option_id, full marks or zero). Other question types must
--     keep using the Python endpoint until ported; the function raises
--     'unsupported question type' when correct_option_id is missing so
--     the backend can fall back.
--  2. Season finalization (resolve_season_movements, inactivity carry)
--     stays in the read path. This function creates the current season
--     row if missing and ensures the player row, porting
--     _next_player_state from backend/app/routers/leaderboard_router.py.
--  3. Next-block adaptive selection stays in Python. For non-final
--     blocks the backend calls this RPC, then selects and inserts the
--     next block (2 extra calls); Finish (final block) is 1 call.
--  4. Everything runs as one transaction: any failure rolls back all
--     writes, which matches the current all-or-error behaviour.
--
-- Rollback: drop function public.submit_practice_block(uuid, integer, uuid, jsonb);
-- =====================================================================

create or replace function public.submit_practice_block(
    p_test_id uuid,
    p_block_number integer,
    p_user_id uuid,
    p_answers jsonb
) returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
    v_test record;
    v_now timestamptz := now();
    v_block record;
    v_answer jsonb;
    v_q record;
    v_scored jsonb := '[]'::jsonb;
    v_correct boolean;
    v_marks integer;
    v_block_score integer := 0;
    v_block_total integer := 0;
    v_current_block int;
    v_answered_count int;
    v_remaining int;
    v_completed boolean := false;
    v_final_score integer;
    v_maximum_marks integer;
    -- leaderboard
    v_points_awarded integer;
    v_raw_score integer;
    v_season record;
    v_player record;
    v_prev_player record;
    v_season_number int;
    v_starts_at timestamptz;
    v_ends_at timestamptz;
    v_qvalue int;
    v_qscore int;
    v_correct_answers int := 0;
    v_scored_count int := 0;
    v_multiplier int;
    -- mastery
    v_concept record;
    v_new_mastery numeric;
    v_prev_mastery record;
    v_status text;
    v_mastery_updates jsonb := '[]'::jsonb;
begin
    -- 1. Lock and validate the attempt ----------------------------------
    select * into v_test
    from test_attempts
    where id = p_test_id and user_id = p_user_id
    for update;

    if not found then
        raise exception 'test attempt not found' using errcode = 'P0002';
    end if;
    if v_test.status <> 'in_progress' then
        raise exception 'test no longer accepting answers' using errcode = 'P0001';
    end if;
    if v_test.current_block <> p_block_number then
        raise exception 'not the current block' using errcode = 'P0001';
    end if;

    -- 2. Score the block (single-select MCQ port of score_block) --------
    for v_block in
        select tq.id as test_question_id, tq.question_id, tq.display_order,
               qb.correct_option_id, qb.concept_id, qb.marks, qb.difficulty
        from test_questions tq
        join question_bank qb on qb.id = tq.question_id
        where tq.test_attempt_id = p_test_id and tq.block_number = p_block_number
        order by tq.display_order
    loop
        select value into v_answer from jsonb_array_elements(p_answers) value
        where (value->>'question_id')::uuid = v_block.question_id;
        if v_answer is null then
            raise exception 'missing answer for question %', v_block.question_id using errcode = 'P0001';
        end if;
        if v_block.correct_option_id is null then
            raise exception 'unsupported question type' using errcode = 'P0001';
        end if;

        v_correct := coalesce(v_answer->>'selected_option_id', '') = v_block.correct_option_id;
        v_marks := case when v_correct then coalesce(v_block.marks, 1) else 0 end;
        v_block_score := v_block_score + v_marks;
        v_block_total := v_block_total + coalesce(v_block.marks, 1);

        v_scored := v_scored || jsonb_build_object(
            'test_question_id', v_block.test_question_id,
            'question_id', v_block.question_id,
            'concept_id', v_block.concept_id,
            'difficulty', coalesce(v_block.difficulty, 'easy'),
            'is_correct', v_correct,
            'marks_awarded', v_marks,
            'selected_option_id', v_answer->>'selected_option_id',
            'response_time_ms', (v_answer->>'response_time_ms')::int
        );
    end loop;

    if jsonb_array_length(v_scored) <> jsonb_array_length(p_answers) then
        raise exception 'submit exactly one answer per question in this block' using errcode = 'P0001';
    end if;

    -- 3. Idempotency: block must not already be submitted ----------------
    if exists (
        select 1 from question_attempts qa
        join test_questions tq on tq.id = qa.test_question_id
        where tq.test_attempt_id = p_test_id and tq.block_number = p_block_number
    ) then
        raise exception 'block already submitted' using errcode = 'P0001';
    end if;

    -- 4. Insert question_attempts ----------------------------------------
    insert into question_attempts (
        test_attempt_id, test_question_id, user_id, chapter_id, question_id,
        selected_option_id, answer_json, is_correct, marks_awarded, response_time_ms
    )
    select p_test_id,
           (s->>'test_question_id')::uuid,
           p_user_id,
           v_test.chapter_id,
           (s->>'question_id')::uuid,
           s->>'selected_option_id',
           jsonb_build_object('value', s->>'selected_option_id'),
           (s->>'is_correct')::boolean,
           (s->>'marks_awarded')::integer,
           (s->>'response_time_ms')::integer
    from jsonb_array_elements(v_scored) s;

    -- 5. student_question_performance cache (port of _update_question_performance)
    insert into student_question_performance as sqp (
        user_id, question_id, chapter_id, attempt_count, correct_count, wrong_count,
        consecutive_wrong_count, last_answer_correct, last_selected_option_id,
        last_answer_json, last_response_time_ms, first_attempted_at,
        last_attempted_at, last_correct_at, last_wrong_at, next_review_at, updated_at
    )
    select p_user_id,
           (s->>'question_id')::uuid,
           v_test.chapter_id,
           1,
           case when (s->>'is_correct')::boolean then 1 else 0 end,
           case when (s->>'is_correct')::boolean then 0 else 1 end,
           case when (s->>'is_correct')::boolean then 0 else 1 end,
           (s->>'is_correct')::boolean,
           s->>'selected_option_id',
           jsonb_build_object('value', s->>'selected_option_id'),
           (s->>'response_time_ms')::integer,
           v_now, v_now,
           case when (s->>'is_correct')::boolean then v_now end,
           case when not (s->>'is_correct')::boolean then v_now end,
           v_now + case when (s->>'is_correct')::boolean then interval '3 days' else interval '1 day' end,
           v_now
    from jsonb_array_elements(v_scored) s
    on conflict (user_id, question_id) do update set
        attempt_count = sqp.attempt_count + 1,
        correct_count = sqp.correct_count + case when excluded.last_answer_correct then 1 else 0 end,
        wrong_count = sqp.wrong_count + case when excluded.last_answer_correct then 0 else 1 end,
        consecutive_wrong_count = case when excluded.last_answer_correct then 0 else sqp.consecutive_wrong_count + 1 end,
        last_answer_correct = excluded.last_answer_correct,
        last_selected_option_id = excluded.last_selected_option_id,
        last_answer_json = excluded.last_answer_json,
        last_response_time_ms = excluded.last_response_time_ms,
        last_attempted_at = v_now,
        last_correct_at = case when excluded.last_answer_correct then v_now else sqp.last_correct_at end,
        last_wrong_at = case when excluded.last_answer_correct then sqp.last_wrong_at else v_now end,
        next_review_at = excluded.next_review_at,
        updated_at = v_now;

    -- 6. Mastery (port of update_mastery: 0.7*prev + 0.3*accuracy*100) ---
    for v_concept in
        select (s->>'concept_id')::uuid as concept_id,
               avg(case when (s->>'is_correct')::boolean then 1.0 else 0.0 end) as accuracy,
               count(*) as attempts,
               (s->>'difficulty') as last_difficulty
        from jsonb_array_elements(v_scored) s
        group by 1
    loop
        select * into v_prev_mastery from concept_mastery
        where user_id = p_user_id and chapter_id = v_test.chapter_id and concept_id = v_concept.concept_id;

        v_new_mastery := round(
            (0.70 * coalesce(v_prev_mastery.mastery_score, 0)::numeric)
            + (0.30 * v_concept.accuracy * 100)::numeric, 2);
        v_status := case when v_new_mastery < 45 then 'needs_review'
                         when v_new_mastery >= 75 then 'challenge_next'
                         else 'on_track' end;

        insert into concept_mastery as cm (
            user_id, chapter_id, concept_id, mastery_score, attempt_count,
            correct_count, last_difficulty, misconception_signals,
            last_attempted_at, next_review_at, updated_at
        ) values (
            p_user_id, v_test.chapter_id, v_concept.concept_id, v_new_mastery,
            v_concept.attempts,
            (select count(*) from jsonb_array_elements(v_scored) s
             where (s->>'concept_id')::uuid = v_concept.concept_id and (s->>'is_correct')::boolean),
            v_concept.last_difficulty,
            coalesce(v_prev_mastery.misconception_signals, '{}'::jsonb),
            v_now,
            v_now + case v_status when 'needs_review' then interval '1 day'
                                  when 'challenge_next' then interval '7 days'
                                  else interval '3 days' end,
            v_now
        )
        on conflict (user_id, chapter_id, concept_id) do update set
            mastery_score = excluded.mastery_score,
            attempt_count = cm.attempt_count + excluded.attempt_count,
            correct_count = cm.correct_count + excluded.correct_count,
            last_difficulty = excluded.last_difficulty,
            last_attempted_at = excluded.last_attempted_at,
            next_review_at = excluded.next_review_at,
            updated_at = excluded.updated_at;

        v_mastery_updates := v_mastery_updates || jsonb_build_object(
            'concept_id', v_concept.concept_id,
            'mastery_score', v_new_mastery,
            'status', v_status
        );
    end loop;

    -- 7. Completion --------------------------------------------------------
    v_current_block := v_test.current_block;
    v_answered_count := v_current_block * v_test.block_size;
    v_remaining := greatest(v_test.question_count - v_answered_count, 0);

    if v_remaining = 0 then
        v_completed := true;

        select coalesce(sum(qa.marks_awarded), 0) into v_final_score
        from question_attempts qa where qa.test_attempt_id = p_test_id;

        select coalesce(sum(coalesce(qb.marks, 1)), 0) into v_maximum_marks
        from test_questions tq join question_bank qb on qb.id = tq.question_id
        where tq.test_attempt_id = p_test_id;

        update test_attempts set
            status = 'completed',
            score = v_final_score,
            maximum_marks = v_maximum_marks,
            completed_at = v_now
        where id = p_test_id;

        -- 8. Seasonal leaderboard (port of award_completed_practice) ------
        if v_test.mode = 'practice'
           and exists (select 1 from student_enrollments
                       where user_id = p_user_id and role = 'student') then

            select points_awarded, raw_score into v_points_awarded, v_raw_score
            from leaderboard_practice_scores where test_attempt_id = p_test_id;

            if not found then
                -- current season window (epoch Monday 2026-01-05 UTC, 14 days)
                v_season_number := floor(
                    extract(epoch from (v_now - timestamptz '2026-01-05 00:00:00+00'))
                    / (14 * 86400))::int + 1;
                v_starts_at := timestamptz '2026-01-05 00:00:00+00' + (v_season_number - 1) * interval '14 days';
                v_ends_at := v_starts_at + interval '14 days';

                insert into leaderboard_seasons (season_key, season_number, starts_at, ends_at, status)
                values ('season-' || lpad(v_season_number::text, 4, '0'), v_season_number, v_starts_at, v_ends_at, 'active')
                on conflict (season_key) do nothing;

                select * into v_season from leaderboard_seasons
                where season_key = 'season-' || lpad(v_season_number::text, 4, '0');

                -- player row, porting _next_player_state (simplified: no
                -- inactivity penalty when creating mid-season; see spec note 2)
                select * into v_player from leaderboard_players
                where season_id = v_season.id and user_id = p_user_id;

                if not found then
                    select * into v_prev_player from leaderboard_players
                    where user_id = p_user_id
                    order by created_at desc limit 1;

                    insert into leaderboard_players (
                        season_id, user_id, league_tier, points, completed_practices,
                        correct_answers, inactive_seasons, created_at, updated_at
                    ) values (
                        v_season.id, p_user_id,
                        coalesce(v_prev_player.final_tier, v_prev_player.league_tier, 'bronze_3'),
                        case when coalesce(v_prev_player.completed_practices, 0) > 0
                             then coalesce(v_prev_player.points, 0) else 0 end,
                        0, 0, 0, v_now, v_now
                    )
                    returning * into v_player;
                end if;

                -- per-question scoring: 4/8/12, wrong -25%, unanswered -50%
                v_raw_score := 0;
                for v_q in
                    select (s->>'question_id')::uuid as question_id,
                           lower(coalesce(qb.difficulty, 'easy')) as difficulty,
                           (s->>'is_correct')::boolean as is_correct,
                           (s->>'selected_option_id') is not null as is_answered
                    from jsonb_array_elements(v_scored) s
                    join question_bank qb on qb.id = (s->>'question_id')::uuid
                    where not exists (
                        select 1 from leaderboard_question_scores lqs
                        where lqs.season_id = v_season.id and lqs.user_id = p_user_id
                          and lqs.question_id = (s->>'question_id')::uuid
                    )
                loop
                    v_qvalue := case v_q.difficulty when 'medium' then 8 when 'hard' then 12 else 4 end;
                    v_qscore := case when v_q.is_correct then v_qvalue
                                     when not v_q.is_answered then -(v_qvalue / 2)
                                     else -(v_qvalue / 4) end;
                    v_raw_score := v_raw_score + v_qscore;
                    v_correct_answers := v_correct_answers + case when v_q.is_correct then 1 else 0 end;
                    v_scored_count := v_scored_count + 1;

                    insert into leaderboard_question_scores (
                        season_id, user_id, question_id, test_attempt_id, points_awarded
                    ) values (v_season.id, p_user_id, v_q.question_id, p_test_id, v_qscore);
                end loop;

                v_raw_score := greatest(0, v_raw_score);
                v_multiplier := case split_part(coalesce(v_player.league_tier, 'bronze_3'), '_', 1)
                                when 'silver' then 90 when 'gold' then 80
                                when 'platinum' then 70 when 'diamond' then 60
                                else 100 end;
                v_points_awarded := case when v_raw_score <= 0 then 0
                                         else greatest(1, floor(v_raw_score * v_multiplier / 100)) end;

                insert into leaderboard_practice_scores (
                    test_attempt_id, season_id, user_id, raw_score, points_awarded,
                    scored_question_count, correct_answers, completed_at
                ) values (
                    p_test_id, v_season.id, p_user_id, v_raw_score, v_points_awarded,
                    v_scored_count, v_correct_answers, v_now
                );

                update leaderboard_players set
                    points = coalesce(v_player.points, 0) + v_points_awarded,
                    completed_practices = coalesce(v_player.completed_practices, 0) + 1,
                    correct_answers = coalesce(v_player.correct_answers, 0) + v_correct_answers,
                    last_activity_at = v_now,
                    last_decay_at = null,
                    inactive_seasons = 0,
                    inactivity_baseline_points = null,
                    inactivity_floor_reached = false,
                    updated_at = v_now
                where season_id = v_season.id and user_id = p_user_id;
            end if;
        end if;
    end if;

    return jsonb_build_object(
        'test_id', p_test_id,
        'block_number', p_block_number,
        'block_score', v_block_score,
        'block_total', v_block_total,
        'percentage', case when v_block_total > 0 then round(v_block_score::numeric / v_block_total * 100, 2) else 0 end,
        'completed', v_completed,
        'final_score', v_final_score,
        'maximum_marks', v_maximum_marks,
        'mastery_updates', v_mastery_updates,
        'scored', v_scored,
        'leaderboard_points_awarded', v_points_awarded,
        'leaderboard_raw_score', v_raw_score
    );
end;
$$;

-- Service-role only: students must not call this directly.
revoke all on function public.submit_practice_block(uuid, integer, uuid, jsonb) from public, anon, authenticated;
grant execute on function public.submit_practice_block(uuid, integer, uuid, jsonb) to service_role;
