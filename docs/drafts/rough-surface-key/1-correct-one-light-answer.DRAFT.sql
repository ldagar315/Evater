-- DRAFT ONLY. Requires explicit approval and real PostgreSQL rehearsal before any execution.
-- Source Q30: https://www.tiwariacademy.in/ncert-solutions-class-8-science-curiosity-chapter-10/
-- Change exactly one currently published row. Preserve rights/review pending state.
-- No historical attempts/mastery/leaderboard repair is included or authorized.
BEGIN;
DO $$
DECLARE changed integer;
BEGIN
  UPDATE public.question_bank
  SET correct_option_id = 'B',
      answer_spec = jsonb_set(answer_spec, '{source_answer}', '"B"'::jsonb),
      content_hash = 'd4112bc530f955ec7c194ff0928122838eb768307e3f9898cd115bc8fc2ab366'
  WHERE id = '6aca65ae-e742-49a6-b542-ce8c11c02ee4'::uuid
    AND status = 'published'
    AND correct_option_id = 'C'
    AND content_hash = 'b28b613657717d3ac6ab2d045744ba212f5dce504b6ccbd03708e111fb58000e'
    AND question_text = 'What happens when light hits a rough surface?'
    AND answer_spec->>'source_answer' = 'C'
    AND NOT (answer_spec ? 'correct_option_id')
    AND options_json = '[{"id": "A", "text": "No reflection"}, {"id": "B", "text": "Diffused reflection"}, {"id": "C", "text": "Real image"}, {"id": "D", "text": "Regular reflection"}]'::jsonb;
  GET DIAGNOSTICS changed = ROW_COUNT;
  IF changed <> 1 THEN RAISE EXCEPTION 'Expected exactly one unchanged target row, got %', changed; END IF;
END $$;
-- Rehearsal default. Switch to COMMIT only with explicit production approval.
ROLLBACK;
