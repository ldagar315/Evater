-- Media is learner-visible question content. Keep answer_spec and generation
-- metadata private while extending the existing column-level public grant.
grant select (media_json, option_media_json)
  on table public.question_bank to authenticated;
