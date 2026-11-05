/**
 * Boundary adapters between the generated wire types (api.generated.ts) and
 * the UI's local shapes. Each function preserves the EXACT runtime behavior
 * of the existing inline conversion it names - no validation or fallback is
 * added or removed here, so adopting an adapter at a call site is behavior-
 * neutral. Known wire/UI mismatches stay type-visible (e.g. CreateTestPage's
 * ApiQuestion.options: string[] vs the wire MCQOption[] surfaces as TS2322
 * under `npx tsc -b`); this file does not paper over them.
 */
import type { components } from './api.generated'

export type WireQuestion = components['schemas']['Question']
export type WireChapterInfo = components['schemas']['ChapterInfo']
export type WireChapterListResponse = components['schemas']['ChapterListResponse']

/** Backend wire casing: "Easy" | "Medium" | "Hard" (Literal in backend models). */
export type WireDifficulty = WireQuestion['difficulty']
/** UI/Supabase casing: lowercase, e.g. Questions_Created.difficulty_level. */
export type UiDifficulty = 'easy' | 'medium' | 'hard'

/**
 * Wire -> store. Preserves CreateTestPage (convertedQuestions) exactly:
 * blind lowercase + cast, NO fallback - an unexpected backend value flows
 * through lowercased, exactly as the current `q.difficulty.toLowerCase() as
 * "easy" | "medium" | "hard"` behaves.
 */
export function wireDifficultyToStore(d: string): UiDifficulty {
  return d.toLowerCase() as UiDifficulty
}

/**
 * Store -> wire. Preserves ViewTestPage's capitalizeDifficulty exactly:
 * case-insensitive title-case mapping with a 'Medium' fallback for values
 * outside the known set.
 */
export function storeDifficultyToWire(d: string): WireDifficulty {
  const map: Record<string, WireDifficulty> = {
    easy: 'Easy',
    medium: 'Medium',
    hard: 'Hard',
  }
  return map[d.toLowerCase()] ?? 'Medium'
}
