import { supabase } from './supabase'

const DEFAULT_MODAL_API_URL = 'https://ldagar315--evater-v1-wrapper.modal.run'

export const MODAL_API_URL = (
  import.meta.env.VITE_MODAL_API_URL || DEFAULT_MODAL_API_URL
).replace(/\/+$/, '')

export const MODAL_WS_URL = (
  import.meta.env.VITE_MODAL_WS_URL || `${MODAL_API_URL.replace(/^http/, 'ws')}/ws/viva`
).replace(/\/+$/, '')

export async function getAccessToken(): Promise<string | null> {
  const {
    data: { session },
  } = await supabase.auth.getSession()

  return session?.access_token ?? null
}

export async function apiFetch(
  input: RequestInfo | URL,
  init: RequestInit = {},
): Promise<Response> {
  const headers = new Headers(init.headers)
  const accessToken = await getAccessToken()

  if (accessToken) {
    headers.set('Authorization', `Bearer ${accessToken}`)
  }

  return fetch(input, {
    ...init,
    headers,
  })
}

export const CLASS8_SCIENCE_CHAPTER1_ID =
  '22222222-2222-4222-8222-222222222222'

export type QuestionBankChapter = {
  id: string
  sequence_number: number
  title: string
  description?: string | null
}

export type QuestionBankOption = {
  id: string
  text: string
  media?: QuestionBankMedia[]
}

export type QuestionBankType =
  | 'mcq_single'
  | 'mcq_multi'
  | 'assertion_reason'
  | 'true_false'
  | 'fill_blank'
  | 'numerical'
  | 'case_study'
  | 'diagram_based'
  | 'matching'

export type QuestionBankMedia = {
  type?: 'image' | 'audio' | 'video' | string
  url?: string
  src?: string
  storage_path?: string
  alt?: string
  caption?: string
}

export type QuestionBankQuestion = {
  id: string
  concept_id: string
  question_type: QuestionBankType
  question_text: string
  options: QuestionBankOption[]
  media?: QuestionBankMedia[]
  render_config?: Record<string, unknown>
  explanation?: string | null
  hint?: string | null
  difficulty: 'easy' | 'medium' | 'hard'
  cognitive_level: 'recall' | 'understand' | 'apply' | 'analyze'
  skill_tags: string[]
  misconception_tags: string[]
  question_style: 'direct' | 'scenario' | 'experiment' | 'data' | 'diagram'
  estimated_time_seconds: number
  maximum_marks: number
}

export type QuestionBankRouting = {
  difficulty: 'easy' | 'medium' | 'hard'
  focus_concept_ids: string[]
  status: 'needs_review' | 'on_track' | 'challenge_next'
}

export type QuestionBankMode = 'diagnostic' | 'practice' | 'remedial' | 'challenge'

export type QuestionBankTest = {
  test_id: string
  block_number: number
  block_size: number
  question_count: number
  questions: QuestionBankQuestion[]
  routing: QuestionBankRouting
}

export type QuestionBankBlockResult = {
  question_id: string
  is_correct: boolean
  marks_awarded: number
  maximum_marks: number
  explanation?: string | null
  selected_option_id?: QuestionBankOption['id'] | null
  correct_option_id?: QuestionBankOption['id'] | null
  selected_option_ids?: string[]
  correct_option_ids?: string[]
  answer_summary?: string | null
  correct_answer_summary?: string | null
}

export type QuestionBankBlockSubmissionResult = {
  test_id: string
  block_number: number
  block_score: number
  block_total: number
  percentage: number
  results: QuestionBankBlockResult[]
  mastery_updates: Array<Record<string, unknown>>
  next_block?: QuestionBankTest | null
  completed: boolean
}

async function parseApiResponse<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = typeof payload?.detail === 'string' ? payload.detail : 'The request failed.'
    throw new Error(detail)
  }
  return payload as T
}

export async function createQuestionBankTest(
  questionCount = 10,
  blockSize = 5,
  chapterId = CLASS8_SCIENCE_CHAPTER1_ID,
  mode: QuestionBankMode = 'practice',
): Promise<QuestionBankTest> {
  const response = await apiFetch(`${MODAL_API_URL}/api/v1/tests`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      chapter_id: chapterId,
      mode,
      question_count: questionCount,
      block_size: blockSize,
    }),
  })
  return parseApiResponse<QuestionBankTest>(response)
}

export async function getQuestionBankChapters(): Promise<QuestionBankChapter[]> {
  const response = await apiFetch(`${MODAL_API_URL}/api/v1/chapters`)
  const payload = await parseApiResponse<{ chapters: QuestionBankChapter[] }>(response)
  return payload.chapters
}

export async function submitQuestionBankBlock(
  testId: string,
  blockNumber: number,
  answers: Array<{ question_id: string; selected_option_id?: QuestionBankOption['id'] | null; answer?: unknown }>,
): Promise<QuestionBankBlockSubmissionResult> {
  const response = await apiFetch(
    `${MODAL_API_URL}/api/v1/tests/${testId}/blocks/${blockNumber}/submit`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ answers }),
    },
  )
  return parseApiResponse<QuestionBankBlockSubmissionResult>(response)
}

export async function flagQuestionBankQuestion(
  questionId: string,
): Promise<{ question_id: string; flagged: boolean }> {
  const response = await apiFetch(`${MODAL_API_URL}/api/v1/questions/${questionId}/flag`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  })
  return parseApiResponse<{ question_id: string; flagged: boolean }>(response)
}

export type LeaderboardScope = 'classroom' | 'school'
export type LeaderboardPeriod = 'weekly' | 'all_time'

export type LeaderboardEntry = {
  rank: number
  display_name: string
  score: number
  correct_answers: number
  completed_tests: number
  is_current_user: boolean
}

export type LeaderboardResponse = {
  scope: LeaderboardScope
  scope_label: string
  period: LeaderboardPeriod
  period_label: string
  scope_available: boolean
  membership_message?: string | null
  entries: LeaderboardEntry[]
  current_user_rank?: number | null
}

export async function getLeaderboard(
  scope: LeaderboardScope = 'classroom',
  period: LeaderboardPeriod = 'weekly',
): Promise<LeaderboardResponse> {
  const params = new URLSearchParams({ scope, period })
  const response = await apiFetch(`${MODAL_API_URL}/api/v1/leaderboard?${params.toString()}`)
  return parseApiResponse<LeaderboardResponse>(response)
}
