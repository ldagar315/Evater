export interface Question {
  question_text: string
  question_type: 'short_answer' | 'long_answer' | 'mcq_single' | 'mcq_multi' | 'true_false'
  question_number: number
  maximum_marks: number
  difficulty: 'easy' | 'medium' | 'hard'
  contains_math_expression: boolean
  options?: Array<{
    text: string
    is_correct?: boolean
  }>
}

export interface TestGenerationParams {
  grade: string
  subject: string
  chapter: string
  difficulty_level: 'easy' | 'medium' | 'hard'
  length: string
  special_instructions: string[]
}

export interface Answer {
  answer: string
}

export interface Feedback {
  max_scored: number
  error_type: string
  explanation: string
  next_step: string
}

export interface FeedbackResponse {
  question_number: number
  question_text: string
  question_type: string
  maximum_marks: number
  answer: Answer
  feedback: Feedback
}

// Database table types matching your schema
export interface UserProfile {
  id: string
  user_name?: string | null
  email?: string | null
  profile_picture?: string | null
  created_by: string
  grade?: number | null
  school?: string | null
  credits?: number | null
  name?: string | null
  created_at: string
}

export interface QuestionsCreated {
  id: number
  created_at: string
  grade?: string
  subject?: string
  chapter?: string
  difficulty_level?: string
  length?: string
  special_instructions?: string[]
  test: any // jsonb
  created_by: string
}

export interface AnswerSheetImage {
  id: number
  created_at: string
  image_url?: string[]
  uploaded_by: string
  for_test?: number
  feedback?: number
}

export interface FeedbackTest {
  id: number
  created_at: string
  given_by: string
  for_test?: number
  feedback: any // jsonb
}

export interface ChapterContent {
  id: number
  created_at: string
  grade?: string
  subject?: string
  board?: string
  summary?: string
  chapter?: string
}

export interface Test {
  id: string
  user_id: string
  subject: string
  topic: string
  difficulty: 'easy' | 'medium' | 'hard'
  questions: any // jsonb
  created_at: string
}

export interface Evaluation {
  id: string
  test_id: string
  answer_sheets: string[]
  feedback: any // jsonb
  created_at: string
}