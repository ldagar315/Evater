import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

let supabase: any

const notConfiguredError = () => ({ message: 'Supabase not configured' })

const createFallbackQuery = () => {
  const result = Promise.resolve({ data: null, error: notConfiguredError() })
  const chain: any = {
    select: () => chain,
    insert: () => chain,
    update: () => chain,
    delete: () => chain,
    eq: () => chain,
    neq: () => chain,
    order: () => chain,
    limit: () => chain,
    single: () => result,
    maybeSingle: () => result,
    then: (onFulfilled: any, onRejected: any) => result.then(onFulfilled, onRejected),
    catch: (onRejected: any) => result.catch(onRejected),
  }
  return chain
}

const createFallbackClient = () => ({
  auth: {
    getSession: async () => ({
      data: { session: null },
      error: notConfiguredError(),
    }),
    onAuthStateChange: (callback: any) => {
      if (import.meta.env.DEV) {
        setTimeout(() => callback('SIGNED_OUT', null), 0)
      }
      return {
        data: {
          subscription: {
            unsubscribe: () => undefined,
          },
        },
      }
    },
    signUp: async () => ({ data: null, error: notConfiguredError() }),
    signInWithPassword: async () => ({ data: null, error: notConfiguredError() }),
    signInWithOAuth: async () => ({ data: null, error: notConfiguredError() }),
    signOut: async () => ({ error: null }),
  },
  from: () => createFallbackQuery(),
  storage: {
    from: () => ({
      upload: async () => ({ data: null, error: notConfiguredError() }),
      getPublicUrl: () => ({ data: { publicUrl: '' }, error: notConfiguredError() }),
    }),
  },
})

if (!supabaseUrl || !supabaseAnonKey) {
  if (import.meta.env.DEV) {
    console.warn('Missing Supabase environment variables - using fallback mode')
  }
  supabase = createFallbackClient()
} else {
  try {
    supabase = createClient(supabaseUrl, supabaseAnonKey)
  } catch (error) {
    if (import.meta.env.DEV) {
      console.error('Failed to create Supabase client:', error)
    }
    supabase = createFallbackClient()
  }
}

export { supabase }

export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export interface Database {
  public: {
    Tables: {
      Users: {
        Row: {
          id: string
          created_at: string
          user_name: string | null
          email: string | null
          profile_picture: string | null
          created_by: string
          grade: number | null
          school: string | null
          credits: number | null
          name: string | null
          class_level: number | null
        }
        Insert: {
          id?: string
          created_at?: string
          user_name?: string | null
          email?: string | null
          profile_picture?: string | null
          created_by?: string
          grade?: number | null
          school?: string | null
          credits?: number | null
          name?: string | null
          class_level?: number | null
        }
        Update: {
          id?: string
          created_at?: string
          user_name?: string | null
          email?: string | null
          profile_picture?: string | null
          created_by?: string
          grade?: number | null
          school?: string | null
          credits?: number | null
          name?: string | null
          class_level?: number | null
        }
      }
      Questions_Created: {
        Row: {
          id: number
          created_at: string
          grade: string | null
          subject: string | null
          chapter: string | null
          difficulty_level: string | null
          length: string | null
          special_instructions: string[] | null
          test: Json
          created_by: string
        }
        Insert: {
          id?: number
          created_at?: string
          grade?: string | null
          subject?: string | null
          chapter?: string | null
          difficulty_level?: string | null
          length?: string | null
          special_instructions?: string[] | null
          test: Json
          created_by?: string
        }
        Update: {
          id?: number
          created_at?: string
          grade?: string | null
          subject?: string | null
          chapter?: string | null
          difficulty_level?: string | null
          length?: string | null
          special_instructions?: string[] | null
          test?: Json
          created_by?: string
        }
      }
      AnswerSheetImages: {
        Row: {
          id: number
          created_at: string
          image_url: string[] | null
          uploaded_by: string
          for_test: number | null
          feedback: number | null
        }
        Insert: {
          id?: number
          created_at?: string
          image_url?: string[] | null
          uploaded_by?: string
          for_test?: number | null
          feedback?: number | null
        }
        Update: {
          id?: number
          created_at?: string
          image_url?: string[] | null
          uploaded_by?: string
          for_test?: number | null
          feedback?: number | null
        }
      }
      FeedbackTest: {
        Row: {
          id: number
          created_at: string
          given_by: string
          for_test: number | null
          feedback: Json
        }
        Insert: {
          id?: number
          created_at?: string
          given_by?: string
          for_test?: number | null
          feedback: Json
        }
        Update: {
          id?: number
          created_at?: string
          given_by?: string
          for_test?: number | null
          feedback?: Json
        }
      }
      Chapter_contents: {
        Row: {
          id: number
          created_at: string
          grade: string | null
          subject: string | null
          board: string | null
          summary: string | null
          chapter: string | null
        }
        Insert: {
          id?: number
          created_at?: string
          grade?: string | null
          subject?: string | null
          board?: string | null
          summary?: string | null
          chapter?: string | null
        }
        Update: {
          id?: number
          created_at?: string
          grade?: string | null
          subject?: string | null
          board?: string | null
          summary?: string | null
          chapter?: string | null
        }
      }
      Feedback_table_2: {
        Row: {
          id: number
          created_at: string
          feedback_by: string
          name: string | null
          feedback: string | null
        }
        Insert: {
          id?: number
          created_at?: string
          feedback_by?: string
          name?: string | null
          feedback?: string | null
        }
        Update: {
          id?: number
          created_at?: string
          feedback_by?: string
          name?: string | null
          feedback?: string | null
        }
      }
      tests: {
        Row: {
          id: string
          user_id: string
          subject: string
          topic: string
          difficulty: string
          questions: Json
          created_at: string
        }
        Insert: {
          id?: string
          user_id: string
          subject: string
          topic: string
          difficulty: string
          questions: Json
          created_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          subject?: string
          topic?: string
          difficulty?: string
          questions?: Json
          created_at?: string
        }
      }
      evaluations: {
        Row: {
          id: string
          test_id: string
          answer_sheets: string[]
          feedback: Json
          created_at: string
        }
        Insert: {
          id?: string
          test_id: string
          answer_sheets?: string[]
          feedback?: Json
          created_at?: string
        }
        Update: {
          id?: string
          test_id?: string
          answer_sheets?: string[]
          feedback?: Json
          created_at?: string
        }
      }
    }
  }
}
