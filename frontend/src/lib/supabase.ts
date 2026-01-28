import { createClient } from '@supabase/supabase-js'

// Debug environment variables
console.log('Environment check:', {
  NODE_ENV: import.meta.env.MODE,
  SUPABASE_URL: import.meta.env.VITE_SUPABASE_URL ? 'Set' : 'Missing',
  SUPABASE_ANON_KEY: import.meta.env.VITE_SUPABASE_ANON_KEY ? 'Set' : 'Missing'
})

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

let supabase: any

if (!supabaseUrl || !supabaseAnonKey) {
  console.warn('Missing Supabase environment variables - using fallback mode')
  
  // Create a more robust fallback client
  const createFallbackClient = () => {
    return {
      auth: {
        getSession: async () => {
          console.warn('Supabase not configured - returning null session')
          return { data: { session: null }, error: null }
        },
        onAuthStateChange: (callback: any) => {
          console.warn('Supabase not configured - auth state change listener disabled')
          // Call callback immediately with null session
          setTimeout(() => callback('SIGNED_OUT', null), 0)
          return { 
            data: { 
              subscription: { 
                unsubscribe: () => console.log('Fallback auth listener unsubscribed') 
              } 
            } 
          }
        },
        signUp: async () => {
          return { data: null, error: { message: 'Supabase not configured' } }
        },
        signInWithPassword: async () => {
          return { data: null, error: { message: 'Supabase not configured' } }
        },
        signInWithOAuth: async () => {
          return { data: null, error: { message: 'Supabase not configured' } }
        },
        signOut: async () => {
          return { error: null }
        }
      },
      from: () => ({
        select: () => ({
          eq: () => ({
            single: async () => ({ data: null, error: { message: 'Supabase not configured' } })
          }),
          order: () => ({
            limit: () => ({
              single: async () => ({ data: null, error: { message: 'Supabase not configured' } })
            })
          })
        }),
        insert: () => ({
          select: () => ({
            single: async () => ({ data: null, error: { message: 'Supabase not configured' } })
          })
        }),
        update: () => ({
          eq: () => ({
            select: () => ({
              single: async () => ({ data: null, error: { message: 'Supabase not configured' } })
            })
          })
        })
      }),
      storage: {
        from: () => ({
          upload: async () => ({ data: null, error: { message: 'Supabase not configured' } }),
          getPublicUrl: () => ({ data: { publicUrl: '' } })
        })
      }
    }
  }
  
  supabase = createFallbackClient()
} else {
  try {
    supabase = createClient(supabaseUrl, supabaseAnonKey)
    console.log('Supabase client created successfully')
  } catch (error) {
    console.error('Failed to create Supabase client:', error)
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