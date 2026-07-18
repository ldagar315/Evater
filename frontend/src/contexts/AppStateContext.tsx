import React, { createContext, useContext, useState, ReactNode } from 'react'
import { QuestionsCreated, FeedbackTest } from '../types'

export interface LatestPracticeResult {
  test_id: string
  chapter_id: string
  chapter_title: string
  subject: string
  block_score: number
  block_total: number
  percentage: number
  completed_at: string
}

interface AppState {
  last_generated_test: QuestionsCreated | null
  last_generated_feedback: FeedbackTest | null
  latest_practice_result: LatestPracticeResult | null
}

interface AppStateContextType {
  appState: AppState
  setLastGeneratedTest: (test: QuestionsCreated) => void
  setLastGeneratedFeedback: (feedback: FeedbackTest) => void
  setLatestPracticeResult: (result: LatestPracticeResult) => void
  clearAppState: () => void
}

const AppStateContext = createContext<AppStateContextType | undefined>(undefined)

const initialState: AppState = {
  last_generated_test: null,
  last_generated_feedback: null,
  latest_practice_result: null,
}

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [appState, setAppState] = useState<AppState>(() => {
    // Try to load from localStorage on initialization
    try {
      const saved = localStorage.getItem('evater_app_state')
      return saved ? { ...initialState, ...JSON.parse(saved) } : initialState
    } catch {
      return initialState
    }
  })

  const updateState = (changes: Partial<AppState>) => {
    setAppState((currentState) => {
      const newState = { ...currentState, ...changes }
      try {
        localStorage.setItem('evater_app_state', JSON.stringify(newState))
      } catch (error) {
        console.warn('Failed to save app state to localStorage:', error)
      }
      return newState
    })
  }

  const setLastGeneratedTest = (test: QuestionsCreated) => {
    updateState({ last_generated_test: test })
  }

  const setLastGeneratedFeedback = (feedback: FeedbackTest) => {
    updateState({ last_generated_feedback: feedback })
  }

  const setLatestPracticeResult = (result: LatestPracticeResult) => {
    updateState({ latest_practice_result: result })
  }

  const clearAppState = () => {
    setAppState(initialState)
    localStorage.removeItem('evater_app_state')
  }

  return (
    <AppStateContext.Provider value={{
    appState,
    setLastGeneratedTest,
    setLastGeneratedFeedback,
    setLatestPracticeResult,
    clearAppState
  }}>
      {children}
    </AppStateContext.Provider>
  )
}

export function useAppState() {
  const context = useContext(AppStateContext)
  if (context === undefined) {
    throw new Error('useAppState must be used within an AppStateProvider')
  }
  return context
}
