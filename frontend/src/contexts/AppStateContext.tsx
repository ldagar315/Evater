import React, { createContext, useContext, useState, ReactNode } from 'react'
import { QuestionsCreated, FeedbackTest } from '../types'

interface AppState {
  last_generated_test: QuestionsCreated | null
  last_generated_feedback: FeedbackTest | null
}

interface AppStateContextType {
  appState: AppState
  setLastGeneratedTest: (test: QuestionsCreated) => void
  setLastGeneratedFeedback: (feedback: FeedbackTest) => void
  clearAppState: () => void
}

const AppStateContext = createContext<AppStateContextType | undefined>(undefined)

const initialState: AppState = {
  last_generated_test: null,
  last_generated_feedback: null
}

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [appState, setAppState] = useState<AppState>(() => {
    // Try to load from localStorage on initialization
    try {
      const saved = localStorage.getItem('evater_app_state')
      return saved ? JSON.parse(saved) : initialState
    } catch {
      return initialState
    }
  })

  const updateState = (newState: AppState) => {
    setAppState(newState)
    // Persist to localStorage
    try {
      localStorage.setItem('evater_app_state', JSON.stringify(newState))
    } catch (error) {
      console.warn('Failed to save app state to localStorage:', error)
    }
  }

  const setLastGeneratedTest = (test: QuestionsCreated) => {
    const newState = { ...appState, last_generated_test: test }
    updateState(newState)
  }

  const setLastGeneratedFeedback = (feedback: FeedbackTest) => {
    const newState = { ...appState, last_generated_feedback: feedback }
    updateState(newState)
  }

  const clearAppState = () => {
    updateState(initialState)
    localStorage.removeItem('evater_app_state')
  }

  return (
    <AppStateContext.Provider value={{
      appState,
      setLastGeneratedTest,
      setLastGeneratedFeedback,
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