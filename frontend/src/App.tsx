import React, { Suspense } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { ErrorBoundary } from './components/ErrorBoundary'
import { LoadingFallback } from './components/LoadingFallback'
import { Footer } from './components/layout/Footer'
import { AuthProvider, useAuthContext } from './contexts/AuthContext'
import { AppStateProvider } from './contexts/AppStateContext'
import { useProfile } from './hooks/useProfile'
import { LandingPage } from './pages/LandingPage'
import { AuthPage } from './pages/AuthPage'
import { ProfilePage } from './pages/ProfilePage'
import { HomePage } from './pages/HomePage'
import { CreateTestPage } from './pages/CreateTestPage'
import { ViewTestPage } from './pages/ViewTestPage'
import { TakeTestPage } from './pages/TakeTestPage'
import { SubmitFeedbackPage } from './pages/SubmitFeedbackPage'
import { ViewFeedbackPage } from './pages/ViewFeedbackPage'
import { GeneralFeedbackPage } from './pages/GeneralFeedbackPage'
import { PreviousTestsPage } from './pages/PreviousTestsPage'
import { PreviousFeedbacksPage } from './pages/PreviousFeedbacksPage'
import { BlogPage } from './pages/BlogPage'
import { BlogPostPage } from './pages/BlogPostPage'
import { VivaPage } from './pages/VivaPage'
import { AboutPage } from './pages/AboutPage'
import { BYPASS_AUTH } from './lib/auth/devBypass'

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, loading, error } = useAuthContext()
  const { profile, loading: profileLoading } = useProfile(user?.id)

  if (BYPASS_AUTH) {
    // DEV ONLY: skip protected-route checks when auth is bypassed locally
    return <>{children}</>
  }

  if (loading || profileLoading) {
    return <LoadingFallback />
  }

  if (error) {
    return (
      <div className="min-h-screen bg-cream flex items-center justify-center p-4">
        <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-6 text-center">
          <h2 className="text-xl font-bold text-dark mb-2">Authentication Error</h2>
          <p className="text-neutral-600 mb-4">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="bg-primary-500 text-white px-4 py-2 rounded-lg hover:bg-primary-600 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (!user) {
    return <Navigate to="/auth" replace />
  }

  if (!profile) {
    return <Navigate to="/profile" replace />
  }

  return <>{children}</>
}

function AppRoutes() {
  const { user, loading, error } = useAuthContext()

  if (loading) {
    return <LoadingFallback />
  }

  if (error) {
    return (
      <div className="min-h-screen bg-cream flex items-center justify-center p-4">
        <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-6 text-center">
          <h2 className="text-xl font-bold text-dark mb-2">Service Unavailable</h2>
          <p className="text-neutral-600 mb-4">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="bg-primary-500 text-white px-4 py-2 rounded-lg hover:bg-primary-600 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex flex-col bg-cream">
      <div className="flex-1">
        <Routes>
          {/* Public routes */}
          <Route path="/" element={!user ? <LandingPage /> : <Navigate to="/home" replace />} />
          <Route path="/blog" element={<BlogPage />} />
          <Route path="/blog/:slug" element={<BlogPostPage />} />
          <Route path="/about" element={<AboutPage />} />
          
          {!user ? (
            <>
              <Route path="/auth" element={<AuthPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </>
          ) : (
            <>
              <Route path="/profile" element={<ProfilePage />} />
              <Route path="/home" element={
                <ProtectedRoute>
                  <HomePage />
                </ProtectedRoute>
              } />
              <Route path="/create-test" element={
                <ProtectedRoute>
                  <CreateTestPage />
                </ProtectedRoute>
              } />
              <Route path="/previous-tests" element={
                <ProtectedRoute>
                  <PreviousTestsPage />
                </ProtectedRoute>
              } />
              <Route path="/previous-feedbacks" element={
                <ProtectedRoute>
                  <PreviousFeedbacksPage />
                </ProtectedRoute>
              } />
              <Route path="/view-test/:testId?" element={
                <ProtectedRoute>
                  <ViewTestPage />
                </ProtectedRoute>
              } />
              <Route path="/take-test/:testId" element={
                <ProtectedRoute>
                  <TakeTestPage />
                </ProtectedRoute>
              } />
              <Route path="/submit-feedback/:testId?" element={
                <ProtectedRoute>
                  <SubmitFeedbackPage />
                </ProtectedRoute>
              } />
              <Route path="/view-feedback/:evaluationId?" element={
                <ProtectedRoute>
                  <ViewFeedbackPage />
                </ProtectedRoute>
              } />
              <Route path="/general-feedback" element={
                <ProtectedRoute>
                  <GeneralFeedbackPage />
                </ProtectedRoute>
              } />
              <Route path="/viva" element={
                <ProtectedRoute>
                  <VivaPage />
                </ProtectedRoute>
              } />
              <Route path="*" element={<Navigate to="/home" replace />} />
            </>
          )}
        </Routes>
      </div>
      {/* Only show footer on authenticated pages */}
      {user && <Footer />}
    </div>
  )
}

function App() {
  return (
    <ErrorBoundary>
      <AuthProvider>
        <AppStateProvider>
          <Router>
            <Suspense fallback={<LoadingFallback />}>
              <AppRoutes />
            </Suspense>
          </Router>
        </AppStateProvider>
      </AuthProvider>
    </ErrorBoundary>
  )
}

export default App
