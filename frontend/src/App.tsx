import React, { Suspense, lazy } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { ErrorBoundary } from './components/ErrorBoundary'
import { LoadingFallback } from './components/LoadingFallback'
import { Footer } from './components/layout/Footer'
import { AuthProvider, useAuthContext } from './contexts/AuthContext'
import { AppStateProvider } from './contexts/AppStateContext'
import { useProfile } from './hooks/useProfile'
import { BYPASS_AUTH } from './lib/auth/devBypass'

const LandingPage = lazy(() => import('./pages/LandingPage').then((m) => ({ default: m.LandingPage })))
const AuthPage = lazy(() => import('./pages/AuthPage').then((m) => ({ default: m.AuthPage })))
const ProfilePage = lazy(() => import('./pages/ProfilePage').then((m) => ({ default: m.ProfilePage })))
const HomePage = lazy(() => import('./pages/HomePage').then((m) => ({ default: m.HomePage })))
const CreateTestPage = lazy(() => import('./pages/CreateTestPage').then((m) => ({ default: m.CreateTestPage })))
const ViewTestPage = lazy(() => import('./pages/ViewTestPage').then((m) => ({ default: m.ViewTestPage })))
const TakeTestPage = lazy(() => import('./pages/TakeTestPage').then((m) => ({ default: m.TakeTestPage })))
const SubmitFeedbackPage = lazy(() => import('./pages/SubmitFeedbackPage').then((m) => ({ default: m.SubmitFeedbackPage })))
const ViewFeedbackPage = lazy(() => import('./pages/ViewFeedbackPage').then((m) => ({ default: m.ViewFeedbackPage })))
const GeneralFeedbackPage = lazy(() => import('./pages/GeneralFeedbackPage').then((m) => ({ default: m.GeneralFeedbackPage })))
const PreviousTestsPage = lazy(() => import('./pages/PreviousTestsPage').then((m) => ({ default: m.PreviousTestsPage })))
const PreviousFeedbacksPage = lazy(() => import('./pages/PreviousFeedbacksPage').then((m) => ({ default: m.PreviousFeedbacksPage })))
const BlogPage = lazy(() => import('./pages/BlogPage').then((m) => ({ default: m.BlogPage })))
const BlogPostPage = lazy(() => import('./pages/BlogPostPage').then((m) => ({ default: m.BlogPostPage })))
const VivaPage = lazy(() => import('./pages/VivaPage').then((m) => ({ default: m.VivaPage })))
const AboutPage = lazy(() => import('./pages/AboutPage').then((m) => ({ default: m.AboutPage })))
const QuestionBankPracticePage = lazy(() => import('./pages/QuestionBankPracticePage').then((m) => ({ default: m.QuestionBankPracticePage })))
const LeaderboardPage = lazy(() => import('./pages/LeaderboardPage').then((m) => ({ default: m.LeaderboardPage })))
const LeagueInfoPage = lazy(() => import('./pages/LeagueInfoPage').then((m) => ({ default: m.LeagueInfoPage })))
const QuestionTypePreviewPage = lazy(() => import('./pages/QuestionTypePreviewPage').then((m) => ({ default: m.QuestionTypePreviewPage })))

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
            className="min-h-11 rounded-lg bg-primary-500 px-4 py-2 text-white transition-colors hover:bg-primary-600"
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
  const location = useLocation()
  const isPublicRoute = location.pathname === '/' || location.pathname === '/about' || location.pathname === '/auth' || location.pathname.startsWith('/blog')

  // Public pages should remain useful when auth is slow or unavailable. This is
  // especially important for the blog, which must be crawlable without a session.
  if (loading && !isPublicRoute) {
    return <LoadingFallback />
  }

  if (error && !isPublicRoute) {
    return (
      <div className="min-h-screen bg-cream flex items-center justify-center p-4">
        <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-6 text-center">
          <h2 className="text-xl font-bold text-dark mb-2">Service Unavailable</h2>
          <p className="text-neutral-600 mb-4">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="min-h-11 rounded-lg bg-primary-500 px-4 py-2 text-white transition-colors hover:bg-primary-600"
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
          <Route path="/general-feedback" element={
            <ProtectedRoute>
              <GeneralFeedbackPage />
            </ProtectedRoute>
          } />
          
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
              <Route path="/practice" element={
                <ProtectedRoute>
                  <QuestionBankPracticePage />
                </ProtectedRoute>
              } />
              <Route path="/question-preview" element={
                <ProtectedRoute>
                  <QuestionTypePreviewPage />
                </ProtectedRoute>
              } />
              <Route path="/leaderboard" element={
                <ProtectedRoute>
                  <LeaderboardPage />
                </ProtectedRoute>
              } />
              <Route path="/leagues" element={
                <ProtectedRoute>
                  <LeagueInfoPage />
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

