import React, { useEffect, useState } from 'react'
import { Link, Navigate } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { Input } from '../components/ui/Input'
import { Card, CardContent, CardHeader } from '../components/ui/Card'
import { supabase } from '../lib/supabase'
import { supabaseConfigured } from '../lib/supabaseConfig'
import { BYPASS_AUTH, logAuthBypassWarning } from '../lib/auth/devBypass'
import { Sparkles, AlertCircle } from 'lucide-react'

export function ResetPasswordPage() {
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [done, setDone] = useState(false)
  const [sessionReady, setSessionReady] = useState<boolean | null>(null)

  useEffect(() => {
    if (!supabaseConfigured) {
      setSessionReady(false)
      return
    }

    let active = true
    const { data: { subscription } } = supabase.auth.onAuthStateChange((event, session) => {
      if (!active) return
      if (event === 'PASSWORD_RECOVERY' || session) {
        setSessionReady(true)
      }
    })

    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!active) return
      setSessionReady((current) => current ?? !!session)
    })

    return () => {
      active = false
      subscription.unsubscribe()
    }
  }, [])

  if (BYPASS_AUTH) {
    logAuthBypassWarning('ResetPasswordPage redirect')
    return <Navigate to="/home" replace />
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (password.length < 6) {
      setError('Password must be at least 6 characters')
      return
    }

    if (password !== confirmPassword) {
      setError('Passwords do not match')
      return
    }

    setLoading(true)

    try {
      const { error } = await supabase.auth.updateUser({ password })

      if (error) {
        setError(error.message)
      } else {
        await supabase.auth.signOut()
        setDone(true)
      }
    } catch (err) {
      setError('An unexpected error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-cream py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full">
        {/* Logo and Branding */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-24 h-16 mb-4">
            <img
              src="/Evater_logo_2.png"
              alt="Evater Logo"
              className="h-16 w-auto object-contain"
            />
          </div>
          <h1 className="text-3xl font-bold text-dark mb-2">
            Evater
          </h1>
          <p className="text-neutral-600 font-medium">Next Gen Learning</p>
        </div>

        <Card className="shadow-lg">
          <CardHeader>
            <div className="text-center">
              <h2 className="text-2xl font-bold text-dark flex items-center justify-center">
                <Sparkles className="h-6 w-6 text-primary-500 mr-2" />
                Choose a new password
              </h2>
            </div>
          </CardHeader>

          <CardContent>
            {sessionReady === null ? (
              <p className="text-center text-sm text-neutral-600">Checking your reset link...</p>
            ) : done ? (
              <div className="text-center space-y-4">
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <p className="text-sm text-green-800 font-medium">Password updated</p>
                  <p className="mt-1 text-sm text-green-700">
                    Your password has been changed. Sign in with your new password.
                  </p>
                </div>
                <Link
                  to="/auth"
                  className="inline-flex min-h-11 items-center text-sm text-primary-600 hover:text-primary-700 font-medium transition-colors duration-200"
                >
                  Go to sign in
                </Link>
              </div>
            ) : !sessionReady ? (
              <div className="text-center space-y-4">
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <div className="flex items-start">
                    <AlertCircle className="h-5 w-5 text-red-600 mr-3 mt-0.5 flex-shrink-0" />
                    <p className="text-sm text-red-700 font-medium">
                      This reset link is invalid or has expired. Please request a new one.
                    </p>
                  </div>
                </div>
                <Link
                  to="/forgot-password"
                  className="inline-flex min-h-11 items-center text-sm text-primary-600 hover:text-primary-700 font-medium transition-colors duration-200"
                >
                  Request a new reset link
                </Link>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="space-y-5">
                <Input
                  label="New Password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  placeholder="Enter a new password"
                />

                <Input
                  label="Confirm New Password"
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  required
                  placeholder="Confirm your new password"
                />

                {error && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                    <div className="flex items-start">
                      <AlertCircle className="h-5 w-5 text-red-600 mr-3 mt-0.5 flex-shrink-0" />
                      <p className="text-sm text-red-700 font-medium">{error}</p>
                    </div>
                  </div>
                )}

                <Button
                  type="submit"
                  className="w-full py-3 text-base font-medium"
                  loading={loading}
                >
                  Update password
                </Button>
              </form>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
