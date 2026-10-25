import React, { useState } from 'react'
import { Link, Navigate } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { Input } from '../components/ui/Input'
import { Card, CardContent, CardHeader } from '../components/ui/Card'
import { supabase } from '../lib/supabase'
import { supabaseConfigured } from '../lib/supabaseConfig'
import { BYPASS_AUTH, logAuthBypassWarning } from '../lib/auth/devBypass'
import { Sparkles, AlertCircle } from 'lucide-react'

export function ForgotPasswordPage() {
  const [email, setEmail] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [sent, setSent] = useState(false)

  if (BYPASS_AUTH) {
    logAuthBypassWarning('ForgotPasswordPage redirect')
    return <Navigate to="/home" replace />
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    if (!supabaseConfigured) {
      setError('Authentication service is not configured. Please contact support.')
      setLoading(false)
      return
    }

    try {
      const { error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: `${window.location.origin}/reset-password`,
      })

      if (error) {
        setError(error.message)
      } else {
        setSent(true)
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
                Reset your password
              </h2>
              <p className="mt-2 text-sm text-neutral-600">
                Enter the email you signed up with and we will send you a reset link
              </p>
            </div>
          </CardHeader>

          <CardContent>
            {sent ? (
              <div className="text-center space-y-4">
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <p className="text-sm text-green-800 font-medium">Reset link sent</p>
                  <p className="mt-1 text-sm text-green-700">
                    If an account exists for {email}, a password reset link is on its way. Check your inbox and spam folder.
                  </p>
                </div>
                <Link
                  to="/auth"
                  className="inline-flex min-h-11 items-center text-sm text-primary-600 hover:text-primary-700 font-medium transition-colors duration-200"
                >
                  Back to sign in
                </Link>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="space-y-5">
                <Input
                  label="Email Address"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  placeholder="Enter your email"
                  disabled={!supabaseConfigured}
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
                  disabled={!supabaseConfigured}
                >
                  Send reset link
                </Button>

                <div className="text-center">
                  <Link
                    to="/auth"
                    className="inline-flex min-h-11 items-center text-sm text-primary-600 hover:text-primary-700 font-medium transition-colors duration-200"
                  >
                    Back to sign in
                  </Link>
                </div>
              </form>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
