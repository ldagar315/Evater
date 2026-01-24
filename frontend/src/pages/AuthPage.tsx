import React from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { AuthForm } from '../components/auth/AuthForm'
import { BYPASS_AUTH, logAuthBypassWarning } from '../lib/auth/devBypass'

export function AuthPage() {
  const navigate = useNavigate()

  if (BYPASS_AUTH) {
    logAuthBypassWarning('AuthPage redirect')
    return <Navigate to="/home" replace />
  }

  const handleAuthSuccess = () => {
    navigate('/profile')
  }

  return <AuthForm onSuccess={handleAuthSuccess} />
}
