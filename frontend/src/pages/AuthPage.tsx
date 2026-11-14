import { authReturn } from '../lib/practiceTarget'
import { Navigate, useNavigate, useLocation } from 'react-router-dom'
import { AuthForm } from '../components/auth/AuthForm'
import { BYPASS_AUTH, logAuthBypassWarning } from '../lib/auth/devBypass'

export function AuthPage() {
  const navigate = useNavigate()
  const location = useLocation()

  if (BYPASS_AUTH) {
    logAuthBypassWarning('AuthPage redirect')
    return <Navigate to="/home" replace />
  }

  const handleAuthSuccess = () => {
    const next = authReturn(location.search)
    navigate(next === '/home' ? '/profile' : next)
  }

  return <AuthForm onSuccess={handleAuthSuccess} />
}
