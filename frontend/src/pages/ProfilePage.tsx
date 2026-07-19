import React, { useEffect, useMemo, useState } from 'react'
import {
  ArrowRight,
  CheckCircle2,
  Coins,
  GraduationCap,
  Mail,
  School,
  ShieldCheck,
  User,
} from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { Input } from '../components/ui/Input'
import { Select } from '../components/ui/Select'
import { Header } from '../components/layout/Header'
import { useAuthContext } from '../contexts/AuthContext'
import { useProfile } from '../hooks/useProfile'

function getDisplayName(name?: string | null, email?: string | null) {
  return name?.trim() || email?.split('@')[0] || 'Learner'
}

function getErrorMessage(error: unknown) {
  if (error instanceof Error) return error.message
  if (typeof error === 'object' && error !== null && 'message' in error) {
    return String(error.message)
  }
  return String(error || 'Something went wrong. Please try again.')
}

export function ProfilePage() {
  const navigate = useNavigate()
  const { user } = useAuthContext()
  const { profile, loading, createProfile, updateProfile } = useProfile(user?.id)
  const [name, setName] = useState('')
  const [selectedGrade, setSelectedGrade] = useState('')
  const [school, setSchool] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(false)

  const gradeOptions = useMemo(
    () => Array.from({ length: 12 }, (_, index) => ({ value: index + 1, label: `Class ${index + 1}` })),
    [],
  )
  const displayName = getDisplayName(profile?.name || profile?.user_name || name, user?.email)
  const initials = displayName.charAt(0).toUpperCase()

  useEffect(() => {
    if (!profile) return
    setName(profile.name || profile.user_name || '')
    setSelectedGrade(profile.grade?.toString() || '')
    setSchool(profile.school || '')
  }, [profile])

  const resetForm = () => {
    setName(profile?.name || profile?.user_name || '')
    setSelectedGrade(profile?.grade?.toString() || '')
    setSchool(profile?.school || '')
    setError('')
    setSaved(false)
  }

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    setError('')
    setSaved(false)

    if (!name.trim() || !selectedGrade || !school.trim()) {
      setError('Please complete your name, class, and school.')
      return
    }

    setSubmitting(true)
    try {
      const profileData = {
        name: name.trim(),
        user_name: name.trim(),
        grade: parseInt(selectedGrade, 10),
        school: school.trim(),
        email: user?.email || '',
      }
      const result = profile
        ? await updateProfile(profileData)
        : await createProfile(profileData)

      if (result.error) {
        setError(getErrorMessage(result.error))
        return
      }

      setSaved(true)
    } catch (caught) {
      setError(getErrorMessage(caught))
    } finally {
      setSubmitting(false)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-cream font-sans">
        <Header />
        <main className="mx-auto flex max-w-7xl items-center justify-center px-4 py-24 sm:px-6 lg:px-8">
          <div className="text-center">
            <div className="mx-auto h-10 w-10 animate-spin rounded-full border-2 border-primary-100 border-t-primary-600" />
            <p className="mt-4 text-sm font-semibold text-neutral-600">Loading your profile...</p>
          </div>
        </main>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-cream font-sans text-dark">
      <Header />

      <main className="mx-auto max-w-7xl px-4 pb-16 pt-10 sm:px-6 lg:px-8 lg:pb-24 lg:pt-14">
        <div className="flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.18em] text-primary-700">Account</p>
            <h1 className="mt-2 text-4xl font-bold tracking-tight text-dark sm:text-5xl">Your profile</h1>
            <p className="mt-4 max-w-2xl text-base leading-7 text-neutral-600">
              Keep your learner details current so practice recommendations and
              classroom comparisons stay relevant.
            </p>
          </div>
          {profile && (
            <Button type="button" variant="ghost" onClick={() => navigate('/home')}>
              Back to home
              <ArrowRight className="ml-2 h-4 w-4" aria-hidden="true" />
            </Button>
          )}
        </div>

        <section className="mt-10 grid gap-6 lg:grid-cols-[minmax(0,1.15fr)_minmax(20rem,0.85fr)]">
          <div className="rounded-3xl bg-dark p-6 text-white shadow-xl shadow-dark/10 sm:p-8">
            <div className="flex items-start justify-between gap-5">
              <div className="flex min-w-0 items-center gap-4">
                <div className="flex h-16 w-16 shrink-0 items-center justify-center rounded-2xl bg-primary-500 text-2xl font-bold text-white">
                  {initials}
                </div>
                <div className="min-w-0">
                  <p className="text-xs font-bold uppercase tracking-[0.18em] text-secondary-300">Learner profile</p>
                  <h2 className="mt-2 truncate text-2xl font-bold sm:text-3xl">{displayName}</h2>
                  <p className="mt-1 truncate text-sm text-neutral-300">{user?.email || 'Email not available'}</p>
                </div>
              </div>
              <ShieldCheck className="h-5 w-5 shrink-0 text-primary-300" aria-label="Account protected" />
            </div>

            <div className="mt-8 grid gap-3 sm:grid-cols-2">
              <div className="rounded-2xl bg-white/10 p-4">
                <p className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.12em] text-neutral-300">
                  <GraduationCap className="h-4 w-4 text-secondary-300" aria-hidden="true" /> Class
                </p>
                <p className="mt-2 text-lg font-bold">{profile?.grade ? `Class ${profile.grade}` : 'Not set'}</p>
              </div>
              <div className="rounded-2xl bg-white/10 p-4">
                <p className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.12em] text-neutral-300">
                  <School className="h-4 w-4 text-secondary-300" aria-hidden="true" /> School
                </p>
                <p className="mt-2 truncate text-lg font-bold">{profile?.school || 'Not set'}</p>
              </div>
            </div>
          </div>

          <aside className="rounded-3xl border border-secondary-200 bg-secondary-50 p-6 sm:p-8">
            <div className="flex items-center justify-between gap-4">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-white text-secondary-700 shadow-sm">
                <Coins className="h-5 w-5" aria-hidden="true" />
              </div>
              <span className="rounded-full border border-secondary-200 bg-white px-3 py-1 text-xs font-bold uppercase tracking-[0.12em] text-secondary-800">
                E-coins
              </span>
            </div>
            <p className="mt-7 text-xs font-bold uppercase tracking-[0.18em] text-secondary-800">Current balance</p>
            <p className="mt-2 text-5xl font-bold tracking-tight text-dark">{profile?.credits ?? '—'}</p>
            <p className="mt-4 text-sm leading-6 text-neutral-600">
              Your credits balance is ready for future premium learning features.
            </p>
            <div className="mt-6 flex items-center gap-2 text-sm font-semibold text-secondary-800">
              <CheckCircle2 className="h-4 w-4" aria-hidden="true" /> Practice is available now
            </div>
          </aside>
        </section>

        <section className="mt-8 rounded-3xl border border-neutral-200 bg-white shadow-sm">
          <div className="border-b border-neutral-100 px-6 py-6 sm:px-8">
            <p className="text-xs font-bold uppercase tracking-[0.18em] text-primary-700">Personal details</p>
            <h2 className="mt-2 text-2xl font-bold tracking-tight text-dark">Keep your profile up to date</h2>
            <p className="mt-2 text-sm leading-6 text-neutral-600">
              These details are used to personalize your practice experience.
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6 px-6 py-6 sm:px-8 sm:py-8">
            <div className="grid gap-6 md:grid-cols-2">
              <Input
                label="Full name"
                value={name}
                onChange={(event) => {
                  setName(event.target.value)
                  setSaved(false)
                }}
                required
                placeholder="Enter your full name"
              />
              <Select
                label="Class"
                value={selectedGrade}
                onChange={(event) => {
                  setSelectedGrade(event.target.value)
                  setSaved(false)
                }}
                options={[{ value: '', label: 'Select your class' }, ...gradeOptions]}
                required
              />
            </div>

            <Input
              label="School"
              value={school}
              onChange={(event) => {
                setSchool(event.target.value)
                setSaved(false)
              }}
              required
              placeholder="Enter your school name"
            />

            <div>
              <label className="block text-sm font-medium text-dark">Email address</label>
              <div className="mt-1 flex min-h-11 items-center gap-3 rounded-lg border border-neutral-200 bg-neutral-50 px-3 text-sm text-neutral-600">
                <Mail className="h-4 w-4 text-neutral-400" aria-hidden="true" />
                <span className="truncate">{user?.email || 'Email not available'}</span>
                <span className="ml-auto shrink-0 text-xs font-semibold text-neutral-400">Managed by sign-in</span>
              </div>
            </div>

            {error && (
              <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800" role="alert">
                {error}
              </div>
            )}

            {saved && (
              <div className="flex items-center gap-2 rounded-xl border border-primary-200 bg-primary-50 px-4 py-3 text-sm font-semibold text-primary-800" role="status" aria-live="polite">
                <CheckCircle2 className="h-4 w-4" aria-hidden="true" /> Profile saved successfully.
              </div>
            )}

            <div className="flex flex-col gap-3 border-t border-neutral-100 pt-6 sm:flex-row sm:justify-end">
              {profile && (
                <Button type="button" variant="ghost" onClick={resetForm} disabled={submitting} className="w-full sm:w-auto">
                  Reset changes
                </Button>
              )}
              <Button type="submit" loading={submitting} disabled={!name.trim() || !selectedGrade || !school.trim()} className="w-full sm:w-auto">
                {profile ? 'Save profile' : 'Create profile'}
              </Button>
            </div>
          </form>
        </section>

        <div className="mt-6 flex items-start gap-3 rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-4 text-sm text-neutral-600">
          <User className="mt-0.5 h-4 w-4 shrink-0 text-primary-600" aria-hidden="true" />
          <p>
            Your profile is private to your account. Only the information needed
            for classroom and school leaderboard placement is used there.
          </p>
        </div>
      </main>
    </div>
  )
}
