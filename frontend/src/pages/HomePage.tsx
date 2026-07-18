import React from 'react'
import {
  ArrowRight,
  ArrowUpRight,
  BarChart3,
  Brain,
  Coins,
  FileText,
  GraduationCap,
  Lock,
  School,
  Sparkles,
  Target,
  Trophy,
} from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { Header } from '../components/layout/Header'
import { useAppState } from '../contexts/AppStateContext'
import { useAuthContext } from '../contexts/AuthContext'
import { useProfile } from '../hooks/useProfile'
import { BYPASS_AUTH } from '../lib/auth/devBypass'

function formatDate(date?: string) {
  if (!date) return 'Recently'

  const parsedDate = new Date(date)
  if (Number.isNaN(parsedDate.getTime())) return 'Recently'

  return parsedDate.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
  })
}

export function HomePage() {
  const navigate = useNavigate()
  const { appState } = useAppState()
  const { user } = useAuthContext()
  const { profile } = useProfile(user?.id)

  const firstName =
    profile?.user_name?.trim().split(/\s+/)[0] ||
    profile?.name?.trim().split(/\s+/)[0] ||
    'there'
  const latestPractice = appState.latest_practice_result

  const accountStats = [
    {
      label: 'Grade',
      value: profile?.grade ? `Class ${profile.grade}` : 'Not set',
      icon: GraduationCap,
      iconClass: 'bg-primary-50 text-primary-600',
    },
    {
      label: 'School',
      value: profile?.school || 'Not set',
      icon: School,
      iconClass: 'bg-secondary-50 text-secondary-700',
    },
    {
      label: 'Credits',
      value: profile?.credits ?? '—',
      icon: Coins,
      iconClass: 'bg-yellow-50 text-yellow-700',
    },
  ]

  const activePaths = [
    {
      eyebrow: 'Practice',
      title: 'Practice',
      description: 'Choose a subject and chapter, work through focused questions, and see what to revisit.',
      cta: 'Start practicing',
      icon: Target,
      path: '/practice',
      cardClass: 'border-secondary-200 bg-secondary-50/70 hover:border-secondary-400 hover:bg-secondary-50',
      iconClass: 'bg-secondary-500 text-dark',
    },
    {
      eyebrow: 'Compete',
      title: 'Class leaderboard',
      description: 'See how your practice is building momentum with students in your classroom and school.',
      cta: 'View leaderboard',
      icon: BarChart3,
      path: '/leaderboard',
      cardClass: 'border-primary-200 bg-primary-50/70 hover:border-primary-400 hover:bg-primary-50',
      iconClass: 'bg-primary-500 text-white',
    },
  ]

  const paidFeatures = [
    {
      title: 'Pre-made test papers',
      description: 'Take a complete paper, then receive AI-based evaluation and feedback.',
      icon: FileText,
    },
    {
      title: 'AI Viva session',
      description: 'Prepare out loud with an interactive oral examination built around your learning.',
      icon: Brain,
    },
  ]

  return (
    <div className="min-h-screen bg-cream font-sans text-dark">
      <Header />

      {BYPASS_AUTH && (
        <div className="mx-auto max-w-7xl px-4 pt-4 sm:px-6 lg:px-8">
          <div
            className="flex flex-col gap-1 rounded-xl border border-yellow-300 bg-yellow-50 px-4 py-3 text-sm text-yellow-900 sm:flex-row sm:items-center sm:justify-between"
            role="alert"
          >
            <p className="font-semibold">Auth bypass enabled for local development</p>
            <p className="text-yellow-800">Remove VITE_BYPASS_AUTH before committing or deploying.</p>
          </div>
        </div>
      )}

      <main className="mx-auto max-w-7xl px-4 pb-16 pt-10 sm:px-6 lg:px-8 lg:pb-24 lg:pt-14">
        <section className="grid gap-8 lg:grid-cols-[minmax(0,1.1fr)_minmax(20rem,0.9fr)] lg:items-stretch">
          <div className="flex flex-col justify-center">
            <p className="mb-4 inline-flex w-fit items-center gap-2 rounded-full border border-primary-200 bg-primary-50 px-3 py-1 text-xs font-bold uppercase tracking-[0.18em] text-primary-700">
              <Sparkles className="h-3.5 w-3.5" aria-hidden="true" />
              Student workspace
            </p>
            <h1 className="max-w-3xl text-4xl font-bold leading-[1.08] tracking-tight text-dark sm:text-5xl lg:text-6xl">
              Good to see you, <span className="text-primary-600">{firstName}</span>.
            </h1>
            <p className="mt-5 max-w-2xl text-base leading-7 text-neutral-600 sm:text-lg">
              Practice one chapter at a time, understand every mistake, and build
              the confidence to keep going.
            </p>
            <div className="mt-7 flex flex-col gap-3 sm:flex-row">
              <Button onClick={() => navigate('/practice')} size="lg" className="w-full sm:w-auto">
                <Target className="mr-2 h-5 w-5" aria-hidden="true" />
                Start practice
              </Button>
              <Button
                onClick={() => navigate('/leaderboard')}
                size="lg"
                variant="outline"
                className="w-full sm:w-auto"
              >
                <Trophy className="mr-2 h-5 w-5" aria-hidden="true" />
                View leaderboard
              </Button>
            </div>
            <p className="mt-5 text-sm font-semibold text-neutral-500">
              10 questions · 10 minutes · Clear next steps
            </p>
          </div>

          <aside className="flex flex-col justify-between rounded-3xl bg-dark p-6 text-white shadow-xl shadow-dark/10 sm:p-8">
            <div>
              <div className="flex items-center justify-between gap-4">
                <p className="text-xs font-bold uppercase tracking-[0.18em] text-secondary-300">
                  Your next session
                </p>
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/10 text-secondary-300">
                  <Target className="h-5 w-5" aria-hidden="true" />
                </div>
              </div>
              <h2 className="mt-8 max-w-sm text-2xl font-bold leading-tight sm:text-3xl">
                Keep your learning streak simple.
              </h2>
              <p className="mt-4 max-w-sm text-sm leading-6 text-neutral-300">
                Choose a chapter, answer one question at a time, and finish with a
                focused review of what to practise next.
              </p>
            </div>
            <button
              type="button"
              onClick={() => navigate('/practice')}
              className="mt-8 inline-flex min-h-11 w-fit items-center gap-2 rounded-xl bg-secondary-500 px-4 py-2.5 text-sm font-bold text-dark transition-colors hover:bg-secondary-400 focus:outline-none focus:ring-2 focus:ring-secondary-300 focus:ring-offset-2 focus:ring-offset-dark"
            >
              Choose a chapter
              <ArrowUpRight className="h-4 w-4" aria-hidden="true" />
            </button>
          </aside>
        </section>

        <section aria-label="Account snapshot" className="mt-8 grid grid-cols-3 gap-2 sm:mt-10 sm:gap-3">
          {accountStats.map((stat) => {
            const Icon = stat.icon
            return (
              <div
                key={stat.label}
                className="flex min-h-[6.75rem] min-w-0 flex-col items-start gap-3 rounded-2xl border border-neutral-200 bg-white p-3 shadow-sm sm:min-h-[7.5rem] sm:flex-row sm:items-center sm:gap-4 sm:p-6"
              >
                <div className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl sm:h-11 sm:w-11 ${stat.iconClass}`}>
                  <Icon className="h-5 w-5" aria-hidden="true" />
                </div>
                <div className="min-w-0">
                  <p className="text-[0.6rem] font-bold uppercase tracking-[0.12em] text-neutral-500 sm:text-xs sm:tracking-[0.16em]">
                    {stat.label}
                  </p>
                  <p className="mt-1 truncate text-sm font-bold text-dark sm:text-xl" title={String(stat.value)}>
                    {stat.value}
                  </p>
                </div>
              </div>
            )
          })}
        </section>

        <section className="mt-16">
          <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="text-xs font-bold uppercase tracking-[0.18em] text-primary-700">Available now</p>
              <h2 className="mt-2 text-3xl font-bold tracking-tight text-dark">Choose your next move</h2>
            </div>
            <p className="text-sm font-medium text-neutral-500">Practice is the place to start</p>
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            {activePaths.map((path) => {
              const Icon = path.icon
              return (
                <button
                  key={path.path}
                  type="button"
                  onClick={() => navigate(path.path)}
                  className={`group flex min-h-[16rem] w-full flex-col rounded-3xl border p-6 text-left shadow-sm transition-all duration-200 hover:-translate-y-1 hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 sm:p-7 ${path.cardClass}`}
                >
                  <span className={`flex h-12 w-12 items-center justify-center rounded-2xl ${path.iconClass}`}>
                    <Icon className="h-6 w-6" aria-hidden="true" />
                  </span>
                  <span className="mt-8 flex items-start justify-between gap-4">
                    <span>
                      <span className="block text-xs font-bold uppercase tracking-[0.16em] text-neutral-500">{path.eyebrow}</span>
                      <span className="mt-2 block text-2xl font-bold leading-tight text-dark">{path.title}</span>
                    </span>
                    <ArrowUpRight
                      className="h-5 w-5 shrink-0 text-neutral-400 transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5 group-hover:text-dark"
                      aria-hidden="true"
                    />
                  </span>
                  <span className="mt-3 block max-w-xl text-sm leading-6 text-neutral-600">{path.description}</span>
                  <span className="mt-auto flex items-center gap-2 pt-6 text-sm font-bold text-dark">
                    {path.cta}
                    <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" aria-hidden="true" />
                  </span>
                </button>
              )
            })}
          </div>
        </section>

        <section className="mt-14">
          <div className="mb-6">
            <p className="text-xs font-bold uppercase tracking-[0.18em] text-neutral-500">Coming soon</p>
            <h2 className="mt-2 text-2xl font-bold tracking-tight text-dark sm:text-3xl">More ways to prepare</h2>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-neutral-600">
              These paid-plan features are in the works. They will stay here so
              you know what is coming, without getting in the way of practice.
            </p>
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            {paidFeatures.map((feature) => {
              const Icon = feature.icon
              return (
                <div
                  key={feature.title}
                  aria-disabled="true"
                  className="flex min-h-[10.5rem] cursor-not-allowed items-start gap-4 rounded-3xl border border-neutral-200 bg-neutral-100 p-6 text-neutral-500 sm:p-7"
                >
                  <span className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-neutral-200 text-neutral-400">
                    <Icon className="h-6 w-6" aria-hidden="true" />
                  </span>
                  <span className="min-w-0">
                    <span className="flex flex-wrap items-center gap-2">
                      <span className="rounded-full bg-white px-2.5 py-1 text-[10px] font-bold uppercase tracking-[0.12em] text-neutral-500">Paid plan</span>
                      <span className="rounded-full border border-neutral-300 px-2.5 py-1 text-[10px] font-bold uppercase tracking-[0.12em] text-neutral-400">Coming soon</span>
                    </span>
                    <span className="mt-3 flex items-center gap-2 text-xl font-bold text-neutral-600">
                      {feature.title}
                      <Lock className="h-4 w-4 text-neutral-400" aria-hidden="true" />
                    </span>
                    <span className="mt-2 block text-sm leading-6 text-neutral-500">{feature.description}</span>
                  </span>
                </div>
              )
            })}
          </div>
        </section>

        <section className="mt-16" aria-labelledby="workspace-heading">
          <div className="rounded-3xl border border-neutral-200 bg-white p-6 shadow-sm sm:p-8">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <p className="text-xs font-bold uppercase tracking-[0.18em] text-primary-700">Your workspace</p>
                <h2 id="workspace-heading" className="mt-2 text-2xl font-bold tracking-tight text-dark sm:text-3xl">
                  Latest practice result
                </h2>
              </div>
              <span className="text-sm font-medium text-neutral-500">One result, one clear next step</span>
            </div>

            <div className="mt-6 border-y border-neutral-100 py-6">
              {latestPractice ? (
                <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
                  <div className="flex min-w-0 items-start gap-4">
                    <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-primary-50 text-primary-700">
                      <Target className="h-6 w-6" aria-hidden="true" />
                    </div>
                    <div className="min-w-0">
                      <p className="text-xs font-bold uppercase tracking-[0.16em] text-neutral-500">{latestPractice.subject}</p>
                      <p className="mt-1 truncate text-lg font-bold text-dark">{latestPractice.chapter_title}</p>
                      <p className="mt-1 text-sm text-neutral-500">Completed {formatDate(latestPractice.completed_at)}</p>
                    </div>
                  </div>
                  <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
                    <div className="rounded-2xl bg-primary-50 px-5 py-3 sm:min-w-32 sm:text-center">
                      <p className="text-2xl font-bold text-primary-700">{Math.round(latestPractice.percentage)}%</p>
                      <p className="mt-1 text-xs font-semibold text-primary-700">
                        {latestPractice.block_score}/{latestPractice.block_total} correct
                      </p>
                    </div>
                    <Button type="button" onClick={() => navigate('/practice')} variant="outline">
                      Practice again
                      <ArrowRight className="ml-2 h-4 w-4" aria-hidden="true" />
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col gap-5 sm:flex-row sm:items-center sm:justify-between">
                  <div className="flex items-start gap-4">
                    <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-neutral-100 text-neutral-400">
                      <Target className="h-6 w-6" aria-hidden="true" />
                    </div>
                    <div>
                      <p className="text-lg font-bold text-dark">Your first result will appear here.</p>
                      <p className="mt-1 max-w-xl text-sm leading-6 text-neutral-500">
                        Start a short practice session to see your score and the chapter you worked on.
                      </p>
                    </div>
                  </div>
                  <Button type="button" onClick={() => navigate('/practice')}>
                    Start first practice
                    <ArrowRight className="ml-2 h-4 w-4" aria-hidden="true" />
                  </Button>
                </div>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}
