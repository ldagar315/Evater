import React, { useEffect, useState } from 'react'
import { ArrowLeft, Crown, Medal, RefreshCw, School, Target, Trophy, Users } from 'lucide-react'
import { Header } from '../components/layout/Header'
import { Button } from '../components/ui/Button'
import {
  getLeaderboard,
  LeaderboardEntry,
  LeaderboardPeriod,
  LeaderboardResponse,
  LeaderboardScope,
} from '../lib/api'
import { useNavigate } from 'react-router-dom'

function rankIcon(rank: number) {
  if (rank === 1) return Crown
  if (rank === 2) return Medal
  if (rank === 3) return Trophy
  return Target
}

function rankLabel(entry: LeaderboardEntry) {
  if (entry.is_current_user) return 'You'
  return `${entry.completed_tests} ${entry.completed_tests === 1 ? 'test' : 'tests'}`
}

export function LeaderboardPage() {
  const navigate = useNavigate()
  const [scope, setScope] = useState<LeaderboardScope>('classroom')
  const [period, setPeriod] = useState<LeaderboardPeriod>('weekly')
  const [leaderboard, setLeaderboard] = useState<LeaderboardResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [reloadToken, setReloadToken] = useState(0)

  useEffect(() => {
    let active = true
    setLoading(true)
    setError(null)

    getLeaderboard(scope, period)
      .then((result) => {
        if (active) setLeaderboard(result)
      })
      .catch((caught) => {
        if (active) {
          setLeaderboard(null)
          setError(caught instanceof Error ? caught.message : 'Could not load the leaderboard.')
        }
      })
      .finally(() => {
        if (active) setLoading(false)
      })

    return () => {
      active = false
    }
  }, [period, reloadToken, scope])

  return (
    <div className="min-h-screen bg-cream font-sans text-dark">
      <Header />
      <main className="mx-auto max-w-5xl px-4 pb-16 pt-8 sm:px-6 lg:px-8 lg:pb-24 lg:pt-12">
        <button
          type="button"
          onClick={() => navigate('/practice')}
          className="inline-flex min-h-11 items-center gap-2 rounded-lg px-2 text-sm font-semibold text-neutral-600 transition-colors hover:bg-white hover:text-dark focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
        >
          <ArrowLeft className="h-4 w-4" aria-hidden="true" />
          Back to practice
        </button>

        <section className="mt-7 flex flex-col gap-7 rounded-3xl border border-primary-100 bg-white p-6 shadow-sm sm:p-8 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-2xl">
            <p className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-[0.18em] text-primary-700">
              <Trophy className="h-4 w-4" aria-hidden="true" />
              Peer progress
            </p>
            <h1 className="mt-3 text-3xl font-bold leading-tight tracking-tight text-dark sm:text-4xl">
              Learn alongside your community.
            </h1>
            <p className="mt-3 text-base leading-7 text-neutral-600">
              See how your practice is building momentum with learners in your
              {scope === 'classroom' ? ' classroom' : ' school'}.
            </p>
          </div>
          <div className="flex h-16 w-16 shrink-0 items-center justify-center rounded-2xl bg-secondary-50 text-secondary-700">
            <Users className="h-8 w-8" aria-hidden="true" />
          </div>
        </section>

        <section className="mt-6 flex flex-col gap-4 rounded-2xl border border-neutral-200 bg-white p-4 shadow-sm sm:flex-row sm:items-center sm:justify-between sm:p-5">
          <div className="flex flex-wrap gap-2" role="group" aria-label="Leaderboard scope">
            {([
              ['classroom', 'My classroom', Users],
              ['school', 'My school', School],
            ] as const).map(([value, label, Icon]) => (
              <button
                key={value}
                type="button"
                aria-pressed={scope === value}
                onClick={() => setScope(value)}
                className={`inline-flex min-h-11 items-center gap-2 rounded-xl px-4 text-sm font-bold transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 ${scope === value ? 'bg-dark text-white' : 'text-neutral-600 hover:bg-neutral-100'}`}
              >
                <Icon className="h-4 w-4" aria-hidden="true" />
                {label}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-2 text-sm text-neutral-500">
            <span className="font-semibold">Period</span>
            {([
              ['weekly', 'Last 7 days'],
              ['all_time', 'All time'],
            ] as const).map(([value, label]) => (
              <button
                key={value}
                type="button"
                aria-pressed={period === value}
                onClick={() => setPeriod(value)}
                className={`min-h-11 rounded-lg px-3 font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 ${period === value ? 'bg-primary-50 text-primary-700' : 'hover:bg-neutral-100'}`}
              >
                {label}
              </button>
            ))}
          </div>
        </section>

        {error && (
          <div className="mt-6 flex items-start gap-3 rounded-2xl border border-red-200 bg-red-50 p-4 text-red-800" role="alert">
            <Target className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
            <div className="min-w-0">
              <p className="font-bold">Leaderboard unavailable</p>
              <p className="mt-1 text-sm">{error}</p>
            </div>
            <button
              type="button"
              onClick={() => setReloadToken((current) => current + 1)}
              className="ml-auto inline-flex min-h-11 shrink-0 items-center gap-2 rounded-lg px-3 text-sm font-bold text-red-800 hover:bg-red-100 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2"
            >
              <RefreshCw className="h-4 w-4" aria-hidden="true" />
              Retry
            </button>
          </div>
        )}

        {loading ? (
          <section className="mt-6 space-y-3" aria-label="Loading leaderboard">
            {[1, 2, 3, 4].map((item) => (
              <div key={item} className="h-20 animate-pulse rounded-2xl border border-neutral-100 bg-white" />
            ))}
          </section>
        ) : leaderboard && !leaderboard.scope_available ? (
          <section className="mt-6 rounded-3xl border border-dashed border-neutral-300 bg-white p-8 text-center sm:p-12">
            <Users className="mx-auto h-10 w-10 text-primary-500" aria-hidden="true" />
            <h2 className="mt-4 text-2xl font-bold text-dark">Your community is waiting</h2>
            <p className="mx-auto mt-2 max-w-lg leading-6 text-neutral-600">
              {leaderboard.membership_message || 'Ask your school admin to assign your account to a classroom.'}
            </p>
            <Button type="button" variant="outline" onClick={() => navigate('/practice')} className="mt-6">
              Continue practicing
            </Button>
          </section>
        ) : leaderboard ? (
          <section className="mt-6 rounded-3xl border border-neutral-200 bg-white p-5 shadow-sm sm:p-8">
            <div className="flex flex-col gap-2 border-b border-neutral-100 pb-5 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <p className="text-xs font-bold uppercase tracking-[0.18em] text-primary-700">{leaderboard.scope_label}</p>
                <h2 className="mt-2 text-2xl font-bold tracking-tight text-dark">{leaderboard.period_label}</h2>
              </div>
              {leaderboard.current_user_rank ? (
                <p className="text-sm font-semibold text-neutral-600">
                  Your rank: <span className="text-dark">#{leaderboard.current_user_rank}</span>
                </p>
              ) : null}
            </div>

            {leaderboard.entries.length === 0 ? (
              <div className="py-12 text-center">
                <Users className="mx-auto h-9 w-9 text-primary-500" aria-hidden="true" />
                <h3 className="mt-3 text-lg font-bold text-dark">No practice scores yet</h3>
                <p className="mt-2 text-sm text-neutral-600">Start a practice test to put your group on the board.</p>
              </div>
            ) : (
              <div className="mt-5 space-y-3">
                {leaderboard.entries.map((entry) => {
                  const Icon = rankIcon(entry.rank)
                  return (
                    <div
                      key={`${entry.rank}-${entry.display_name}`}
                      className={`grid min-h-20 grid-cols-[2.5rem_minmax(0,1fr)_auto] items-center gap-3 rounded-2xl border p-3 sm:grid-cols-[3.5rem_minmax(0,1fr)_auto] sm:p-4 ${entry.is_current_user ? 'border-primary-300 bg-primary-50/60' : 'border-neutral-100 bg-neutral-50/60'}`}
                    >
                      <div className="flex flex-col items-center gap-1 text-neutral-500">
                        <Icon className={`h-5 w-5 ${entry.rank <= 3 ? 'text-secondary-600' : 'text-neutral-400'}`} aria-hidden="true" />
                        <span className="text-xs font-bold">#{entry.rank}</span>
                      </div>
                      <div className="min-w-0">
                        <p className="truncate text-base font-bold text-dark">{entry.display_name}</p>
                        <p className="mt-1 text-xs font-semibold uppercase tracking-[0.12em] text-neutral-500">{rankLabel(entry)}</p>
                      </div>
                      <div className="text-right">
                        <p className="text-xl font-bold text-dark">{entry.score}</p>
                        <p className="text-xs font-semibold text-neutral-500">points</p>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </section>
        ) : null}
      </main>
    </div>
  )
}
