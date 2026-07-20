import React, { useEffect, useState } from 'react'
import {
  ArrowLeft,
  CalendarDays,
  Crown,
  Medal,
  RefreshCw,
  ShieldCheck,
  Target,
  Trophy,
  Users,
} from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import { Header } from '../components/layout/Header'
import { Button } from '../components/ui/Button'
import { LeagueEmblem } from '../components/leaderboard/LeagueEmblem'
import { getLeaderboard, LeaderboardEntry, LeaderboardResponse } from '../lib/api'

function rankIcon(rank: number) {
  if (rank === 1) return Crown
  if (rank === 2) return Medal
  if (rank === 3) return Trophy
  return Target
}

function formatDate(value?: string | null) {
  if (!value) return 'soon'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return 'soon'
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

function rankLabel(entry: LeaderboardEntry) {
  return `${entry.league_label} · #${entry.league_rank} in league`
}

export function LeaderboardPage() {
  const navigate = useNavigate()
  const [leaderboard, setLeaderboard] = useState<LeaderboardResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [reloadToken, setReloadToken] = useState(0)

  useEffect(() => {
    let active = true
    setLoading(true)
    setError(null)

    getLeaderboard()
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
  }, [reloadToken])

  const currentEntry = leaderboard?.entries.find((entry) => entry.is_current_user)
  const currentTier = leaderboard?.current_user_league || currentEntry?.league_tier || 'bronze_3'
  const currentLabel = leaderboard?.current_user_league_label || currentEntry?.league_label || 'Not placed yet'
  const currentPoints = currentEntry?.score ?? 0
  const leagueRank = leaderboard?.current_user_league_rank || currentEntry?.league_rank || '—'

  return (
    <div className="min-h-screen bg-cream font-sans text-dark">
      <Header />
      <main className="mx-auto max-w-6xl px-4 pb-16 pt-8 sm:px-6 lg:px-8 lg:pb-24 lg:pt-12">
        <button
          type="button"
          onClick={() => navigate('/practice')}
          className="inline-flex min-h-11 items-center gap-2 rounded-lg px-2 text-sm font-semibold text-neutral-600 transition-colors hover:bg-white hover:text-dark focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
        >
          <ArrowLeft className="h-4 w-4" aria-hidden="true" />
          Back to practice
        </button>

        <section className="relative mt-7 overflow-hidden rounded-[2rem] border border-dark bg-dark p-6 text-white shadow-xl shadow-dark/10 sm:p-9">
          <div className="pointer-events-none absolute -right-20 -top-24 h-72 w-72 rounded-full bg-primary-700/30 blur-3xl" />
          <div className="pointer-events-none absolute -bottom-32 left-1/3 h-64 w-64 rounded-full bg-secondary-500/20 blur-3xl" />
          <div className="relative grid gap-8 lg:grid-cols-[1.1fr_.9fr] lg:items-end">
            <div>
              <p className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-[0.2em] text-secondary-300">
                <Trophy className="h-4 w-4" aria-hidden="true" />
                Global student league
              </p>
              <h1 className="mt-4 max-w-2xl text-3xl font-bold leading-tight tracking-tight sm:text-5xl">
                Earn your way up the ladder.
              </h1>
              <p className="mt-4 max-w-2xl text-base leading-7 text-neutral-300 sm:text-lg">
                Complete practice sessions, earn points, and compete with verified students worldwide.
              </p>
              <div className="mt-6 flex flex-wrap items-center gap-3">
                <span className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/10 px-3 py-2 text-xs font-bold uppercase tracking-[0.12em] text-neutral-300">
                  <ShieldCheck className="h-4 w-4 text-secondary-300" aria-hidden="true" />
                  Verified students
                </span>
                <Link
                  to="/leagues"
                  className="inline-flex min-h-11 items-center rounded-xl bg-secondary-300 px-4 py-3 text-sm font-bold text-dark transition-colors hover:bg-secondary-200 focus:outline-none focus:ring-2 focus:ring-secondary-200 focus:ring-offset-2 focus:ring-offset-dark"
                >
                  Know more about leagues
                </Link>
              </div>
            </div>

            <div className="rounded-3xl border border-white/15 bg-white/10 p-5 backdrop-blur sm:p-6">
              <div className="flex items-center gap-4">
                <LeagueEmblem tier={currentTier} size="lg" />
                <div>
                  <p className="text-xs font-bold uppercase tracking-[0.16em] text-neutral-400">Your current league</p>
                  <p className="mt-1 text-2xl font-bold text-white">{currentLabel}</p>
                </div>
              </div>
              <div className="mt-5 grid grid-cols-3 gap-3 border-t border-white/10 pt-5">
                <div>
                  <p className="text-xs font-bold uppercase tracking-[0.12em] text-neutral-400">Points</p>
                  <p className="mt-1 text-2xl font-bold text-white">{currentPoints}</p>
                </div>
                <div>
                  <p className="text-xs font-bold uppercase tracking-[0.12em] text-neutral-400">Global rank</p>
                  <p className="mt-1 text-2xl font-bold text-white">#{leaderboard?.current_user_rank || '—'}</p>
                </div>
                <div>
                  <p className="text-xs font-bold uppercase tracking-[0.12em] text-neutral-400">League rank</p>
                  <p className="mt-1 text-2xl font-bold text-white">#{leagueRank}</p>
                </div>
              </div>
              <div className="mt-5 flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-white/10 pt-4 text-sm font-semibold text-neutral-300">
                <span className="inline-flex items-center gap-2"><CalendarDays className="h-4 w-4 text-secondary-300" aria-hidden="true" />Season {leaderboard?.season_number || '—'}</span>
                <span>Ends {formatDate(leaderboard?.season_ends_at)}</span>
              </div>
            </div>
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
            <h2 className="mt-4 text-2xl font-bold text-dark">Verification needed</h2>
            <p className="mx-auto mt-2 max-w-lg leading-6 text-neutral-600">
              {leaderboard.membership_message || 'Only verified student accounts appear in the global league.'}
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
              <div className="text-sm font-semibold text-neutral-600 sm:text-right">
                {leaderboard.topper_name && <p>Topper: <span className="text-dark">{leaderboard.topper_name}</span></p>}
              </div>
            </div>

            {leaderboard.entries.length === 0 ? (
              <div className="py-12 text-center">
                <Users className="mx-auto h-9 w-9 text-primary-500" aria-hidden="true" />
                <h3 className="mt-3 text-lg font-bold text-dark">No league scores yet</h3>
                <p className="mt-2 text-sm text-neutral-600">Complete a practice session to enter the league.</p>
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
                      <div className="flex min-w-0 items-center gap-3">
                        <LeagueEmblem tier={entry.league_tier} size="sm" />
                        <div className="min-w-0">
                          <div className="flex flex-wrap items-center gap-2">
                            <p className="truncate text-base font-bold text-dark">{entry.display_name}</p>
                            {entry.is_current_user && <span className="rounded-full bg-primary-600 px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.08em] text-white">You</span>}
                          </div>
                          <p className="mt-1 text-xs font-semibold uppercase tracking-[0.12em] text-neutral-500">{rankLabel(entry)}</p>
                        </div>
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
