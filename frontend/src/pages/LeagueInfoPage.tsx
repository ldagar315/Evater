import React from 'react'
import {
  ArrowLeft,
  BookOpen,
  CalendarDays,
  CheckCircle2,
  Clock3,
  Info,
  Trophy,
  XCircle,
} from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import { Header } from '../components/layout/Header'
import { LeagueEmblem } from '../components/leaderboard/LeagueEmblem'
import { LEAGUE_LADDER, SCORING_ROWS } from '../components/leaderboard/leagueData'

export function LeagueInfoPage() {
  const navigate = useNavigate()

  return (
    <div className="min-h-screen bg-cream font-sans text-dark">
      <Header />
      <main className="mx-auto max-w-6xl px-4 pb-16 pt-8 sm:px-6 lg:px-8 lg:pb-24 lg:pt-12">
        <button
          type="button"
          onClick={() => navigate('/leaderboard')}
          className="inline-flex min-h-11 items-center gap-2 rounded-lg px-2 text-sm font-semibold text-neutral-600 transition-colors hover:bg-white hover:text-dark focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
        >
          <ArrowLeft className="h-4 w-4" aria-hidden="true" />
          Back to leaderboard
        </button>

        <section className="relative mt-7 overflow-hidden rounded-[2rem] border border-dark bg-dark p-6 text-white shadow-xl shadow-dark/10 sm:p-9">
          <div className="pointer-events-none absolute -right-20 -top-24 h-72 w-72 rounded-full bg-primary-700/30 blur-3xl" />
          <div className="pointer-events-none absolute -bottom-32 left-1/3 h-64 w-64 rounded-full bg-secondary-500/20 blur-3xl" />
          <div className="relative grid gap-8 lg:grid-cols-[1.1fr_.9fr] lg:items-end">
            <div>
              <p className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-[0.2em] text-secondary-300">
                <BookOpen className="h-4 w-4" aria-hidden="true" />
                League guide
              </p>
              <h1 className="mt-4 max-w-2xl text-3xl font-bold leading-tight tracking-tight sm:text-5xl">
                Know the rules. Enjoy the climb.
              </h1>
              <p className="mt-4 max-w-2xl text-base leading-7 text-neutral-300 sm:text-lg">
                Leagues turn consistent practice into a friendly global race. Here is everything you need to know before you chase your next crest.
              </p>
            </div>
            <div className="flex items-center gap-4 rounded-3xl border border-white/15 bg-white/10 p-5 backdrop-blur sm:p-6">
              <LeagueEmblem tier="diamond_1" size="lg" />
              <div>
                <p className="text-xs font-bold uppercase tracking-[0.16em] text-neutral-400">The destination</p>
                <p className="mt-1 text-2xl font-bold text-white">Diamond I</p>
                <p className="mt-1 text-sm font-semibold text-neutral-300">15 stages · one step at a time</p>
              </div>
            </div>
          </div>
        </section>

        <section className="mt-8 grid gap-6 lg:grid-cols-[1.05fr_.95fr]">
          <div className="rounded-3xl border border-primary-100 bg-primary-900 p-6 text-white shadow-sm sm:p-8">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-xs font-bold uppercase tracking-[0.18em] text-secondary-300">The simple version</p>
                <h2 className="mt-2 text-2xl font-bold tracking-tight">Practice → points → promotion</h2>
              </div>
              <Trophy className="h-7 w-7 shrink-0 text-secondary-300" aria-hidden="true" />
            </div>
            <div className="mt-7 space-y-5">
              <div className="flex gap-4">
                <span className="text-sm font-bold text-secondary-300">01</span>
                <div>
                  <h3 className="font-bold">What is a league?</h3>
                  <p className="mt-1 text-sm leading-6 text-neutral-300">Your league is your current stage on a 15-step ladder, from Bronze III to Diamond I. Your global rank is always shown alongside it.</p>
                </div>
              </div>
              <div className="flex gap-4">
                <span className="text-sm font-bold text-secondary-300">02</span>
                <div>
                  <h3 className="font-bold">What is a season?</h3>
                  <p className="mt-1 text-sm leading-6 text-neutral-300">A season lasts 14 days. Promotion and demotion are checked only when it ends, while your points carry forward from season to season like trophies.</p>
                </div>
              </div>
              <div className="flex gap-4">
                <span className="text-sm font-bold text-secondary-300">03</span>
                <div>
                  <h3 className="font-bold">How do I move up?</h3>
                  <p className="mt-1 text-sm leading-6 text-neutral-300">The topper promotes by one stage when the threshold is reached. Only one student can promote; demotion happens only when a league has at least three students.</p>
                </div>
              </div>
              <div className="flex gap-4">
                <span className="text-sm font-bold text-secondary-300">04</span>
                <div>
                  <h3 className="font-bold">What if I take a break?</h3>
                  <p className="mt-1 text-sm leading-6 text-neutral-300">Your first inactive season is protected: no penalty and no demotion. From the next inactive season, points reduce by 10% per season until they reach half of the balance held when the break began. Then the reduction and demotion stop.</p>
                </div>
              </div>
            </div>
          </div>

          <div className="rounded-3xl border border-neutral-200 bg-white p-6 shadow-sm sm:p-8">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-xs font-bold uppercase tracking-[0.18em] text-primary-700">Practice scoring</p>
                <h2 className="mt-2 text-2xl font-bold tracking-tight text-dark">Every question matters</h2>
              </div>
              <Clock3 className="h-7 w-7 shrink-0 text-primary-600" aria-hidden="true" />
            </div>
            <div className="mt-6 overflow-hidden rounded-2xl border border-neutral-100">
              <table className="w-full text-left text-sm">
                <caption className="sr-only">Points awarded for correct, wrong, and skipped questions</caption>
                <thead className="bg-neutral-50 text-[10px] font-bold uppercase tracking-[0.12em] text-neutral-500">
                  <tr>
                    <th className="px-3 py-3 sm:px-4">Difficulty</th>
                    <th className="px-3 py-3 text-right sm:px-4">Correct</th>
                    <th className="px-3 py-3 text-right sm:px-4">Wrong</th>
                    <th className="px-3 py-3 text-right sm:px-4">Skipped</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-neutral-100">
                  {SCORING_ROWS.map((row) => (
                    <tr key={row.difficulty}>
                      <th className="px-3 py-4 font-bold text-dark sm:px-4">
                        <span className={`mr-2 inline-block h-2 w-2 rounded-full ${row.dotColor}`} />
                        {row.difficulty}
                      </th>
                      <td className="px-3 py-4 text-right font-bold text-emerald-700 sm:px-4">{row.value}</td>
                      <td className="px-3 py-4 text-right font-semibold text-red-600 sm:px-4">{row.wrong}</td>
                      <td className="px-3 py-4 text-right font-semibold text-red-600 sm:px-4">{row.skipped}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-5 space-y-3 text-sm leading-6 text-neutral-600">
              <p className="flex gap-2"><CheckCircle2 className="mt-1 h-4 w-4 shrink-0 text-emerald-600" aria-hidden="true" />Correct answers add points to your season score.</p>
              <p className="flex gap-2"><Info className="mt-1 h-4 w-4 shrink-0 text-primary-600" aria-hidden="true" />Wrong answers cost 25% of the question value; skips cost 50%.</p>
              <p className="flex gap-2"><XCircle className="mt-1 h-4 w-4 shrink-0 text-red-600" aria-hidden="true" />Higher leagues award fewer points, so climbing stays challenging.</p>
            </div>
          </div>
        </section>

        <section className="mt-8 rounded-3xl border border-neutral-200 bg-white p-6 shadow-sm sm:p-8">
          <div className="flex flex-col gap-3 border-b border-neutral-100 pb-5 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="text-xs font-bold uppercase tracking-[0.18em] text-primary-700">The 15-stage ladder</p>
              <h2 className="mt-2 text-2xl font-bold tracking-tight text-dark">Find your next crest</h2>
            </div>
            <p className="max-w-md text-sm leading-6 text-neutral-600">The number shown is the minimum score the league topper needs at season end to promote by one stage.</p>
          </div>
          <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {LEAGUE_LADDER.map((league, index) => (
              <div key={league.tier} className="flex items-center gap-3 rounded-2xl border border-neutral-100 bg-neutral-50/70 p-3">
                <LeagueEmblem tier={league.tier} size="sm" />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-bold text-dark">{league.label}</p>
                  <p className="mt-1 text-xs font-semibold text-neutral-500">{league.threshold ? `${league.threshold} points to promote` : 'Top league · no higher stage'}</p>
                </div>
                <span className="text-xs font-bold text-neutral-400">{String(index + 1).padStart(2, '0')}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="mt-8 flex flex-col gap-4 rounded-3xl border border-secondary-200 bg-secondary-50 p-6 sm:flex-row sm:items-center sm:justify-between sm:p-8">
          <div className="flex items-start gap-3">
            <CalendarDays className="mt-1 h-5 w-5 shrink-0 text-secondary-700" aria-hidden="true" />
            <div>
              <p className="font-bold text-dark">Ready to see your place?</p>
              <p className="mt-1 text-sm leading-6 text-neutral-700">Return to the leaderboard to see your current crest, season rank, and the students you are chasing.</p>
            </div>
          </div>
          <Link to="/leaderboard" className="inline-flex min-h-11 shrink-0 items-center justify-center rounded-xl bg-dark px-5 py-3 text-sm font-bold text-white transition-colors hover:bg-primary-900 focus:outline-none focus:ring-2 focus:ring-dark focus:ring-offset-2 focus:ring-offset-secondary-50">
            View leaderboard
          </Link>
        </section>
      </main>
    </div>
  )
}
