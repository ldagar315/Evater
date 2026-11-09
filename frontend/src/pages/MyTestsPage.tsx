import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, BookOpen, FileText, History } from 'lucide-react'
import { Header } from '../components/layout/Header'
import { useAuthContext } from '../contexts/AuthContext'
import { supabase } from '../lib/supabase'

type PracticeAttempt = {
  id: string
  mode: string
  status: string
  score: number | null
  maximum_marks: number | null
  started_at: string
  completed_at: string | null
  chapters: { title: string; curriculum_versions: { subject: string } | null } | null
}
const PAGE_SIZE = 20
const linkClass = 'inline-flex min-h-11 items-center justify-center gap-2 rounded-xl border border-primary-200 bg-white px-4 py-2 text-sm font-bold text-primary-700 hover:bg-primary-50 focus:outline-none focus:ring-2 focus:ring-primary-500'
function dateLabel(value: string) {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? 'Date unavailable' : date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

export function MyTestsPage() {
  const { user } = useAuthContext()
  const [attempts, setAttempts] = useState<PracticeAttempt[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [page, setPage] = useState(0)
  const [hasMore, setHasMore] = useState(false)
  const [retry, setRetry] = useState(0)

  useEffect(() => { setAttempts([]); setPage(0); setHasMore(false) }, [user?.id])
  useEffect(() => {
    let active = true
    if (!user?.id) { setLoading(false); return }
    setLoading(true); setError('')
    void Promise.resolve(supabase.from('test_attempts')
      .select('id,status,mode,score,maximum_marks,started_at,completed_at,chapters(title,curriculum_versions(subject))')
      .eq('user_id', user.id).order('started_at', { ascending: false }).order('id', { ascending: false })
      .range(page * PAGE_SIZE, (page + 1) * PAGE_SIZE))
      .then(({ data, error: readError }) => {
        if (!active) return
        if (readError) throw readError
        const rows = (data || []) as PracticeAttempt[]
        setHasMore(rows.length > PAGE_SIZE)
        setAttempts((current) => page === 0 ? rows.slice(0, PAGE_SIZE) : [...current, ...rows.slice(0, PAGE_SIZE).filter((row) => !current.some((old) => old.id === row.id))])
      })
      .catch(() => { if (active) setError('Your practice history could not be loaded. Try again. Your saved tests have not been changed.') })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [user?.id, page, retry])

  return <div className="min-h-screen bg-cream font-sans text-dark">
    <Header />
    <main className="mx-auto max-w-5xl px-4 py-10 sm:px-6 sm:py-14">
      <p className="text-xs font-bold uppercase tracking-[0.18em] text-primary-700">Your learning</p>
      <h1 className="mt-2 text-4xl font-bold tracking-tight">My tests</h1>
      <p className="mt-3 max-w-2xl text-base leading-7 text-neutral-600">Find your previous practice tests, saved test papers and paper feedback here.</p>
      <div className="mt-7 grid gap-4 sm:grid-cols-2">
        <Link to="/previous-tests" className="rounded-2xl border border-neutral-200 bg-white p-5 hover:border-primary-300 focus:outline-none focus:ring-2 focus:ring-primary-500">
          <FileText className="h-6 w-6 text-primary-600" aria-hidden="true" /><h2 className="mt-3 text-lg font-bold">Test papers <ArrowRight className="ml-1 inline h-4 w-4" aria-hidden="true" /></h2><p className="mt-1 text-sm leading-6 text-neutral-600">Open papers you created earlier.</p>
        </Link>
        <Link to="/previous-feedbacks" className="rounded-2xl border border-neutral-200 bg-white p-5 hover:border-primary-300 focus:outline-none focus:ring-2 focus:ring-primary-500">
          <BookOpen className="h-6 w-6 text-primary-600" aria-hidden="true" /><h2 className="mt-3 text-lg font-bold">Paper feedback <ArrowRight className="ml-1 inline h-4 w-4" aria-hidden="true" /></h2><p className="mt-1 text-sm leading-6 text-neutral-600">Find feedback from papers you submitted.</p>
        </Link>
      </div>
      <section className="mt-8 rounded-3xl border border-neutral-200 bg-white p-5 sm:p-7" aria-labelledby="practice-history-title">
        <h2 id="practice-history-title" className="flex items-center gap-2 text-2xl font-bold"><History className="h-6 w-6 text-primary-600" aria-hidden="true" />Practice history</h2>
        <p className="mt-2 text-sm leading-6 text-neutral-600">Chapter practice and weak-spot review, newest first. Scores appear only for completed tests.</p>
        {error && <div role="alert" className="mt-5 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">{error}<button type="button" className="ml-2 min-h-11 font-bold underline" onClick={() => setRetry((value) => value + 1)}>Retry history</button></div>}
        {!loading && !error && attempts.length === 0 && <div className="py-10 text-center"><p className="font-semibold">No practice tests yet</p><p className="mt-2 text-sm text-neutral-500">Your chapter practice and review tests will appear here.</p><Link to="/practice" className={`${linkClass} mt-5`}>Start practice <ArrowRight className="h-4 w-4" aria-hidden="true" /></Link></div>}
        {attempts.length > 0 && <ul className="mt-5 divide-y divide-neutral-100">
          {attempts.map((attempt) => <li key={attempt.id} className="flex flex-col gap-3 py-5 sm:flex-row sm:items-center sm:justify-between">
            <div className="min-w-0"><p className="text-xs font-bold uppercase tracking-wide text-primary-700">{attempt.mode === 'review' ? 'Across chapters · Review' : `${attempt.chapters?.curriculum_versions?.subject || 'Chapter practice'} · ${attempt.mode}`}</p><h3 className="mt-1 break-words text-lg font-bold">{attempt.mode === 'review' ? 'Weak-spot review' : attempt.chapters?.title || 'Chapter unavailable'}</h3><p className="mt-1 text-sm text-neutral-500">{attempt.status === 'completed' && attempt.completed_at ? 'Completed' : 'Started'} {dateLabel(attempt.completed_at || attempt.started_at)}</p></div>
            <div className="shrink-0 sm:text-right"><span className={`inline-block rounded-full px-3 py-1 text-xs font-bold ${attempt.status === 'completed' ? 'bg-primary-50 text-primary-700' : 'bg-neutral-100 text-neutral-600'}`}>{attempt.status === 'completed' ? 'Completed' : attempt.status === 'in_progress' ? 'Not completed' : 'Abandoned'}</span>{attempt.status === 'completed' && attempt.score != null && attempt.maximum_marks != null && <p className="mt-2 text-xl font-bold">{attempt.score} / {attempt.maximum_marks}<span className="ml-1 text-xs font-medium text-neutral-500">marks</span></p>}</div>
          </li>)}
        </ul>}
        {loading && <p role="status" className="py-6 text-center text-sm text-neutral-500">{attempts.length ? 'Loading more tests...' : 'Loading your practice history...'}</p>}
        {hasMore && !loading && !error && <button type="button" onClick={() => setPage((value) => value + 1)} className={`${linkClass} mt-4 w-full`}>Show older practice tests</button>}
      </section>
    </main>
  </div>
}
