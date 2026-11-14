import { ArrowUpRight, Coins, ShieldCheck } from 'lucide-react'
import { Link } from 'react-router-dom'

export function creditLabel(balance: number | null | undefined): string {
  return balance == null ? 'Balance unavailable' : `${balance.toLocaleString('en-IN')} credits`
}

export function CreditBalance({ balance, variant = 'compact', onNavigate }: {
  balance: number | null | undefined
  onNavigate?: () => void
  variant?: 'compact' | 'summary' | 'panel'
}) {
  const known = balance != null
  const value = known ? balance.toLocaleString('en-IN') : '—'
  if (variant === 'panel') return (
    <aside id="credits" tabIndex={-1} aria-labelledby="credit-heading" className="scroll-mt-28 overflow-hidden rounded-3xl border border-primary-200 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-primary-500">
      <div className="bg-dark p-6 text-white sm:p-8">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-white/10 text-secondary-300"><Coins className="h-5 w-5" aria-hidden="true" /></span>
          <span className="rounded-full border border-white/20 px-3 py-1 text-[10px] font-bold uppercase tracking-[0.12em] text-neutral-200">Your account</span>
        </div>
        <h2 id="credit-heading" className="mt-6 text-xs font-bold uppercase tracking-[0.18em] text-secondary-300">Credits</h2>
        <p className="mt-2 break-all text-5xl font-bold tracking-tight" aria-label={creditLabel(balance)}>{value}</p>
        <p className="mt-2 text-sm text-neutral-300">{known ? 'Current balance' : 'Your balance could not be loaded.'}</p>
      </div>
      <div className="space-y-5 p-6 sm:p-8">
        <div><p className="text-sm font-bold text-dark">A place for future learning extras.</p><p className="mt-2 text-sm leading-6 text-neutral-600">Credits are reserved for future premium learning features. Those features are not available here yet.</p></div>
        <div className="flex gap-3 rounded-2xl border border-neutral-200 bg-cream p-4"><ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-primary-700" aria-hidden="true" /><p className="text-xs leading-5 text-neutral-600">Credits and leaderboard points are separate. Your practice results build your leaderboard score, not this balance.</p></div>
        <Link to="/practice" className="flex min-h-11 items-center justify-between gap-3 rounded-xl bg-primary-700 px-4 py-3 text-sm font-bold text-white transition-colors hover:bg-primary-800 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2">Go to practice<ArrowUpRight className="h-4 w-4 shrink-0" aria-hidden="true" /></Link>
      </div>
    </aside>
  )
  if (variant === 'summary') return (
    <Link to="/profile#credits" onClick={onNavigate} aria-label={`View credits: ${creditLabel(balance)}`} className="group flex min-h-[6.75rem] min-w-0 flex-col items-start gap-3 rounded-2xl border border-primary-200 bg-primary-50/60 p-3 shadow-sm transition-colors hover:bg-primary-50 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 sm:min-h-[7.5rem] sm:flex-row sm:items-center sm:gap-4 sm:p-6">
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-primary-100 text-primary-700 sm:h-11 sm:w-11"><Coins className="h-5 w-5" aria-hidden="true" /></span>
      <span className="min-w-0"><span className="block text-[0.6rem] font-bold uppercase tracking-[0.12em] text-primary-700 sm:text-xs">Credits</span><span className="mt-1 block break-all text-sm font-bold text-dark sm:text-xl">{value}</span><span className="mt-1 flex items-center gap-1 text-[10px] font-semibold text-primary-700 sm:text-xs">View balance<ArrowUpRight className="h-3 w-3" aria-hidden="true" /></span></span>
    </Link>
  )
  return (
    <Link to="/profile#credits" onClick={onNavigate} aria-label={`View credits: ${creditLabel(balance)}`} className="inline-flex min-h-11 min-w-0 self-start items-center gap-2 rounded-xl border border-primary-200 bg-primary-50 px-3 py-2 text-primary-800 transition-colors hover:bg-primary-100 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2">
      <Coins className="h-4 w-4 shrink-0" aria-hidden="true" /><span className="min-w-0"><span className="block break-all text-sm font-bold leading-4">{value}</span><span className="block text-[10px] font-semibold leading-4">Credits</span></span><ArrowUpRight className="h-3 w-3 shrink-0" aria-hidden="true" />
    </Link>
  )
}
