import { Link } from 'react-router-dom'
import { reviewedPracticeTargets } from '../../data/reviewedPracticeTargets'
import { practiceHref } from '../../lib/practiceTarget'

export function ChapterConceptLinks({ subject, chapter }: { subject: string; chapter: string }) {
  const targets = reviewedPracticeTargets.filter(target => target.subject === subject && target.chapter === chapter)
  if (targets.length === 0) return null
  return (
    <section aria-label="Single-concept practice" className="border-t border-neutral-100 pt-7">
      <h2 className="text-sm font-bold text-dark">Practise a single concept</h2>
      <p className="mt-2 text-sm leading-6 text-neutral-500">Open a concept to check its available questions. Your test starts only when you choose Start.</p>
      <div className="mt-3 flex flex-wrap gap-2">
        {targets.map(target => <Link key={target.concept} to={practiceHref(target)} className="inline-flex min-h-12 items-center rounded-xl border border-primary-200 bg-primary-50 px-4 py-3 text-sm font-semibold text-primary-800 transition-colors hover:border-primary-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2">{target.label}</Link>)}
      </div>
    </section>
  )
}
