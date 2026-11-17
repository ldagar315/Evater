import { Download, ArrowUpRight, FileText, Search, SlidersHorizontal, X } from 'lucide-react'
import { Link, useSearchParams } from 'react-router-dom'
import { Header } from '../components/layout/Header'
import { Footer } from '../components/layout/Footer'
import { Seo } from '../components/seo/Seo'
import { useAuthContext } from '../contexts/AuthContext'
import { filterPapers, samplePapers } from '../data/samplePapers'
import { paperCollections, type PaperCollectionKey } from '../data/paperCollections'

const controlClass = 'min-h-12 w-full rounded-xl border border-neutral-200 bg-white px-3 text-sm text-dark focus:outline-none focus:ring-2 focus:ring-primary-600 focus:ring-offset-2'

export function PapersPage({ collectionKey = 'all' }: { collectionKey?: PaperCollectionKey }) {
  const collection = paperCollections[collectionKey]
  const collectionPapers = collection.subject ? samplePapers.filter(p => p.subject === collection.subject) : samplePapers
  const { user } = useAuthContext()
  const [params, setParams] = useSearchParams()
  const filters = { query: params.get('q') || '', classLevel: params.get('class') || '', subject: params.get('subject') || '', topic: params.get('topic') || '' }
  const results = filterPapers(collectionPapers, filters)
  const hasFilters = Object.values(filters).some(Boolean)
  const update = (key: string, value: string, replace = false) => {
    const next = new URLSearchParams(params)
    if (value) next.set(key, value)
    else next.delete(key)
    setParams(next, { replace, preventScrollReset: true })
  }
  const clear = () => setParams({}, { preventScrollReset: true })
  return <div className="min-h-screen bg-cream text-dark">
    <Seo title={collection.title} description={collection.description} path={collection.path} robots={hasFilters ? 'noindex, follow' : undefined} />
    <Header />
    <main id="papers-main" className="mx-auto max-w-7xl px-4 pb-16 pt-8 sm:px-6 sm:pb-24 sm:pt-16 lg:px-8">
      <div className="max-w-3xl">
        <p className="text-xs font-bold uppercase tracking-[0.16em] text-primary-700">The paper shelf</p>
        <h1 className="mt-3 text-3xl font-extrabold leading-tight tracking-tight sm:text-5xl">{collection.heading}</h1>
        <p className="mt-5 max-w-xl text-base leading-7 text-neutral-600">{collection.intro}</p>
        <div className="mt-5 flex flex-wrap gap-x-5 gap-y-2 text-xs font-semibold text-neutral-600"><span>Printable PDFs</span><span>Answers included</span><span>No sign-in needed</span></div>
      </div>

      <nav aria-label="Paper collections" className="mt-7 flex flex-wrap gap-2">{Object.entries(paperCollections).map(([key, item]) => <Link key={key} to={item.path} aria-current={key === collectionKey ? 'page' : undefined} className={`inline-flex min-h-11 items-center rounded-xl border px-4 py-2 text-sm font-bold ${key === collectionKey ? 'border-primary-700 bg-primary-700 text-white' : 'border-neutral-200 bg-white text-primary-800 hover:bg-primary-50'}`}>{item.label}</Link>)}</nav>
      <section aria-label="Find sample papers" className="mt-9 rounded-2xl border border-neutral-200 bg-white p-4 sm:mt-12 sm:p-6">
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-[minmax(0,2fr)_repeat(3,minmax(0,1fr))]">
          <div className="col-span-2 lg:col-span-1"><label htmlFor="paper-search" className="mb-2 block text-xs font-bold text-neutral-600">Search papers</label><div className="relative"><Search aria-hidden="true" className="pointer-events-none absolute left-3 top-3.5 h-5 w-5 text-neutral-400" /><input id="paper-search" type="search" value={filters.query} onChange={e => update('q', e.target.value, true)} placeholder="Try light, triangles, magnetism…" className={`${controlClass} pl-10`} /></div></div>
          <div><label htmlFor="paper-class" className="mb-2 block text-xs font-bold text-neutral-600">Class</label><select id="paper-class" value={filters.classLevel} onChange={e => update('class', e.target.value)} className={controlClass}><option value="">All classes</option><option value="8">Class 8</option></select></div>
          <div><label htmlFor="paper-subject" className="mb-2 block text-xs font-bold text-neutral-600">Subject</label><select id="paper-subject" value={filters.subject} onChange={e => update('subject', e.target.value)} className={controlClass}><option value="">All subjects</option>{(!collection.subject || collection.subject === 'Science') && <option>Science</option>}{(!collection.subject || collection.subject === 'Mathematics') && <option>Mathematics</option>}</select></div>
          <div className="col-span-2 lg:col-span-1"><label htmlFor="paper-topic" className="mb-2 block text-xs font-bold text-neutral-600">Topic</label><select id="paper-topic" value={filters.topic} onChange={e => update('topic', e.target.value)} className={controlClass}><option value="">All topics</option>{collectionPapers.map(p => <option key={p.id}>{p.topic}</option>)}</select></div>
        </div>
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-neutral-100 pt-4">
          <p className="flex items-center gap-2 text-xs leading-5 text-neutral-500"><SlidersHorizontal aria-hidden="true" className="h-4 w-4 shrink-0" />{collection.shelf}</p>
          {hasFilters && <button type="button" onClick={clear} className="inline-flex min-h-11 items-center gap-1.5 rounded-lg px-2 text-sm font-bold text-primary-700 hover:bg-primary-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary-700"><X aria-hidden="true" className="h-4 w-4" />Clear filters</button>}
        </div>
      </section>

      <div className="mb-5 mt-9 flex flex-wrap items-end justify-between gap-2"><h2 className="text-xl font-bold">{hasFilters ? 'Your results' : collection.listHeading}</h2><p role="status" aria-live="polite" aria-atomic="true" className="text-sm text-neutral-500">{results.length} {results.length === 1 ? 'paper' : 'papers'}</p></div>
      {results.length ? <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">{results.map(paper => <article key={paper.id} className="flex min-w-0 flex-col overflow-hidden rounded-2xl border border-neutral-200 bg-white transition-shadow hover:shadow-lg hover:shadow-primary-900/5">
        <div className={`relative flex h-44 items-center justify-between overflow-hidden border-b border-neutral-100 px-6 sm:h-48 ${paper.subject === 'Science' ? 'bg-[#edf5f1]' : 'bg-[#f6eedc]'}`} aria-hidden="true">
          <div className="z-10"><p className="text-[10px] font-bold uppercase tracking-[0.15em] text-neutral-500">Evater · Class {paper.classLevel}</p><p className="mt-3 max-w-40 text-2xl font-bold leading-tight text-[#173b38]">{paper.topic}</p><p className="mt-2 text-xs text-neutral-600">Practice. Check. Improve.</p></div>
          <div className="absolute -right-3 top-6 rotate-12 rounded-xl border border-white bg-white/70 p-6 shadow-sm"><FileText className="h-20 w-16 text-primary-700/50" strokeWidth={1} /></div>
        </div>
        <div className="flex flex-1 flex-col p-6">
          <div className="flex flex-wrap gap-2 text-[11px] font-bold"><span className="rounded-md bg-neutral-100 px-2 py-1 text-neutral-600">Class {paper.classLevel}</span><span className="rounded-md bg-primary-50 px-2 py-1 text-primary-700">{paper.subject}</span><span className="rounded-md bg-neutral-100 px-2 py-1 text-neutral-600">{paper.format === 'mixed' ? 'Mixed chapters' : 'Chapter paper'}</span></div>
          <h3 className="mt-4 text-xl font-bold leading-snug">{paper.title}</h3><p className="mt-2 text-sm leading-6 text-neutral-500">{paper.subtitle}</p>
          <p className="mb-5 mt-4 text-xs leading-5 text-neutral-600">{paper.questions} questions · {paper.marks} marks · {paper.minutes} min suggested</p>
          <p className="mt-auto border-t border-neutral-100 pt-4 text-xs leading-5 text-neutral-500">{paper.objectiveQuestions ?? 10} objective + {paper.writtenQuestions ?? 5} written responses.<br />Answers and marking guidance included.</p>
          <div className="mt-5 flex flex-wrap gap-2"><a href={paper.href} download className="inline-flex min-h-11 flex-1 items-center justify-center gap-2 rounded-xl bg-primary-700 px-4 text-sm font-bold text-white hover:bg-primary-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary-700" aria-label={`Download ${paper.title} PDF`}><Download aria-hidden="true" className="h-4 w-4" />Download</a><a href={paper.href} target="_blank" rel="noopener noreferrer" className="inline-flex min-h-11 items-center justify-center gap-1 rounded-xl border border-neutral-200 px-3 text-sm font-bold text-neutral-600 hover:bg-neutral-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary-700" aria-label={`Open ${paper.title} PDF in a new tab`}>Open PDF<ArrowUpRight aria-hidden="true" className="h-4 w-4" /></a></div>
        </div>
      </article>)}</div> : <div className="rounded-2xl border border-dashed border-neutral-300 bg-white px-5 py-14 text-center"><Search aria-hidden="true" className="mx-auto h-8 w-8 text-neutral-400" /><h3 className="mt-4 text-lg font-bold">No papers match those filters</h3><p className="mx-auto mt-2 max-w-md text-sm leading-6 text-neutral-500">Try another topic or clear your filters to see all available papers.</p><button type="button" onClick={clear} className="mt-5 min-h-11 rounded-xl bg-primary-700 px-5 text-sm font-bold text-white hover:bg-primary-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary-700">Show all papers</button></div>}
      <section aria-label="What's in these papers" className="mt-7 max-w-3xl text-sm leading-7 text-neutral-600"><h2 className="text-lg font-bold text-dark">Two ways to practise, with answers you can check</h2><p className="mt-2">{collection.coverage}</p><p className="mt-2">Each chapter paper has 15 questions for 22 marks: 10 objective questions and 5 written responses. Allow about 45 minutes. The two mixed-chapter papers have 24 questions for 60 marks: 12 objective questions and 12 written responses, including case questions. Allow about 150 minutes. They sample topics across chapters, not every syllabus skill. None of these papers is an official CBSE exam paper. Schools may use different exam patterns.</p></section>
      <section className="mt-12 max-w-3xl" aria-labelledby="paper-study-method"><h2 id="paper-study-method" className="text-xl font-bold">How to use a practice paper</h2><ol className="mt-4 list-decimal space-y-2 pl-5 text-sm leading-7 text-neutral-600"><li>Pick chapter revision or mixed-chapter practice and download its PDF. No account is needed.</li><li>Set aside the suggested 45 or 150 minutes shown on your paper. Attempt the questions before opening the answer pages, and write your working for Maths.</li><li>Check the model answers and marking guidance. Revise the ideas behind your mistakes, then try those questions again without looking.</li></ol><p className="mt-4 text-sm leading-7 text-neutral-600">Want an explanation before you practise? <Link to="/blog/" className="font-bold text-primary-700 underline underline-offset-4">Read the Class 8 study guides</Link> for worked explanations and topic practice.</p></section>
      <aside className="mt-9 max-w-2xl border-l-2 border-primary-200 pl-4"><p className="text-xs leading-6 text-neutral-500">Original Evater practice papers, not official NCERT papers or marking schemes. These PDFs are for self-study. Downloading a paper does not start a test or provide automatic grading.</p></aside>
    </main>
    {!user && <Footer />}
  </div>
}
