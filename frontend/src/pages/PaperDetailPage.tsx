import { ArrowUpRight, BookOpen, CheckCircle2, Download, FileText } from 'lucide-react'
import { Link, useParams } from 'react-router-dom'
import { Header } from '../components/layout/Header'
import { Footer } from '../components/layout/Footer'
import { Seo } from '../components/seo/Seo'
import { useAuthContext } from '../contexts/AuthContext'
import { getPaperDetailBySlug, paperForDetail } from '../data/paperDetails'
import { paperCollections } from '../data/paperCollections'
import { paperDetails } from '../data/paperDetails'
import { NotFoundPage } from './NotFoundPage'

const chipClass = 'rounded-md bg-neutral-100 px-2 py-1 text-[11px] font-bold text-neutral-600'
const subjectChipClass = 'rounded-md bg-primary-50 px-2 py-1 text-[11px] font-bold text-primary-700'
const buttonFocus = 'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary-700'

export function PaperDetailPage() {
  const { paperSlug } = useParams()
  const detail = getPaperDetailBySlug(paperSlug)
  const { user } = useAuthContext()

  if (!detail) return <NotFoundPage />

  const paper = paperForDetail(detail)
  const collection = paper.subject === 'Science' ? paperCollections.science : paperCollections.maths
  const related = paperDetails
    .filter(d => d.paperId !== detail.paperId)
    .map(d => ({ d, p: paperForDetail(d) }))
    .filter(({ p }) => p.subject === paper.subject)
    .sort((a, b) => Number(b.p.format === 'mixed') - Number(a.p.format === 'mixed'))
    .slice(0, 3)

  const objective = paper.objectiveQuestions ?? 10
  const written = paper.writtenQuestions ?? 5
  const jsonLd = [
    {
      '@context': 'https://schema.org',
      '@type': 'BreadcrumbList',
      itemListElement: [
        { '@type': 'ListItem', position: 1, name: 'Class 8 sample papers', item: `https://evater.xyz/papers/` },
        { '@type': 'ListItem', position: 2, name: collection.label, item: `https://evater.xyz${collection.path}` },
        { '@type': 'ListItem', position: 3, name: detail.heading, item: `https://evater.xyz${detail.path}` },
      ],
    },
    {
      '@context': 'https://schema.org',
      '@type': 'LearningResource',
      name: detail.heading,
      description: detail.seoDescription,
      url: `https://evater.xyz${detail.path}`,
      learningResourceType: 'Practice problems',
      educationalLevel: 'Class 8',
      teaches: paper.topic,
      inLanguage: 'en',
      isAccessibleForFree: true,
      provider: { '@type': 'Organization', name: 'Evater', url: 'https://evater.xyz' },
      hasPart: { '@type': 'DigitalDocument', name: `${paper.title} PDF`, url: `https://evater.xyz${paper.href}`, encodingFormat: 'application/pdf' },
    },
  ]

  return <div className="min-h-screen bg-cream text-dark">
    <Seo
      title={detail.seoTitle}
      description={detail.seoDescription}
      path={detail.path}
      keywords={[paper.title, paper.topic, `class ${paper.classLevel} ${paper.subject}`, 'sample paper', 'practice paper', 'answers', ...paper.keywords]}
      jsonLd={jsonLd}
    />
    <Header />
    <main id="paper-detail-main" className="mx-auto max-w-7xl px-4 pb-16 pt-8 sm:px-6 sm:pb-24 sm:pt-16 lg:px-8">
      <nav aria-label="Breadcrumb" className="text-xs font-semibold text-neutral-500">
        <ol className="flex flex-wrap items-center gap-1.5">
          <li><Link to="/papers/" className="hover:text-primary-700 hover:underline underline-offset-4">Class 8 sample papers</Link></li>
          <li aria-hidden="true">/</li>
          <li><Link to={collection.path} className="hover:text-primary-700 hover:underline underline-offset-4">{collection.label}</Link></li>
          <li aria-hidden="true">/</li>
          <li aria-current="page" className="text-neutral-700">{paper.format === 'mixed' ? 'Mixed-chapter paper' : paper.topic}</li>
        </ol>
      </nav>

      <div className="mt-8 grid gap-10 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <div className="min-w-0">
          <p className="text-xs font-bold uppercase tracking-[0.16em] text-primary-700">Evater · Class {paper.classLevel} {paper.subject}</p>
          <h1 className="mt-3 text-3xl font-extrabold leading-tight tracking-tight sm:text-4xl">{detail.heading}</h1>
          <p className="mt-5 max-w-2xl text-base leading-7 text-neutral-600">{detail.intro}</p>
          <div className="mt-5 flex flex-wrap gap-2">
            <span className={chipClass}>Class {paper.classLevel}</span>
            <span className={subjectChipClass}>{paper.subject}</span>
            <span className={chipClass}>{paper.format === 'mixed' ? 'Mixed chapters' : 'Chapter paper'}</span>
            <span className={chipClass}>{paper.questions} questions</span>
            <span className={chipClass}>{paper.marks} marks</span>
            <span className={chipClass}>{paper.minutes} min suggested</span>
          </div>

          <div className="mt-7 flex flex-wrap gap-3">
            <a href={paper.href} download className={`inline-flex min-h-11 items-center justify-center gap-2 rounded-xl bg-primary-700 px-5 text-sm font-bold text-white hover:bg-primary-800 ${buttonFocus}`} aria-label={`Download ${paper.title} PDF`}>
              <Download aria-hidden="true" className="h-4 w-4" />Download PDF
            </a>
            <a href={paper.href} target="_blank" rel="noopener noreferrer" className={`inline-flex min-h-11 items-center justify-center gap-1.5 rounded-xl border border-neutral-200 bg-white px-4 text-sm font-bold text-neutral-700 hover:bg-neutral-50 ${buttonFocus}`} aria-label={`Open ${paper.title} PDF in a new tab`}>
              Open PDF<ArrowUpRight aria-hidden="true" className="h-4 w-4" />
            </a>
          </div>
          <p className="mt-3 flex items-center gap-2 text-xs leading-5 text-neutral-500"><CheckCircle2 aria-hidden="true" className="h-4 w-4 shrink-0 text-primary-700" />Free printable PDF. Answers and marking guidance included. No sign-in needed.</p>

          <section aria-labelledby="inside-paper" className="mt-12">
            <h2 id="inside-paper" className="text-xl font-bold">What's inside this paper</h2>
            <ul className="mt-4 space-y-2.5">
              {detail.inside.map(item => <li key={item} className="flex gap-3 text-sm leading-6 text-neutral-600"><CheckCircle2 aria-hidden="true" className="mt-1 h-4 w-4 shrink-0 text-primary-700" /><span>{item}</span></li>)}
            </ul>
            {paper.format !== 'mixed' && <p className="mt-4 max-w-2xl text-sm leading-6 text-neutral-500">The paper has {objective} objective questions and {written} written-response questions. It practises this chapter's ideas; it is not an official CBSE exam paper and schools may use different patterns.</p>}
          </section>

          {detail.chapterSample && <section aria-labelledby="chapter-sample" className="mt-12">
            <h2 id="chapter-sample" className="text-xl font-bold">Chapters this paper samples</h2>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-neutral-600">One or more questions from each Class {paper.classLevel} {paper.subject} chapter. This is selective concept sampling, not exhaustive syllabus coverage.</p>
            <ul className="mt-4 grid gap-2 sm:grid-cols-2">
              {detail.chapterSample.map(c => <li key={c.chapter} className="flex items-center justify-between gap-3 rounded-xl border border-neutral-200 bg-white px-4 py-2.5 text-sm"><span className="font-semibold text-dark">{c.chapter}</span><span className="shrink-0 text-xs font-bold text-neutral-500">{c.questions} {c.questions === 1 ? 'question' : 'questions'}</span></li>)}
            </ul>
          </section>}

          <section aria-labelledby="paper-format" className="mt-12">
            <h2 id="paper-format" className="text-xl font-bold">Paper format</h2>
            <div className="mt-4 overflow-hidden rounded-2xl border border-neutral-200 bg-white">
              {detail.sections.map((s, i) => <div key={s.name} className={`px-5 py-4 ${i > 0 ? 'border-t border-neutral-100' : ''}`}>
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <h3 className="text-sm font-bold text-dark">{s.name}</h3>
                  <p className="text-xs font-bold text-neutral-500">{s.questions} questions · {s.marks} marks</p>
                </div>
                <p className="mt-1 text-sm leading-6 text-neutral-600">{s.detail}</p>
              </div>)}
            </div>
            <p className="mt-3 text-xs leading-5 text-neutral-500">Suggested time: {paper.minutes} minutes. The suggested time is an Evater practice choice, not an official board rule.{paper.format === 'mixed' ? ' Attempt all 24 questions; there is no internal choice.' : ''}</p>
          </section>

          <section aria-labelledby="answers-guidance" className="mt-12">
            <h2 id="answers-guidance" className="text-xl font-bold">Answers and marking guidance</h2>
            <p className="mt-3 max-w-2xl text-sm leading-7 text-neutral-600">{detail.answersNote}</p>
          </section>

          <section aria-labelledby="how-to-use" className="mt-12">
            <h2 id="how-to-use" className="text-xl font-bold">How to use this paper</h2>
            <ol className="mt-4 list-decimal space-y-2 pl-5 text-sm leading-7 text-neutral-600">
              <li>Download the PDF and set aside the suggested {paper.minutes} minutes. No account is needed.</li>
              <li>Attempt every question before opening the answer pages{paper.subject === 'Mathematics' ? ', and write your working' : ''}.</li>
              <li>Check the model answers and marking guidance, then retry the questions you missed without looking.</li>
            </ol>
            {detail.studyGuide && <p className="mt-4 text-sm leading-7 text-neutral-600">Want an explanation first? <Link to={detail.studyGuide.href} className="font-bold text-primary-700 underline underline-offset-4">{detail.studyGuide.label}</Link> for a worked refresher on this chapter.</p>}
          </section>

          <section aria-labelledby="source-note" className="mt-12 max-w-3xl">
            <h2 id="source-note" className="text-sm font-bold uppercase tracking-wide text-neutral-500">Source and originality</h2>
            <p className="mt-2 text-sm leading-7 text-neutral-600">{detail.sourceNote}</p>
          </section>
        </div>

        <aside className="min-w-0 lg:pt-2">
          <div className={`relative flex h-44 items-center justify-between overflow-hidden rounded-2xl border border-neutral-200 px-6 sm:h-52 ${paper.subject === 'Science' ? 'bg-[#edf5f1]' : 'bg-[#f6eedc]'}`} aria-hidden="true">
            <div className="z-10"><p className="text-[10px] font-bold uppercase tracking-[0.15em] text-neutral-500">Evater · Class {paper.classLevel}</p><p className="mt-3 max-w-40 text-2xl font-bold leading-tight text-[#173b38]">{paper.topic}</p><p className="mt-2 text-xs text-neutral-600">Practice. Check. Improve.</p></div>
            <div className="absolute -right-3 top-6 rotate-12 rounded-xl border border-white bg-white/70 p-6 shadow-sm"><FileText className="h-20 w-16 text-primary-700/50" strokeWidth={1} /></div>
          </div>

          <div className="mt-6 rounded-2xl border border-neutral-200 bg-white p-5">
            <h2 className="text-sm font-bold text-dark">More Class {paper.classLevel} {paper.subject} papers</h2>
            <ul className="mt-3 space-y-2">
              {related.map(({ d, p }) => <li key={d.slug}><Link to={d.path} className="group flex items-center justify-between gap-2 rounded-lg px-2 py-2 text-sm font-semibold text-primary-800 hover:bg-primary-50">
                <span className="min-w-0 truncate">{p.format === 'mixed' ? p.title : p.topic}</span>
                <ArrowUpRight aria-hidden="true" className="h-4 w-4 shrink-0 text-neutral-400 group-hover:text-primary-700" />
              </Link></li>)}
            </ul>
            <Link to={collection.path} className="mt-3 inline-flex min-h-11 items-center gap-1.5 px-2 text-sm font-bold text-primary-700 hover:underline underline-offset-4">Browse all {collection.label} papers</Link>
          </div>

          {detail.studyGuide && <Link to={detail.studyGuide.href} className="mt-6 flex items-center gap-3 rounded-2xl border border-primary-100 bg-primary-50 p-5 hover:bg-primary-100">
            <BookOpen aria-hidden="true" className="h-6 w-6 shrink-0 text-primary-700" />
            <span><span className="block text-sm font-bold text-primary-900">Revise the chapter first</span><span className="mt-0.5 block text-xs leading-5 text-primary-800">{detail.studyGuide.label} with worked explanations and quick checks.</span></span>
          </Link>}

          <p className="mt-6 border-l-2 border-primary-200 pl-4 text-xs leading-6 text-neutral-500">Original Evater practice paper, not an official NCERT paper or marking scheme. For self-study; downloading does not start a test or provide automatic grading.</p>
        </aside>
      </div>
    </main>
    {!user && <Footer />}
  </div>
}
