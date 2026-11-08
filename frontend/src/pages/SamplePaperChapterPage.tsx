import React, { useEffect, useMemo, useState } from 'react'
import { ArrowLeft, ArrowRight, CheckCircle2 } from 'lucide-react'
import { Link, useParams } from 'react-router-dom'
import { Header } from '../components/layout/Header'
import { Footer } from '../components/layout/Footer'
import { Seo } from '../components/seo/Seo'
import { Card, CardContent } from '../components/ui/Card'
import {
  getSamplePaper,
  getSamplePapersIndex,
  SamplePaper,
  SamplePaperChapterSummary,
} from '../lib/api'

const GRADE = 8
const MIN_PUBLISHED = 15

const typeLabels: Record<string, string> = {
  true_false: 'True / False',
  numerical: 'Numerical',
  fill_blank: 'Fill in the blank',
  mcq_multi: 'Multiple answers',
  assertion_reason: 'Assertion - Reason',
  case_study: 'Case study',
  diagram_based: 'Diagram based',
  matching: 'Matching',
}

function displayTitle(title: string): string {
  return title.replace(/:\s+/g, ' - ')
}

function metaTitle(paper: SamplePaper): string {
  const plain = paper.chapter.title.replace(/[^a-zA-Z0-9\s-]/g, '').replace(/\s+/g, ' ').trim()
  return `Class ${paper.grade} ${paper.subject_name} Ch ${paper.chapter.sequence_number} ${plain} Sample Paper - Evater`
}

export function SamplePaperChapterPage() {
  const { subject: subjectSlug = '', chapterSlug = '' } = useParams()
  const [paper, setPaper] = useState<SamplePaper | null>(null)
  const [siblings, setSiblings] = useState<SamplePaperChapterSummary[]>([])
  const [status, setStatus] = useState<'loading' | 'ready' | 'notfound' | 'error'>('loading')

  useEffect(() => {
    let cancelled = false
    setStatus('loading')
    setPaper(null)
    Promise.all([getSamplePaper(subjectSlug, chapterSlug, GRADE), getSamplePapersIndex(GRADE)])
      .then(([paperData, indexData]) => {
        if (cancelled) return
        setPaper(paperData)
        const subject = indexData.subjects.find((s) => s.subject === subjectSlug)
        setSiblings(
          (subject?.chapters ?? [])
            .filter((c) => c.published_count >= MIN_PUBLISHED)
            .sort((a, b) => a.sequence_number - b.sequence_number),
        )
        setStatus('ready')
      })
      .catch((err) => {
        if (cancelled) return
        setStatus(err instanceof Error && /not found|not available/i.test(err.message) ? 'notfound' : 'error')
      })
    return () => {
      cancelled = true
    }
  }, [subjectSlug, chapterSlug])

  const { previous, next } = useMemo(() => {
    const position = siblings.findIndex((c) => c.slug === chapterSlug)
    return {
      previous: position > 0 ? siblings[position - 1] : null,
      next: position >= 0 && position < siblings.length - 1 ? siblings[position + 1] : null,
    }
  }, [siblings, chapterSlug])

  const h1 = paper
    ? `Class ${paper.grade} ${paper.subject_name} Chapter ${paper.chapter.sequence_number}: ${displayTitle(paper.chapter.title)} - Sample Paper`
    : 'Sample Paper'

  return (
    <div className="flex min-h-screen flex-col bg-cream font-sans">
      <Seo
        title={paper ? metaTitle(paper) : 'Sample Paper - Evater'}
        description={
          paper
            ? `Practice Class ${paper.grade} ${paper.subject_name} Chapter ${paper.chapter.sequence_number} ${paper.chapter.title} with a free sample paper - ${paper.question_count} questions from Evater's question bank, with answer key.`
            : 'Chapter-wise sample paper with answer key from Evater.'
        }
        path={`/sample-papers/class-${GRADE}/${subjectSlug}/${chapterSlug}`}
        robots={status === 'ready' ? undefined : 'noindex'}
      />
      <Header />
      <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-10 sm:px-6 sm:py-14">
        <nav className="mb-6 flex flex-wrap items-center gap-1 text-sm text-neutral-500">
          <Link to="/sample-papers" className="hover:text-primary-600">Sample papers</Link>
          <span>/</span>
          <Link to="/sample-papers/class-8" className="hover:text-primary-600">Class 8</Link>
          <span>/</span>
          <Link to={`/sample-papers/class-8/${subjectSlug}`} className="hover:text-primary-600">
            {paper?.subject_name ?? subjectSlug}
          </Link>
        </nav>

        {status === 'loading' && <p className="text-sm text-neutral-500">Loading sample paper…</p>}
        {status === 'notfound' && (
          <p className="text-sm text-neutral-600">This sample paper is not available yet.</p>
        )}
        {status === 'error' && (
          <p className="rounded-lg bg-red-50 p-4 text-sm text-red-700">
            The sample paper could not be loaded right now. Please try again shortly.
          </p>
        )}

        {status === 'ready' && paper && (
          <>
            <h1 className="text-2xl font-bold leading-snug text-dark sm:text-3xl">{h1}</h1>
            {paper.chapter.description && (
              <p className="mt-4 text-base leading-7 text-neutral-600">{paper.chapter.description}</p>
            )}
            <p className="mt-3 text-base leading-7 text-neutral-600">
              Try these {paper.question_count} questions from Evater's question bank, then check
              the answer key at the bottom.
            </p>

            <ol className="mt-10 space-y-8">
              {paper.questions.map((question) => (
                <li key={question.id}>
                  <Card>
                    <CardContent className="p-5 sm:p-6">
                      <div className="flex items-start gap-3">
                        <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary-100 text-sm font-bold text-primary-700">
                          {question.position}
                        </span>
                        <div className="min-w-0 flex-1">
                          {typeLabels[question.question_type] && (
                            <span className="mb-2 inline-block rounded-full bg-neutral-100 px-2.5 py-0.5 text-xs font-semibold text-neutral-600">
                              {typeLabels[question.question_type]}
                            </span>
                          )}
                          <p className="text-base font-semibold leading-7 text-dark">{question.question_text}</p>
                          {question.options.length > 0 && (
                            <ul className="mt-3 space-y-2">
                              {question.options.map((option) => (
                                <li key={option.id} className="flex items-start gap-2 text-sm leading-6 text-neutral-700">
                                  <span className="font-semibold text-neutral-500">{option.id}.</span>
                                  <span>{option.text}</span>
                                </li>
                              ))}
                            </ul>
                          )}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </li>
              ))}
            </ol>

            <div className="mt-12 rounded-2xl bg-primary-600 p-6 text-center sm:p-8">
              <h2 className="text-xl font-bold text-white sm:text-2xl">Practice this chapter on Evater</h2>
              <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-primary-100">
                Adaptive tests, instant feedback and explanations for every mistake. Free to try.
              </p>
              <Link
                to="/practice"
                className="mt-5 inline-flex items-center rounded-xl bg-white px-6 py-3 text-sm font-bold text-primary-700 transition-colors hover:bg-primary-50"
              >
                Start practising
                <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </div>

            <section className="mt-12 rounded-2xl border border-neutral-200 bg-white p-6 sm:p-8">
              <div className="mb-4 flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-primary-600" />
                <h2 className="text-xl font-bold text-dark">Answer Key</h2>
              </div>
              <ol className="grid grid-cols-1 gap-x-8 gap-y-2 sm:grid-cols-2">
                {paper.questions.map((question) => (
                  <li key={question.id} className="text-sm leading-6 text-neutral-700">
                    <span className="font-semibold text-dark">{question.position}.</span> {question.answer_summary}
                  </li>
                ))}
              </ol>
            </section>

            <div className="mt-10 flex items-center justify-between gap-4">
              {previous ? (
                <Link
                  to={`/sample-papers/class-8/${subjectSlug}/${previous.slug}`}
                  className="inline-flex items-center text-sm font-medium text-neutral-500 hover:text-primary-600"
                >
                  <ArrowLeft className="mr-1 h-4 w-4" />
                  <span className="max-w-40 truncate sm:max-w-none">Ch {previous.sequence_number}: {previous.title}</span>
                </Link>
              ) : (
                <span />
              )}
              {next ? (
                <Link
                  to={`/sample-papers/class-8/${subjectSlug}/${next.slug}`}
                  className="inline-flex items-center text-right text-sm font-medium text-neutral-500 hover:text-primary-600"
                >
                  <span className="max-w-40 truncate sm:max-w-none">Ch {next.sequence_number}: {next.title}</span>
                  <ArrowRight className="ml-1 h-4 w-4" />
                </Link>
              ) : (
                <span />
              )}
            </div>
          </>
        )}
      </main>
      <Footer />
    </div>
  )
}
