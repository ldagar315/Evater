import React, { useEffect, useState } from 'react'
import { ArrowLeft, ArrowRight } from 'lucide-react'
import { Link, useParams } from 'react-router-dom'
import { Header } from '../components/layout/Header'
import { Footer } from '../components/layout/Footer'
import { Seo } from '../components/seo/Seo'
import { Card, CardContent } from '../components/ui/Card'
import { getSamplePapersIndex, SamplePapersSubject } from '../lib/api'

const GRADE = 8
const MIN_PUBLISHED = 15

export function SamplePapersSubjectPage() {
  const { subject: subjectSlug = '' } = useParams()
  const [subject, setSubject] = useState<SamplePapersSubject | null>(null)
  const [notFound, setNotFound] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    getSamplePapersIndex(GRADE)
      .then((data) => {
        if (cancelled) return
        const match = data.subjects.find((s) => s.subject === subjectSlug)
        if (match) setSubject(match)
        else setNotFound(true)
      })
      .catch(() => {
        if (!cancelled) setError('Sample papers could not be loaded right now. Please try again shortly.')
      })
    return () => {
      cancelled = true
    }
  }, [subjectSlug])

  const subjectName = subject?.subject_name ?? subjectSlug
  const available = subject?.chapters.filter((c) => c.published_count >= MIN_PUBLISHED) ?? []

  return (
    <div className="flex min-h-screen flex-col bg-cream font-sans">
      <Seo
        title={`Class 8 ${subjectName} Sample Papers - All Chapters - Evater`}
        description={`Chapter-wise Class 8 ${subjectName} sample papers with answer keys, drawn from Evater's question bank.`}
        path={`/sample-papers/class-8/${subjectSlug}`}
        robots={subject && !error && !notFound ? undefined : 'noindex'}
      />
      <Header />
      <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-10 sm:px-6 sm:py-14">
        <Link to="/sample-papers/class-8" className="mb-6 inline-flex items-center text-sm font-medium text-neutral-500 hover:text-primary-600">
          <ArrowLeft className="mr-1 h-4 w-4" /> Class 8 sample papers
        </Link>
        <h1 className="text-3xl font-bold text-dark sm:text-4xl">Class 8 {subjectName} Sample Papers</h1>

        {error && <p className="mt-8 rounded-lg bg-red-50 p-4 text-sm text-red-700">{error}</p>}
        {notFound && !error && (
          <p className="mt-8 text-sm text-neutral-600">No sample papers for this subject yet.</p>
        )}
        {!error && !notFound && !subject && <p className="mt-8 text-sm text-neutral-500">Loading chapters…</p>}

        {subject && (
          <div className="mt-8 grid grid-cols-1 gap-4">
            {available.map((chapter) => (
              <Link
                key={chapter.slug}
                to={`/sample-papers/class-8/${subject.subject}/${chapter.slug}`}
                className="group"
              >
                <Card className="transition-shadow group-hover:shadow-md">
                  <CardContent className="flex items-center justify-between p-5">
                    <div>
                      <h2 className="text-base font-bold text-dark sm:text-lg">
                        Chapter {chapter.sequence_number}: {chapter.title}
                      </h2>
                      <p className="mt-1 text-sm text-neutral-600">Sample paper with answer key</p>
                    </div>
                    <ArrowRight className="h-5 w-5 shrink-0 text-primary-600 transition-transform group-hover:translate-x-1" />
                  </CardContent>
                </Card>
              </Link>
            ))}
          </div>
        )}
      </main>
      <Footer />
    </div>
  )
}
