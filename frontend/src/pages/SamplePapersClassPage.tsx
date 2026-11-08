import React, { useEffect, useState } from 'react'
import { ArrowLeft, ArrowRight, FlaskConical, Sigma } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Header } from '../components/layout/Header'
import { Footer } from '../components/layout/Footer'
import { Seo } from '../components/seo/Seo'
import { Card, CardContent } from '../components/ui/Card'
import { getSamplePapersIndex, SamplePapersIndex } from '../lib/api'

const GRADE = 8
const MIN_PUBLISHED = 15

const subjectIcons: Record<string, React.ReactNode> = {
  science: <FlaskConical className="h-6 w-6" />,
  mathematics: <Sigma className="h-6 w-6" />,
}

export function SamplePapersClassPage() {
  const [index, setIndex] = useState<SamplePapersIndex | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    getSamplePapersIndex(GRADE)
      .then((data) => {
        if (!cancelled) setIndex(data)
      })
      .catch(() => {
        if (!cancelled) setError('Sample papers could not be loaded right now. Please try again shortly.')
      })
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <div className="flex min-h-screen flex-col bg-cream font-sans">
      <Seo
        title="Class 8 Sample Papers - Science and Maths (NCERT) - Evater"
        description="Chapter-wise Class 8 sample papers with answer keys for the new NCERT books: Curiosity (Science) and Ganita Prakash (Mathematics)."
        path="/sample-papers/class-8"
        robots={index && !error ? undefined : 'noindex'}
      />
      <Header />
      <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-10 sm:px-6 sm:py-14">
        <Link to="/sample-papers" className="mb-6 inline-flex items-center text-sm font-medium text-neutral-500 hover:text-primary-600">
          <ArrowLeft className="mr-1 h-4 w-4" /> All sample papers
        </Link>
        <h1 className="text-3xl font-bold text-dark sm:text-4xl">Class 8 Sample Papers</h1>
        <p className="mt-3 max-w-2xl text-base leading-7 text-neutral-600">
          Pick a subject to see a paper for every chapter, each with its answer key.
        </p>

        {error && <p className="mt-8 rounded-lg bg-red-50 p-4 text-sm text-red-700">{error}</p>}
        {!error && !index && <p className="mt-8 text-sm text-neutral-500">Loading papers…</p>}

        {index && (
          <div className="mt-8 grid grid-cols-1 gap-6 sm:grid-cols-2">
            {index.subjects.map((subject) => {
              const available = subject.chapters.filter((c) => c.published_count >= MIN_PUBLISHED).length
              return (
                <Link key={subject.subject} to={`/sample-papers/class-8/${subject.subject}`} className="group">
                  <Card className="h-full transition-shadow group-hover:shadow-md">
                    <CardContent className="flex h-full flex-col p-6">
                      <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary-100 text-primary-600">
                        {subjectIcons[subject.subject] ?? <FileTextFallback />}
                      </div>
                      <h2 className="mt-4 text-xl font-bold text-dark">Class 8 {subject.subject_name}</h2>
                      <p className="mt-2 flex-1 text-sm leading-6 text-neutral-600">
                        {available} chapter {available === 1 ? 'paper' : 'papers'} with answer keys.
                      </p>
                      <span className="mt-4 inline-flex items-center text-sm font-semibold text-primary-600">
                        View chapters
                        <ArrowRight className="ml-1 h-4 w-4 transition-transform group-hover:translate-x-1" />
                      </span>
                    </CardContent>
                  </Card>
                </Link>
              )
            })}
          </div>
        )}
      </main>
      <Footer />
    </div>
  )
}

function FileTextFallback() {
  return <Sigma className="h-6 w-6" />
}
