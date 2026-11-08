import React from 'react'
import { ArrowRight, FileText } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Header } from '../components/layout/Header'
import { Footer } from '../components/layout/Footer'
import { Seo } from '../components/seo/Seo'
import { Card, CardContent } from '../components/ui/Card'

const upcomingClasses = [6, 7, 9, 10]

export function SamplePapersHubPage() {
  return (
    <div className="flex min-h-screen flex-col bg-cream font-sans">
      <Seo
        title="Sample Papers for Class 6-10 (NCERT, with Answer Keys) - Evater"
        description="Free chapter-wise sample papers with answer keys. Class 8 Science and Mathematics papers live now, more classes coming soon."
        path="/sample-papers"
      />
      <Header />
      <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-10 sm:px-6 sm:py-14">
        <div className="mb-10 text-center">
          <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-primary-100 text-primary-600">
            <FileText className="h-7 w-7" />
          </div>
          <h1 className="text-3xl font-bold text-dark sm:text-4xl">Sample Papers</h1>
          <p className="mx-auto mt-3 max-w-2xl text-base leading-7 text-neutral-600 sm:text-lg">
            Chapter-wise sample papers drawn from Evater's question bank, each with an
            answer key at the bottom. Practise on paper, then try the chapter in the app.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          <Link to="/sample-papers/class-8" className="group">
            <Card className="h-full transition-shadow group-hover:shadow-md">
              <CardContent className="flex h-full flex-col p-6">
                <p className="text-sm font-semibold uppercase tracking-wide text-primary-600">Available now</p>
                <h2 className="mt-2 text-xl font-bold text-dark">Class 8</h2>
                <p className="mt-2 flex-1 text-sm leading-6 text-neutral-600">
                  Science and Mathematics papers for every chapter of the new NCERT books
                  (Curiosity and Ganita Prakash).
                </p>
                <span className="mt-4 inline-flex items-center text-sm font-semibold text-primary-600">
                  Browse Class 8 papers
                  <ArrowRight className="ml-1 h-4 w-4 transition-transform group-hover:translate-x-1" />
                </span>
              </CardContent>
            </Card>
          </Link>

          {upcomingClasses.map((grade) => (
            <Card key={grade} className="h-full opacity-70">
              <CardContent className="flex h-full flex-col p-6">
                <p className="text-sm font-semibold uppercase tracking-wide text-neutral-400">Coming soon</p>
                <h2 className="mt-2 text-xl font-bold text-dark">Class {grade}</h2>
                <p className="mt-2 flex-1 text-sm leading-6 text-neutral-600">
                  Papers arrive here as the Class {grade} question bank grows.
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      </main>
      <Footer />
    </div>
  )
}
