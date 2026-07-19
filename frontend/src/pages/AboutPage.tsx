import React from 'react'
import {
  ArrowRight,
  Brain,
  MessageSquare,
  Sparkles,
} from 'lucide-react'
import { Link } from 'react-router-dom'
import { Header } from '../components/layout/Header'
import { Footer } from '../components/layout/Footer'
import { Seo } from '../components/seo/Seo'
import { SITE_URL } from '../components/seo/seoConfig'
import { useAuthContext } from '../contexts/AuthContext'

const faqs = [
  {
    question: 'Who is Evater for?',
    answer: 'Evater is designed for students in Classes 6–10 who want a focused place to practise, understand mistakes, and build confidence across their subjects.',
  },
  {
    question: 'What makes Evater different from a question bank?',
    answer: 'A question bank gives you more questions. Evater is built around what happens after an attempt: feedback that helps you understand the mistake and decide what to do next.',
  },
  {
    question: 'Is Evater a replacement for a teacher?',
    answer: 'No. Evater is a practice and feedback aid. It is designed to help learners arrive at their teacher conversations with clearer questions and better evidence of what they need to work on.',
  },
]

const aboutDescription = 'Learn how Evater helps students in Classes 6–10 practise with purpose, understand mistakes, and improve through better feedback.'

export function AboutPage() {
  const { user } = useAuthContext()
  const primaryPath = user ? '/practice' : '/auth'
  const primaryLabel = user ? 'Start practising' : 'Experience Evater'

  const aboutSchema = {
    '@context': 'https://schema.org',
    '@graph': [
      {
        '@type': 'AboutPage',
        name: 'About Evater',
        description: aboutDescription,
        url: `${SITE_URL}/about`,
        mainEntity: { '@id': `${SITE_URL}/#organization` },
      },
      {
        '@type': 'Organization',
        '@id': `${SITE_URL}/#organization`,
        name: 'Evater',
        url: SITE_URL,
        logo: `${SITE_URL}/Evater_logo_2.png`,
        employee: {
          '@type': 'Person',
          name: 'Lakshay Dagar',
          jobTitle: 'Builder',
        },
      },
      {
        '@type': 'FAQPage',
        mainEntity: faqs.map((faq) => ({
          '@type': 'Question',
          name: faq.question,
          acceptedAnswer: { '@type': 'Answer', text: faq.answer },
        })),
      },
    ],
  }

  return (
    <div className="min-h-screen bg-cream font-sans">
      <Seo
        title="About Evater | Feedback-first learning for Classes 6–10"
        description={aboutDescription}
        path="/about"
        jsonLd={aboutSchema}
        keywords={['about Evater', 'feedback-first learning', 'Class 6-10 practice', 'student learning platform']}
      />
      <Header />

      <main>
        <section className="relative isolate overflow-hidden border-b border-neutral-200 bg-[#173b38] text-white">
          <div className="absolute -right-28 -top-28 h-80 w-80 rounded-full border-[36px] border-secondary-300/20" aria-hidden="true" />
          <div className="absolute -bottom-36 left-[-4rem] h-80 w-80 rounded-full bg-primary-400/10 blur-2xl" aria-hidden="true" />
          <div className="relative mx-auto grid max-w-7xl gap-12 px-4 py-16 sm:px-6 sm:py-20 lg:grid-cols-[1.05fr_0.95fr] lg:items-center lg:px-8 lg:py-24">
            <div>
              <p className="mb-6 inline-flex items-center gap-2 rounded-full border border-secondary-300/30 bg-secondary-300/10 px-3 py-1.5 text-xs font-extrabold uppercase tracking-[0.16em] text-secondary-200">
                <Sparkles className="h-3.5 w-3.5" aria-hidden="true" /> About Evater
              </p>
              <h1 className="max-w-3xl text-4xl font-extrabold leading-[1.05] tracking-[-0.04em] sm:text-5xl lg:text-7xl">
                Practice should tell you what to do next.
              </h1>
              <p className="mt-6 max-w-2xl text-base leading-8 text-primary-50/75 sm:text-lg">
                Evater is a feedback-first learning workspace for Classes 6–10. It helps students move from attempting questions to understanding mistakes and choosing a better next step.
              </p>
              <div className="mt-8 flex flex-wrap gap-3">
                <Link to={primaryPath} className="inline-flex min-h-12 items-center justify-center rounded-xl bg-secondary-300 px-5 text-sm font-extrabold text-dark transition-colors hover:bg-secondary-200 focus:outline-none focus:ring-2 focus:ring-secondary-200 focus:ring-offset-2 focus:ring-offset-[#173b38]">
                  {primaryLabel} <ArrowRight className="ml-2 h-4 w-4" aria-hidden="true" />
                </Link>
                <Link to="/blog" className="inline-flex min-h-12 items-center justify-center rounded-xl border border-white/30 px-5 text-sm font-extrabold text-white transition-colors hover:bg-white/10 focus:outline-none focus:ring-2 focus:ring-white">
                  Read the learning journal
                </Link>
              </div>
            </div>

            <div className="rounded-3xl border border-white/15 bg-white/10 p-4 shadow-2xl shadow-black/10 backdrop-blur-sm sm:p-6" aria-label="The Evater learning loop">
              <div className="rounded-2xl border border-white/10 bg-dark/30 p-5 sm:p-6">
                <div className="flex items-center justify-between gap-4 border-b border-white/10 pb-5">
                  <div>
                    <p className="text-xs font-extrabold uppercase tracking-[0.16em] text-secondary-200">The Evater loop</p>
                    <p className="mt-2 text-lg font-extrabold text-white">Every attempt leaves you clearer.</p>
                  </div>
                  <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary-400/15 text-primary-200">
                    <Brain className="h-5 w-5" aria-hidden="true" />
                  </span>
                </div>
                <div className="mt-5 space-y-3">
                  {[
                    ['01', 'Attempt', 'Try a focused question set.'],
                    ['02', 'Understand', 'See the reasoning behind the result.'],
                    ['03', 'Improve', 'Choose the next concept to practise.'],
                  ].map(([number, title, description], index) => (
                    <div key={title} className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/5 p-3">
                      <span className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl text-xs font-extrabold ${index === 2 ? 'bg-secondary-300 text-dark' : 'bg-primary-300/15 text-primary-100'}`}>{number}</span>
                      <div>
                        <p className="text-sm font-extrabold text-white">{title}</p>
                        <p className="mt-0.5 text-xs leading-5 text-primary-50/60">{description}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>

        <section aria-labelledby="mission-heading" className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8 lg:py-24">
          <div className="grid gap-12 lg:grid-cols-[0.8fr_1.2fr] lg:items-start lg:gap-20">
            <div>
              <p className="mb-3 text-xs font-extrabold uppercase tracking-[0.16em] text-primary-700">Why Evater exists</p>
              <h2 id="mission-heading" className="text-3xl font-extrabold leading-tight tracking-tight text-dark sm:text-4xl">I wanted to make testing feel useful — and fun.</h2>
            </div>
            <div className="space-y-5 text-lg leading-8 text-neutral-600">
              <p>Last year, while searching for good practice resources for my younger siblings, I could not find what I was looking for. I wanted something that was fun to use, helped them test what they knew, and gave them useful feedback afterwards.</p>
              <p>I also noticed that children often enjoyed watching lessons more than taking tests. Tests felt scary, so I wanted to reverse that feeling and make testing a natural, encouraging part of learning.</p>
              <p>There is plenty of content to teach, but not enough opportunity to test, reflect, and get meaningful feedback. I am building Evater slowly and steadily to help close that gap. With AI, we can give students an almost endless supply of unique, personalised practice.</p>
              <p className="font-bold text-dark">Evater is my attempt to build the full learning loop: try, understand, improve, and come back stronger.</p>
            </div>
          </div>

        </section>

        <section aria-labelledby="mission-vision-heading" className="border-y border-neutral-200 bg-white">
          <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8 lg:py-24">
            <div className="grid gap-12 lg:grid-cols-[0.8fr_1.2fr] lg:items-start lg:gap-20">
              <div>
                <p className="mb-3 text-xs font-extrabold uppercase tracking-[0.16em] text-primary-700">Our mission</p>
                <h2 id="mission-vision-heading" className="text-3xl font-extrabold leading-tight tracking-tight text-dark sm:text-4xl">Knowledge is free. Testing should be too.</h2>
              </div>
              <div className="space-y-5 text-lg leading-8 text-neutral-600">
                <p><span className="font-extrabold text-dark">Reading is not the same as understanding.</span> Students should discover that early, in a way that feels safe, accessible, and worth returning to.</p>
                <p>Testing should not be something to fear or something reserved for the few. It should be a free, everyday part of learning: a chance to try, fail, learn, and try again.</p>
                <p>We are building Evater to make meaningful practice and feedback accessible to every student, so testing becomes a tool for confidence rather than a barrier to it.</p>
                <p className="font-bold text-dark">Practice makes progress. Let us make the practice accessible.</p>
              </div>
            </div>
          </div>
        </section>

        <section aria-labelledby="builder-heading" className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8 lg:py-24">
          <div className="rounded-3xl border border-neutral-200 bg-primary-50 p-6 sm:p-10 lg:p-12">
            <div className="grid gap-10 lg:grid-cols-[0.7fr_1.3fr] lg:items-center lg:gap-16">
              <div className="flex flex-col items-center text-center lg:items-start lg:text-left">
                <div className="relative flex min-h-[23rem] w-full max-w-md items-end justify-center overflow-hidden rounded-[2rem] border border-primary-900/10 bg-[#173b38] p-3 shadow-xl shadow-primary-900/10 sm:min-h-[30rem]">
                  <div className="absolute -right-12 -top-16 h-48 w-48 rounded-full border-[24px] border-secondary-300/30" aria-hidden="true" />
                  <div className="absolute -bottom-16 -left-12 h-48 w-48 rounded-full bg-primary-300/15 blur-2xl" aria-hidden="true" />
                  <div className="absolute inset-4 rounded-[1.5rem] border border-white/15" aria-hidden="true" />
                  <div className="absolute left-7 top-7 z-10 rounded-full border border-white/15 bg-white/10 px-3 py-1 text-[10px] font-extrabold uppercase tracking-[0.16em] text-secondary-200 backdrop-blur-sm">Builder / Evater</div>
                  <img
                    src="/builder-lakshay.png"
                    alt="Lakshay Dagar, builder of Evater"
                    className="relative z-10 h-[22rem] w-auto max-w-[82%] object-contain object-bottom drop-shadow-2xl sm:h-[28rem]"
                    width="433"
                    height="577"
                    loading="lazy"
                    decoding="async"
                  />
                  <div className="absolute bottom-6 right-6 z-10 rounded-xl border border-white/15 bg-dark/50 px-3 py-2 text-right text-[10px] font-bold text-white/80 backdrop-blur-sm">
                    <span className="block text-secondary-200">feedback-first</span>
                    <span className="block text-white/60">built steadily</span>
                  </div>
                </div>
                <p className="mt-5 text-xs font-extrabold uppercase tracking-[0.16em] text-primary-700">Built with care</p>
                <p className="mt-2 text-sm font-semibold text-neutral-600">Builder · IIT Delhi, Class of 2024</p>
              </div>
              <div>
                <p className="mb-3 text-xs font-extrabold uppercase tracking-[0.16em] text-primary-700">Meet the builder</p>
                <h2 id="builder-heading" className="text-3xl font-extrabold tracking-tight text-dark sm:text-4xl">A small team, focused on the feedback gap.</h2>
                <p className="mt-5 text-lg leading-8 text-neutral-600">I&apos;m Lakshay Dagar. I&apos;m building Evater because the difference between “I studied this” and “I know what to improve” is where a lot of learning gets lost.</p>
                <p className="mt-4 leading-7 text-neutral-600">The product is being built steadily, with a simple belief: every learner should leave practice with a clearer next step.</p>
                <Link to={user ? '/general-feedback' : '/auth'} className="mt-6 inline-flex min-h-11 items-center gap-2 rounded-xl px-2 text-sm font-extrabold text-primary-700 hover:bg-white focus:outline-none focus:ring-2 focus:ring-primary-500"><MessageSquare className="h-4 w-4" aria-hidden="true" /> Share feedback with Evater</Link>
              </div>
            </div>
          </div>
        </section>

        <section aria-labelledby="faq-heading" className="border-t border-neutral-200 bg-white">
          <div className="mx-auto max-w-4xl px-4 py-16 sm:px-6 lg:py-20">
            <div className="text-center">
              <p className="mb-3 text-xs font-extrabold uppercase tracking-[0.16em] text-primary-700">A few answers</p>
              <h2 id="faq-heading" className="text-3xl font-extrabold tracking-tight text-dark sm:text-4xl">What Evater is — and is not</h2>
            </div>
            <div className="mt-10 space-y-4">
              {faqs.map((faq) => (
                <details key={faq.question} className="group rounded-2xl border border-neutral-200 bg-cream p-5 sm:p-6">
                  <summary className="cursor-pointer list-none pr-8 font-extrabold text-dark marker:hidden group-open:text-primary-700">{faq.question}</summary>
                  <p className="mt-3 leading-7 text-neutral-600">{faq.answer}</p>
                </details>
              ))}
            </div>
          </div>
        </section>

        <section className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8 lg:py-20" aria-labelledby="trust-heading">
          <div className="mb-8">
            <p className="mb-3 text-xs font-extrabold uppercase tracking-[0.16em] text-primary-700">Trust &amp; contact</p>
            <h2 id="trust-heading" className="text-2xl font-extrabold tracking-tight text-dark">A clear product deserves clear expectations.</h2>
          </div>
          <div className="grid grid-cols-1 gap-5 md:grid-cols-3">
            <div id="privacy" className="scroll-mt-24 rounded-2xl border border-neutral-200 bg-white p-6">
              <h3 className="text-lg font-extrabold text-dark">Privacy overview</h3>
              <p className="mt-2 text-sm leading-7 text-neutral-600">Evater uses account and practice information to provide the learning experience, save progress, and improve feedback. Contact us if you have a question about your data.</p>
            </div>
            <div id="terms" className="scroll-mt-24 rounded-2xl border border-neutral-200 bg-white p-6">
              <h3 className="text-lg font-extrabold text-dark">Terms overview</h3>
              <p className="mt-2 text-sm leading-7 text-neutral-600">Use Evater for personal learning, keep your account secure, and treat generated feedback as a study aid rather than a substitute for teacher guidance.</p>
            </div>
            <div id="cookies" className="scroll-mt-24 rounded-2xl border border-neutral-200 bg-white p-6">
              <h3 className="text-lg font-extrabold text-dark">Cookie overview</h3>
              <p className="mt-2 text-sm leading-7 text-neutral-600">The app may use essential browser storage to keep sessions and preferences working. If you have a question, share it through the <Link className="font-bold text-primary-700 hover:underline" to={user ? '/general-feedback' : '/auth'}>feedback form</Link>.</p>
            </div>
          </div>
        </section>
      </main>
      {!user && <Footer />}
    </div>
  )
}
