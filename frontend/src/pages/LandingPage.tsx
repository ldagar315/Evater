import { ArrowRight, BookOpen, Brain, CheckCircle2, ChevronRight, CircleDot, Lightbulb, MessageCircle, PenLine, Quote, RotateCcw, Sparkles, Star, Target, Trophy, Users } from 'lucide-react'
import { Link } from 'react-router-dom'
import { useAuthContext } from '../contexts/AuthContext'
import { BlogSection } from '../components/blog/BlogSection'
import { Footer } from '../components/layout/Footer'
import { Header } from '../components/layout/Header'
import { Seo } from '../components/seo/Seo'
import { SITE_URL } from '../components/seo/seoConfig'

const steps = [
  {
    number: '01',
    icon: Target,
    title: 'Choose what to focus on',
    description: 'Start with a subject, chapter, or question type that needs your attention today.',
  },
  {
    number: '02',
    icon: PenLine,
    title: 'Make a real attempt',
    description: 'Practise with focused questions instead of scrolling through content without a plan.',
  },
  {
    number: '03',
    icon: Lightbulb,
    title: 'Leave with a next step',
    description: 'See what you understood, where you slipped, and what to revisit before you try again.',
  },
]

const learningPillars = [
  {
    icon: Brain,
    accent: 'bg-primary-50 text-primary-700',
    eyebrow: 'Practise with purpose',
    title: 'Questions that meet you where you are',
    description: 'Build a focused session around your class, subject, and the topic you want to improve—not a random worksheet.',
  },
  {
    icon: MessageCircle,
    accent: 'bg-secondary-50 text-secondary-800',
    eyebrow: 'Understand the why',
    title: 'Feedback you can actually use',
    description: 'Turn a wrong answer into a useful explanation, so you know what changed and how to approach the question next time.',
  },
  {
    icon: RotateCcw,
    accent: 'bg-purple-50 text-purple-700',
    eyebrow: 'Keep the momentum',
    title: 'A clearer path to the next attempt',
    description: 'Use each practice session to decide what to review, what to retry, and when you are ready to move on.',
  },
]

const focusAreas = ['Class 6–10 practice', 'Maths & Science', 'Chapter revision', 'Exam preparation']

const feedbackPoints = [
  {
    icon: PenLine,
    title: 'Your actual attempt stays at the centre',
    description: 'Review the response you gave, not a generic explanation disconnected from your thinking.',
  },
  {
    icon: MessageCircle,
    title: 'Feedback explains what to improve',
    description: 'See what is working, where the reasoning needs more care, and how to approach it next time.',
  },
  {
    icon: RotateCcw,
    title: 'Retry while the idea is still fresh',
    description: 'Use the review to return to the question with a stronger plan instead of simply moving on.',
  },
]

const testimonials = [
  {
    name: 'Darsh',
    grade: 'Class 10th',
    school: 'Presidium Sector-57',
    quote: 'Evater has completely transformed how I prepare for my board exams. The AI-generated tests are exactly what I need to practice, and the instant feedback helps me understand my mistakes immediately.',
    image: '/IMG-20251020-WA0056.jpg',
    objectPosition: '50% 68%',
  },
  {
    name: 'Manvi',
    grade: 'Class 7th',
    school: 'Presidium Sec-57',
    quote: 'I love how Evater makes studying fun! The tests are challenging but fair, and I can see my progress improving every day. My teachers are impressed with my performance.',
    image: '/IMG-20230912-WA0025.jpg',
    objectPosition: '50% 44%',
  },
  {
    name: 'Ishika',
    grade: 'Class 10th',
    school: 'Euro International Sector-45',
    quote: 'The AI checking feature is amazing! I can upload my handwritten answers and get detailed feedback instantly. It is like having a personal tutor available 24/7.',
    image: '/IMG_20241028_145336.jpg',
    objectPosition: '68% 38%',
  },
]

export function LandingPage() {
  const { user } = useAuthContext()
  const ctaPath = user ? '/practice' : '/auth'
  const ctaLabel = user ? 'Continue practising' : 'Try your first test'

  return (
    <div className="min-h-screen bg-cream font-sans text-dark">
      <Seo
        title="Evater | Tests made fun. Progress made visible."
        description="Evater helps students in Classes 6–10 take tests, understand feedback, and practise again without leaving the learning loop."
        path="/"
        keywords={['Class 6–10 practice', 'NCERT practice', 'study feedback', 'exam preparation', 'Evater']}
        jsonLd={{
          '@context': 'https://schema.org',
          '@graph': [
            {
              '@type': 'WebSite',
              '@id': `${SITE_URL}/#website`,
              name: 'Evater',
              url: SITE_URL,
              description: 'A focused test, feedback, and retry loop for students in Classes 6–10.',
              publisher: { '@id': `${SITE_URL}/#organization` },
            },
            {
              '@type': 'Organization',
              '@id': `${SITE_URL}/#organization`,
              name: 'Evater',
              url: SITE_URL,
              logo: `${SITE_URL}/Evater_logo_2.png`,
            },
          ],
        }}
      />

      <Header />

      <main>
        <section className="relative isolate overflow-hidden bg-[#173b38] text-white">
          <div className="absolute -left-40 -top-48 h-[34rem] w-[34rem] rounded-full border-[3rem] border-primary-400/10" aria-hidden="true" />
          <div className="absolute -right-48 top-24 h-[30rem] w-[30rem] rounded-full border-[2rem] border-secondary-300/10" aria-hidden="true" />
          <div className="absolute bottom-0 left-1/3 h-72 w-72 rounded-full bg-primary-400/10 blur-3xl" aria-hidden="true" />

          <div className="relative mx-auto grid max-w-7xl items-center gap-14 px-4 pb-20 pt-16 sm:px-6 sm:pb-24 sm:pt-20 lg:grid-cols-[minmax(0,0.9fr)_minmax(28rem,1.1fr)] lg:gap-16 lg:px-8 lg:pb-28 lg:pt-24">
            <div className="max-w-2xl">
              <div className="mb-7 inline-flex items-center gap-2 rounded-full border border-primary-200/20 bg-white/10 px-3.5 py-2 text-xs font-extrabold uppercase tracking-[0.16em] text-primary-100">
                <Sparkles className="h-4 w-4 text-secondary-300" aria-hidden="true" />
                Testing without the stress
              </div>

              <h1 className="max-w-2xl text-4xl font-extrabold leading-[1.04] tracking-[-0.04em] sm:text-6xl lg:text-[4.65rem]">
                Tests made fun.{' '}
                <span className="text-secondary-300">Progress made visible.</span>
              </h1>

              <p className="mt-7 max-w-xl text-lg leading-8 text-white/75 sm:text-xl">
                Take a test, see what you know, understand what to fix, and try again—all without leaving the learning loop. Evater keeps it in one place.
              </p>

              <div className="mt-9 flex flex-col gap-3 sm:flex-row sm:items-center">
                <Link
                  to={ctaPath}
                  className="inline-flex min-h-13 items-center justify-center gap-2 rounded-xl bg-secondary-300 px-6 py-3.5 text-base font-extrabold text-dark shadow-lg shadow-black/10 transition-colors hover:bg-secondary-200 focus:outline-none focus:ring-2 focus:ring-secondary-200 focus:ring-offset-2 focus:ring-offset-[#173b38]"
                >
                  {ctaLabel}
                  <ArrowRight className="h-5 w-5" aria-hidden="true" />
                </Link>
                <a
                  href="#how-it-works"
                  className="inline-flex min-h-13 items-center justify-center gap-2 rounded-xl border border-white/20 px-6 py-3.5 text-base font-bold text-white transition-colors hover:bg-white/10 focus:outline-none focus:ring-2 focus:ring-primary-200 focus:ring-offset-2 focus:ring-offset-[#173b38]"
                >
                  See the loop
                  <ChevronRight className="h-5 w-5" aria-hidden="true" />
                </a>
              </div>

              <div className="mt-10 flex flex-wrap gap-x-6 gap-y-3 text-sm font-semibold text-white/60">
                <span className="inline-flex items-center gap-2"><CheckCircle2 className="h-4 w-4 text-primary-300" aria-hidden="true" /> Take a test</span>
                <span className="inline-flex items-center gap-2"><CheckCircle2 className="h-4 w-4 text-primary-300" aria-hidden="true" /> See your feedback</span>
                <span className="inline-flex items-center gap-2"><CheckCircle2 className="h-4 w-4 text-primary-300" aria-hidden="true" /> Try again</span>
              </div>
            </div>

            <div className="relative mx-auto w-full max-w-xl lg:justify-self-end">
              <div className="absolute -inset-5 rounded-[2.5rem] bg-primary-300/20 blur-2xl" aria-hidden="true" />
              <div className="relative rotate-1 rounded-[2rem] border border-white/15 bg-white/10 p-3 shadow-2xl backdrop-blur-sm sm:p-4">
                <div className="overflow-hidden rounded-[1.5rem] bg-cream text-dark shadow-xl">
                  <div className="flex items-center justify-between border-b border-neutral-200 px-5 py-4 sm:px-7">
                    <div>
                      <p className="text-xs font-extrabold uppercase tracking-[0.16em] text-neutral-500">Practice review</p>
                      <p className="mt-1 text-sm font-bold text-dark">Science · Class 8</p>
                    </div>
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-primary-50 px-3 py-1.5 text-xs font-extrabold text-primary-700">
                      <CircleDot className="h-3.5 w-3.5" aria-hidden="true" /> In progress
                    </span>
                  </div>

                  <div className="space-y-5 p-5 sm:p-7">
                    <div>
                      <p className="text-xs font-bold uppercase tracking-[0.14em] text-primary-700">Question 4 of 8</p>
                      <h2 className="mt-3 text-xl font-extrabold leading-snug text-dark sm:text-2xl">Why do we see different phases of the Moon?</h2>
                    </div>

                    <div className="rounded-2xl border border-neutral-200 bg-white p-4 text-sm leading-6 text-neutral-600 sm:p-5">
                      The Moon appears different because we see changing amounts of its sunlit half as it moves around Earth.
                    </div>

                    <div className="rounded-2xl border border-primary-200 bg-primary-50/70 p-4 sm:p-5">
                      <div className="flex items-start gap-3">
                        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-primary-500 text-white">
                          <Lightbulb className="h-4 w-4" aria-hidden="true" />
                        </div>
                        <div>
                          <p className="text-sm font-extrabold text-primary-900">You have the core idea</p>
                          <p className="mt-1 text-sm leading-6 text-primary-900/70">Next, add the role of the Sun’s light to make your explanation complete.</p>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center justify-between border-t border-neutral-200 pt-4">
                      <span className="text-sm font-bold text-neutral-500">One small improvement</span>
                      <span className="inline-flex items-center gap-1 text-sm font-extrabold text-primary-700">Keep going <ArrowRight className="h-4 w-4" aria-hidden="true" /></span>
                    </div>
                  </div>
                </div>
              </div>
              <div className="absolute -bottom-5 -left-5 hidden rounded-2xl border border-white/15 bg-[#234b47] px-4 py-3 shadow-xl sm:block">
                <p className="text-xs font-bold text-white/60">Your next review</p>
                <p className="mt-1 text-sm font-extrabold text-secondary-200">Sun, Earth & Moon</p>
              </div>
            </div>
          </div>
        </section>

        <section className="border-b border-neutral-200 bg-white" aria-label="Evater focus areas">
          <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-3 px-4 py-6 sm:px-6 lg:px-8">
            <p className="text-sm font-extrabold uppercase tracking-[0.14em] text-neutral-500">Made for focused learning</p>
            <div className="flex flex-wrap gap-2">
              {focusAreas.map((area) => (
                <span key={area} className="rounded-full border border-neutral-200 bg-cream px-3.5 py-1.5 text-sm font-bold text-neutral-700">{area}</span>
              ))}
            </div>
          </div>
        </section>

        <section id="how-it-works" className="scroll-mt-24 bg-cream py-20 sm:py-24">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div className="max-w-3xl">
              <p className="text-sm font-extrabold uppercase tracking-[0.16em] text-primary-700">The Evater loop</p>
              <h2 className="mt-4 text-3xl font-extrabold leading-tight tracking-[-0.03em] text-dark sm:text-5xl">Take it. Get it. Try again.</h2>
              <p className="mt-5 text-lg leading-8 text-neutral-600">Everything you need to complete the loop stays in one place: make an attempt, learn from the feedback, and come back stronger.</p>
            </div>

            <div className="mt-12 grid gap-5 md:grid-cols-3">
              {steps.map((step) => {
                const Icon = step.icon
                return (
                  <article key={step.number} className="relative rounded-3xl border border-neutral-200 bg-white p-7 shadow-sm transition-transform hover:-translate-y-1 hover:shadow-lg sm:p-8">
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-dark text-secondary-300">
                        <Icon className="h-5 w-5" aria-hidden="true" />
                      </div>
                      <span className="text-4xl font-extrabold tracking-[-0.06em] text-neutral-200">{step.number}</span>
                    </div>
                    <h3 className="mt-8 text-xl font-extrabold text-dark">{step.title}</h3>
                    <p className="mt-3 leading-7 text-neutral-600">{step.description}</p>
                  </article>
                )
              })}
            </div>
          </div>
        </section>

        <section id="leaderboard" className="scroll-mt-24 overflow-hidden bg-white py-20 sm:py-24">
          <div className="mx-auto grid max-w-7xl items-center gap-12 px-4 sm:px-6 lg:grid-cols-[0.9fr_1.1fr] lg:gap-20 lg:px-8">
            <div className="max-w-xl">
              <div className="inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-secondary-50 text-secondary-700">
                <Trophy className="h-6 w-6" aria-hidden="true" />
              </div>
              <p className="mt-7 text-sm font-extrabold uppercase tracking-[0.16em] text-secondary-700">Make progress social</p>
              <h2 className="mt-4 text-3xl font-extrabold leading-tight tracking-[-0.03em] text-dark sm:text-5xl">Compete with your classmates. Become your school champion.</h2>
              <p className="mt-5 text-lg leading-8 text-neutral-600">Practice feels different when your progress has a little energy behind it. See how you rank in your classroom or school, cheer on your classmates, and keep climbing.</p>
              <Link to={ctaPath} className="mt-8 inline-flex min-h-12 items-center gap-2 rounded-xl bg-dark px-5 py-3 text-sm font-extrabold text-white transition-colors hover:bg-neutral-800 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2">
                {user ? 'View leaderboard' : 'Join the leaderboard'}
                <ArrowRight className="h-4 w-4" aria-hidden="true" />
              </Link>
            </div>

            <div className="relative mx-auto w-full max-w-xl">
              <div className="absolute -inset-5 rounded-[2.5rem] bg-secondary-300/20 blur-2xl" aria-hidden="true" />
              <div className="relative overflow-hidden rounded-[2rem] border border-neutral-200 bg-cream p-4 shadow-xl sm:p-6">
                <div className="rounded-[1.5rem] bg-[#173b38] p-5 text-white sm:p-7">
                  <div className="flex items-start justify-between gap-4 border-b border-white/10 pb-5">
                    <div>
                      <p className="text-xs font-extrabold uppercase tracking-[0.16em] text-secondary-200">Leaderboard</p>
                      <h3 className="mt-2 text-2xl font-extrabold">My school</h3>
                      <p className="mt-1 text-sm font-semibold text-white/55">Last 7 days</p>
                    </div>
                    <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-secondary-300 text-dark">
                      <Trophy className="h-6 w-6" aria-hidden="true" />
                    </div>
                  </div>

                  <div className="mt-5 space-y-3" aria-label="Leaderboard preview">
                    <div className="flex items-center gap-3 rounded-2xl border border-secondary-300/40 bg-secondary-300/15 p-4">
                      <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-secondary-300 text-sm font-extrabold text-dark">#1</span>
                      <div className="min-w-0 flex-1">
                        <p className="font-extrabold">School champion</p>
                        <p className="mt-1 text-xs font-semibold text-white/55">The top spot is waiting</p>
                      </div>
                      <Trophy className="h-5 w-5 shrink-0 text-secondary-300" aria-hidden="true" />
                    </div>
                    <div className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/5 p-4">
                      <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-white/10 text-sm font-extrabold text-white">#2</span>
                      <div className="min-w-0 flex-1">
                        <p className="font-extrabold">Classroom challenger</p>
                        <p className="mt-1 text-xs font-semibold text-white/55">One more test can change the board</p>
                      </div>
                      <Users className="h-5 w-5 shrink-0 text-primary-300" aria-hidden="true" />
                    </div>
                    <div className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/5 p-4">
                      <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-white/10 text-sm font-extrabold text-white">#3</span>
                      <div className="min-w-0 flex-1">
                        <p className="font-extrabold">Your next rank</p>
                        <p className="mt-1 text-xs font-semibold text-white/55">Every completed test counts</p>
                      </div>
                      <ArrowRight className="h-5 w-5 shrink-0 text-secondary-300" aria-hidden="true" />
                    </div>
                  </div>

                  <div className="mt-5 flex items-center gap-2 border-t border-white/10 pt-5 text-sm font-bold text-white/65">
                    <Users className="h-4 w-4 text-primary-300" aria-hidden="true" />
                    Compete in your classroom or school
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section id="features" className="scroll-mt-24 bg-white py-20 sm:py-24">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div className="grid gap-12 lg:grid-cols-[0.8fr_1.2fr] lg:gap-20">
              <div className="max-w-md">
                <div className="inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-secondary-50 text-secondary-800">
                  <BookOpen className="h-6 w-6" aria-hidden="true" />
                </div>
                <p className="mt-7 text-sm font-extrabold uppercase tracking-[0.16em] text-secondary-700">Built around the learner</p>
                <h2 className="mt-4 text-3xl font-extrabold leading-tight tracking-[-0.03em] text-dark sm:text-5xl">The point is progress—not just more questions.</h2>
                <p className="mt-5 text-lg leading-8 text-neutral-600">Evater brings practice and reflection together, so every session helps you learn something about your thinking.</p>
                <Link to={ctaPath} className="mt-8 inline-flex min-h-11 items-center gap-2 rounded-xl px-1 text-sm font-extrabold text-primary-700 transition-colors hover:text-primary-800 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-4">
                  {user ? 'Open practice' : 'Try a focused session'}
                  <ArrowRight className="h-4 w-4" aria-hidden="true" />
                </Link>
              </div>

              <div className="grid gap-5 sm:grid-cols-2">
                {learningPillars.map((pillar, index) => {
                  const Icon = pillar.icon
                  return (
                    <article key={pillar.title} className={`rounded-3xl border border-neutral-200 p-6 shadow-sm sm:p-7 ${index === 2 ? 'sm:col-span-2 sm:max-w-[calc(50%-0.625rem)]' : ''}`}>
                      <div className={`flex h-11 w-11 items-center justify-center rounded-2xl ${pillar.accent}`}>
                        <Icon className="h-5 w-5" aria-hidden="true" />
                      </div>
                      <p className="mt-7 text-xs font-extrabold uppercase tracking-[0.14em] text-neutral-500">{pillar.eyebrow}</p>
                      <h3 className="mt-2 text-xl font-extrabold leading-snug text-dark">{pillar.title}</h3>
                      <p className="mt-3 leading-7 text-neutral-600">{pillar.description}</p>
                    </article>
                  )
                })}
              </div>
            </div>
          </div>
        </section>

        <section className="overflow-hidden bg-cream py-20 sm:py-24">
          <div className="mx-auto grid max-w-7xl items-center gap-12 px-4 sm:px-6 lg:grid-cols-[1.05fr_0.95fr] lg:gap-20 lg:px-8">
            <div className="relative order-2 lg:order-1">
              <div className="absolute -inset-5 rounded-[2.5rem] bg-secondary-300/20 blur-2xl" aria-hidden="true" />
              <div className="relative rounded-[2rem] border border-neutral-200 bg-white p-6 shadow-xl sm:p-8">
                <div className="flex items-center justify-between border-b border-neutral-200 pb-5">
                  <div>
                    <p className="text-xs font-extrabold uppercase tracking-[0.16em] text-neutral-500">After your attempt</p>
                    <h3 className="mt-1 text-xl font-extrabold text-dark">Make the mistake useful</h3>
                  </div>
                  <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-secondary-50 text-secondary-800">
                    <Sparkles className="h-5 w-5" aria-hidden="true" />
                  </div>
                </div>

                <div className="mt-6 space-y-3">
                  <div className="rounded-2xl border border-primary-200 bg-primary-50 p-4">
                    <div className="flex items-center gap-2 text-sm font-extrabold text-primary-800"><CheckCircle2 className="h-4 w-4" aria-hidden="true" /> What you got right</div>
                    <p className="mt-2 text-sm leading-6 text-primary-900/70">You identified the main concept and used the correct formula.</p>
                  </div>
                  <div className="rounded-2xl border border-secondary-200 bg-secondary-50 p-4">
                    <div className="flex items-center gap-2 text-sm font-extrabold text-secondary-900"><Target className="h-4 w-4" aria-hidden="true" /> What to tighten</div>
                    <p className="mt-2 text-sm leading-6 text-secondary-900/70">Review units before the final step. Try one similar question next.</p>
                  </div>
                  <div className="rounded-2xl border border-neutral-200 bg-cream p-4">
                    <div className="flex items-center gap-2 text-sm font-extrabold text-dark"><ArrowRight className="h-4 w-4" aria-hidden="true" /> Your next move</div>
                    <p className="mt-2 text-sm leading-6 text-neutral-600">Revisit the chapter example, then retry this question without looking at the hint.</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="order-1 max-w-xl lg:order-2">
              <p className="text-sm font-extrabold uppercase tracking-[0.16em] text-primary-700">Feedback that moves you forward</p>
              <h2 className="mt-4 text-3xl font-extrabold leading-tight tracking-[-0.03em] text-dark sm:text-5xl">A score tells you where you are. Feedback shows you where to go.</h2>
              <p className="mt-5 text-lg leading-8 text-neutral-600">The most valuable part of practice is knowing what to do with the result. Evater keeps that next move close at hand.</p>
              <div className="mt-8 space-y-4">
                {['Spot the gap without losing confidence', 'Review the idea behind the answer', 'Retry while the learning is still fresh'].map((item) => (
                  <div key={item} className="flex items-start gap-3">
                    <CheckCircle2 className="mt-1 h-5 w-5 shrink-0 text-primary-600" aria-hidden="true" />
                    <span className="font-bold text-neutral-700">{item}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section id="feedback" className="relative overflow-hidden bg-[#173b38] py-20 text-white sm:py-24">
          <div className="absolute -bottom-40 -left-28 h-96 w-96 rounded-full border-[3rem] border-primary-300/10" aria-hidden="true" />
          <div className="relative mx-auto grid max-w-7xl items-center gap-12 px-4 sm:px-6 lg:grid-cols-[0.85fr_1.15fr] lg:gap-20 lg:px-8">
            <div className="max-w-xl">
              <div className="inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-white/10 text-secondary-300">
                <Users className="h-6 w-6" aria-hidden="true" />
              </div>
              <p className="mt-7 text-sm font-extrabold uppercase tracking-[0.16em] text-secondary-200">Complete the loop, all in one place</p>
              <h2 className="mt-4 text-3xl font-extrabold leading-tight tracking-[-0.03em] sm:text-5xl">Test, learn, and try again without the guesswork.</h2>
              <p className="mt-5 text-lg leading-8 text-white/70">A test should do more than give you a result. Evater keeps your answer, the feedback, and your next attempt connected.</p>
            </div>

            <div className="grid gap-3 sm:grid-cols-3">
              {feedbackPoints.map((point) => {
                const Icon = point.icon
                return (
                  <article key={point.title} className="rounded-3xl border border-white/10 bg-white/10 p-5 backdrop-blur-sm sm:p-6">
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-secondary-300 text-dark">
                      <Icon className="h-5 w-5" aria-hidden="true" />
                    </div>
                    <h3 className="mt-6 text-lg font-extrabold leading-snug">{point.title}</h3>
                    <p className="mt-3 text-sm leading-6 text-white/65">{point.description}</p>
                  </article>
                )
              })}
            </div>
          </div>
        </section>

        <section className="bg-white py-20 sm:py-24">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div className="mx-auto max-w-3xl text-center">
              <p className="inline-flex items-center gap-2 text-sm font-extrabold uppercase tracking-[0.16em] text-primary-700"><Quote className="h-4 w-4" aria-hidden="true" /> From the Evater community</p>
              <h2 className="mt-4 text-3xl font-extrabold leading-tight tracking-[-0.03em] text-dark sm:text-5xl">Students are using feedback to feel more ready.</h2>
              <p className="mt-5 text-lg leading-8 text-neutral-600">Real learners. Real practice habits. Real words about what changes when mistakes become part of the learning process.</p>
            </div>

            <div className="mt-12 grid gap-5 lg:grid-cols-3">
              {testimonials.map((testimonial) => (
                <figure key={testimonial.name} className="group flex h-full flex-col overflow-hidden rounded-3xl border border-neutral-200 bg-cream shadow-sm transition-all duration-300 hover:-translate-y-1 hover:shadow-xl">
                  <div className="relative h-80 overflow-hidden bg-neutral-200 sm:h-96">
                    <img
                      src={testimonial.image}
                      alt={`${testimonial.name}, Evater learner`}
                      className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-105"
                      style={{ objectPosition: testimonial.objectPosition }}
                    />
                    <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-dark/65 via-dark/10 to-transparent px-5 pb-4 pt-16">
                      <p className="text-sm font-extrabold text-white">{testimonial.name}</p>
                      <p className="text-xs font-bold uppercase tracking-[0.08em] text-white/75">{testimonial.grade} · {testimonial.school}</p>
                    </div>
                  </div>
                  <div className="flex flex-1 flex-col p-6 sm:p-7">
                    <div className="flex items-center gap-1 text-secondary-600" aria-label="5 out of 5 stars">
                      {Array.from({ length: 5 }).map((_, index) => <Star key={index} className="h-4 w-4 fill-current" aria-hidden="true" />)}
                    </div>
                    <blockquote className="relative mt-6 flex-1 text-base leading-7 text-neutral-700">
                      <Quote className="absolute -left-1 -top-3 h-7 w-7 -scale-x-100 text-primary-200" aria-hidden="true" />
                      <span className="relative">“{testimonial.quote}”</span>
                    </blockquote>
                  </div>
                </figure>
              ))}
            </div>
          </div>
        </section>

        <BlogSection />

        <section className="relative overflow-hidden bg-[#173b38] py-20 text-white sm:py-24">
          <div className="absolute -right-24 -top-40 h-96 w-96 rounded-full border-[3rem] border-secondary-300/10" aria-hidden="true" />
          <div className="relative mx-auto max-w-4xl px-4 text-center sm:px-6 lg:px-8">
            <p className="text-sm font-extrabold uppercase tracking-[0.16em] text-secondary-200">Your next attempt starts here</p>
            <h2 className="mt-5 text-3xl font-extrabold leading-tight tracking-[-0.03em] sm:text-5xl">Make practice feel less like a test and more like progress.</h2>
            <p className="mx-auto mt-5 max-w-2xl text-lg leading-8 text-white/70">Choose one topic, make one honest attempt, and leave knowing what to do next.</p>
            <Link to={ctaPath} className="mt-9 inline-flex min-h-13 items-center justify-center gap-2 rounded-xl bg-secondary-300 px-7 py-3.5 text-base font-extrabold text-dark shadow-lg transition-colors hover:bg-secondary-200 focus:outline-none focus:ring-2 focus:ring-secondary-200 focus:ring-offset-2 focus:ring-offset-[#173b38]">
              {ctaLabel}
              <ArrowRight className="h-5 w-5" aria-hidden="true" />
            </Link>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  )
}
