import React, { useCallback, useEffect, useMemo, useState } from 'react'
import {
  AlertCircle,
  ArrowLeft,
  ArrowRight,
  AlertTriangle,
  BookOpenCheck,
  Calculator,
  CheckCircle2,
  Clock3,
  Flag,
  Globe2,
  Languages,
  Loader2,
  RotateCcw,
  Trophy,
} from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { Header } from '../components/layout/Header'
import {
  CLASS8_SCIENCE_CHAPTER1_ID,
  createQuestionBankTest,
  getQuestionBankChapters,
  QuestionBankBlockSubmissionResult,
  QuestionBankChapter,
  QuestionBankMedia,
  QuestionBankMode,
  QuestionBankOption,
  QuestionBankQuestion,
  QuestionBankTest,
  QuestionBankType,
  flagQuestionBankQuestion,
  submitQuestionBankBlock,
} from '../lib/api'

const QUESTION_COUNT = 10
const TEST_DURATION_SECONDS = 10 * 60

const SUBJECTS = [
  { id: 'science', label: 'Science', meta: 'Class 8', available: true, icon: BookOpenCheck },
  { id: 'mathematics', label: 'Mathematics', meta: 'Coming soon', available: false, icon: Calculator },
  { id: 'english', label: 'English', meta: 'Coming soon', available: false, icon: Languages },
  { id: 'social-science', label: 'Social Science', meta: 'Coming soon', available: false, icon: Globe2 },
] as const

const TEST_MODES: Array<{
  id: QuestionBankMode
  label: string
  description: string
  helper: string
}> = [
  {
    id: 'practice',
    label: 'Practice',
    description: 'A balanced set to build confidence.',
    helper: 'Recommended',
  },
  {
    id: 'remedial',
    label: 'Revise gaps',
    description: 'Spend more time on concepts you missed.',
    helper: 'Focused review',
  },
  {
    id: 'challenge',
    label: 'Challenge',
    description: 'Stretch yourself with tougher questions.',
    helper: 'Go further',
  },
]

const SKILL_LABELS: Record<string, string> = {
  focused_question: 'writing focused investigation questions',
  independent_variable: 'identifying what you change in a test',
  dependent_variable: 'choosing a measurable outcome',
  controlled_variable: 'keeping a fair test consistent',
  fair_test: 'planning fair tests',
  observation: 'separating observations from explanations',
  hypothesis: 'forming testable explanations',
  testable_explanation: 'forming testable explanations',
  causal_reasoning: 'connecting evidence to a result',
  confounded_variables: 'spotting when too many conditions change',
  data_recording: 'recording evidence clearly',
  measurement: 'measuring outcomes precisely',
  evidence: 'using evidence to support a conclusion',
  repeatability: 'checking results with repeated trials',
}

function MediaGallery({ media, compact = false }: { media?: QuestionBankMedia[]; compact?: boolean }) {
  const images = (media || []).filter((asset) => (asset.type || 'image') === 'image' && (asset.url || asset.src))
  if (images.length === 0) return null

  if (compact) {
    return (
      <span className="mt-3 flex flex-wrap gap-2" aria-label="Option images">
        {images.map((asset, index) => (
          <span key={`${asset.url || asset.src}-${index}`} className="inline-flex max-w-44 flex-col overflow-hidden rounded-xl border border-neutral-200 bg-neutral-50 align-top">
            <img
              src={asset.url || asset.src}
              alt={asset.alt || 'Option illustration'}
              className="h-24 w-32 object-contain"
            />
            {asset.caption && <span className="px-2 py-1 text-[11px] font-semibold text-neutral-500">{asset.caption}</span>}
          </span>
        ))}
      </span>
    )
  }

  return (
    <div className="mt-4 grid gap-3 sm:grid-cols-2">
      {images.map((asset, index) => (
        <figure key={`${asset.url || asset.src}-${index}`} className="overflow-hidden rounded-2xl border border-neutral-200 bg-neutral-50">
          <img
            src={asset.url || asset.src}
            alt={asset.alt || 'Question illustration'}
            className="max-h-64 w-full object-contain"
          />
          {asset.caption && <figcaption className="px-3 py-2 text-xs font-semibold text-neutral-500">{asset.caption}</figcaption>}
        </figure>
      ))}
    </div>
  )
}

function formatTime(seconds: number) {
  const minutes = Math.floor(seconds / 60).toString().padStart(2, '0')
  const remainingSeconds = (seconds % 60).toString().padStart(2, '0')
  return `${minutes}:${remainingSeconds}`
}

function getAnswerText(question: QuestionBankQuestion, optionId?: QuestionBankOption['id'] | null) {
  if (!optionId) return 'Not answered'
  const option = question.options.find((candidate) => candidate.id === optionId)
  return option ? `${option.id}. ${option.text}` : String(optionId)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

const QUESTION_TYPE_LABELS: Record<QuestionBankType, string> = {
  mcq_single: 'Choose one',
  mcq_multi: 'Choose all that apply',
  assertion_reason: 'Assertion · reason',
  true_false: 'True / false',
  fill_blank: 'Fill in the blank',
  numerical: 'Numerical answer',
  case_study: 'Case study',
  diagram_based: 'Diagram-based',
  matching: 'Match the following',
}

function DeterministicDiagram({ config }: { config: Record<string, unknown> }) {
  if (config.diagram_type !== 'investigation_flow') return null
  const labels = Array.isArray(config.labels) ? config.labels.map(String) : ['Change', 'Measure', 'Keep same']
  return (
    <div className="mt-7 rounded-2xl border border-primary-100 bg-primary-50/60 p-5">
      <p className="text-xs font-bold uppercase tracking-[0.12em] text-primary-700">Investigation diagram</p>
      <div className="mt-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        {labels.map((label, index) => (
          <React.Fragment key={`${label}-${index}`}>
            <div className="flex min-h-16 flex-1 items-center justify-center rounded-xl border border-primary-200 bg-white px-3 text-center text-sm font-bold text-dark shadow-sm">{label}</div>
            {index < labels.length - 1 && <ArrowRight className="mx-auto h-4 w-4 shrink-0 text-primary-500 sm:mx-1" aria-hidden="true" />}
          </React.Fragment>
        ))}
      </div>
    </div>
  )
}

function getReviewSummary(
  questions: QuestionBankQuestion[],
  result: QuestionBankBlockSubmissionResult,
) {
  const resultByQuestionId = new Map(result.results.map((item) => [item.question_id, item]))
  const missedQuestions = questions.filter((question) => !resultByQuestionId.get(question.id)?.is_correct)
  const focusAreas = [...new Set(
    missedQuestions.flatMap((question) => question.skill_tags.map((tag) => SKILL_LABELS[tag] || tag.replaceAll('_', ' '))),
  )].slice(0, 2)

  if (missedQuestions.length === 0) {
    return {
      title: 'You have a strong handle on this chapter.',
      body: 'Every answer was correct. Keep building speed, then try another session to make the ideas stick.',
      focusAreas: [],
    }
  }

  if (result.percentage >= 70) {
    return {
      title: 'You have a solid foundation to build on.',
      body: 'Review the questions below, then practise the focus areas once more to turn small gaps into confidence.',
      focusAreas,
    }
  }

  return {
    title: 'Start with the core ideas and build from there.',
    body: 'The explanations below show the next concepts to revisit. A short follow-up practice will help you check your understanding again.',
    focusAreas,
  }
}

export function QuestionBankPracticePage() {
  const navigate = useNavigate()
  const [test, setTest] = useState<QuestionBankTest | null>(null)
  const [answers, setAnswers] = useState<Record<string, unknown>>({})
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0)
  const [timeRemaining, setTimeRemaining] = useState(0)
  const [lastResult, setLastResult] = useState<QuestionBankBlockSubmissionResult | null>(null)
  const [reviewQuestions, setReviewQuestions] = useState<QuestionBankQuestion[]>([])
  const [submittedByTimeout, setSubmittedByTimeout] = useState(false)
  const [timedOut, setTimedOut] = useState(false)
  const [loading, setLoading] = useState(false)
  const [chaptersLoading, setChaptersLoading] = useState(true)
  const [chapters, setChapters] = useState<QuestionBankChapter[]>([])
  const [selectedSubjectId, setSelectedSubjectId] = useState('science')
  const [selectedChapterId, setSelectedChapterId] = useState(CLASS8_SCIENCE_CHAPTER1_ID)
  const [selectedMode, setSelectedMode] = useState<QuestionBankMode>('practice')
  const [flaggedQuestionIds, setFlaggedQuestionIds] = useState<Record<string, boolean>>({})
  const [flaggingQuestionId, setFlaggingQuestionId] = useState<string | null>(null)
  const [flagMessage, setFlagMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const selectedSubject = SUBJECTS.find((subject) => subject.id === selectedSubjectId) || SUBJECTS[0]
  const selectedModeOption = TEST_MODES.find((mode) => mode.id === selectedMode) || TEST_MODES[0]
  const selectedChapter = chapters.find((chapter) => chapter.id === selectedChapterId)
  const answeredCount = useMemo(() => Object.keys(answers).length, [answers])
  const currentQuestion = test?.questions[currentQuestionIndex]
  const currentAnswer = currentQuestion ? answers[currentQuestion.id] : undefined
  const currentRenderConfig = currentQuestion?.render_config || {}
  const isLastQuestion = Boolean(test && currentQuestionIndex === test.questions.length - 1)
  const progress = test ? ((currentQuestionIndex + 1) / test.questions.length) * 100 : 0
  const timerIsUrgent = timeRemaining > 0 && timeRemaining <= 60
  const reviewSummary = lastResult ? getReviewSummary(reviewQuestions, lastResult) : null

  const start = async () => {
    if (!selectedChapterId) return
    setLoading(true)
    setError(null)
    setLastResult(null)
    setReviewQuestions([])
    setSubmittedByTimeout(false)
    setTimedOut(false)
    setAnswers({})
    setCurrentQuestionIndex(0)
    setFlagMessage(null)
    try {
      const nextTest = await createQuestionBankTest(QUESTION_COUNT, QUESTION_COUNT, selectedChapterId, selectedMode)
      setTimeRemaining(TEST_DURATION_SECONDS)
      setTest(nextTest)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Could not start practice.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    let active = true
    void getQuestionBankChapters()
      .then((availableChapters) => {
        if (!active) return
        setChapters(availableChapters)
        if (availableChapters.length > 0) {
          setSelectedChapterId((current) => availableChapters.some((chapter) => chapter.id === current) ? current : availableChapters[0].id)
        }
      })
      .catch((caught) => {
        if (active) setError(caught instanceof Error ? caught.message : 'Could not load practice chapters.')
      })
      .finally(() => {
        if (active) setChaptersLoading(false)
      })

    return () => {
      active = false
    }
  }, [])

  const submit = useCallback(async (wasTimedOut = false) => {
    if (!test || loading) return

    setLoading(true)
    setError(null)
    try {
      const result = await submitQuestionBankBlock(
        test.test_id,
        test.block_number,
        test.questions.map((question) => ({
          question_id: question.id,
          answer: answers[question.id] ?? null,
        })),
      )
      setLastResult(result)
      setReviewQuestions(test.questions)
      setSubmittedByTimeout(wasTimedOut)
      setTimedOut(false)
      setAnswers({})
      setCurrentQuestionIndex(0)
      setTimeRemaining(0)
      setTest(result.next_block || null)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Could not submit this test.')
    } finally {
      setLoading(false)
    }
  }, [answers, loading, test])

  useEffect(() => {
    if (!test || loading || timeRemaining <= 0) return

    const timer = window.setTimeout(() => {
      setTimeRemaining((current) => Math.max(current - 1, 0))
    }, 1000)

    return () => window.clearTimeout(timer)
  }, [loading, test, timeRemaining])

  useEffect(() => {
    if (!test || loading || timeRemaining !== 0 || timedOut) return

    setTimedOut(true)
    void submit(true)
  }, [loading, submit, test, timeRemaining, timedOut])

  const goToNextQuestion = () => {
    if (!test) return
    if (isLastQuestion) {
      void submit()
      return
    }
    setFlagMessage(null)
    setCurrentQuestionIndex((current) => current + 1)
  }

  const goToPreviousQuestion = () => {
    setFlagMessage(null)
    setCurrentQuestionIndex((current) => Math.max(current - 1, 0))
  }

  const flagCurrentQuestion = async () => {
    if (!currentQuestion || flaggedQuestionIds[currentQuestion.id] || flaggingQuestionId) return

    const questionId = currentQuestion.id
    setFlaggingQuestionId(questionId)
    setFlagMessage(null)
    try {
      await flagQuestionBankQuestion(questionId)
      setFlaggedQuestionIds((current) => ({ ...current, [questionId]: true }))
      setFlagMessage('Flagged for review.')
    } catch (caught) {
      setFlagMessage(caught instanceof Error ? caught.message : 'Could not flag this question.')
    } finally {
      setFlaggingQuestionId((current) => current === questionId ? null : current)
    }
  }

  return (
    <div className="min-h-screen bg-cream font-sans">
      <Header />
      <main className="mx-auto max-w-4xl px-4 py-8 sm:px-6 sm:py-10 lg:px-8">
        {error && (
          <div className="mb-6 flex items-start gap-3 rounded-2xl border border-red-200 bg-red-50 p-4 text-red-800" role="alert">
            <AlertCircle className="mt-0.5 h-5 w-5 shrink-0" />
            <div>
              <p className="font-semibold">Practice is unavailable</p>
              <p className="mt-1 text-sm">{error}</p>
            </div>
          </div>
        )}

        {test && currentQuestion ? (
          <div className="mx-auto max-w-3xl space-y-5">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-sm font-semibold uppercase tracking-[0.16em] text-primary-600">Practice session</p>
                <h1 className="mt-2 text-2xl font-bold text-dark sm:text-3xl">Class 8 Science</h1>
                <p className="mt-1 text-sm text-neutral-500">{selectedChapter?.title || 'Practice chapter'}</p>
              </div>
              <div
                className={`inline-flex min-h-11 items-center gap-2 rounded-xl border px-3 text-sm font-bold tabular-nums ${timerIsUrgent ? 'border-red-200 bg-red-50 text-red-700' : 'border-neutral-200 bg-white text-dark'}`}
                aria-label={`${formatTime(timeRemaining)} remaining`}
                aria-live="polite"
              >
                <Clock3 className="h-4 w-4" aria-hidden="true" />
                {formatTime(timeRemaining)}
              </div>
            </div>

            <div className="rounded-2xl border border-neutral-100 bg-white p-4 shadow-sm sm:p-5">
              <div className="flex items-center justify-between gap-4 text-sm font-semibold text-neutral-500">
                <span>Question {currentQuestionIndex + 1} of {test.questions.length}</span>
                <span>{answeredCount}/{test.questions.length} answered</span>
              </div>
              <div className="mt-3 h-2 overflow-hidden rounded-full bg-neutral-100" aria-hidden="true">
                <div className="h-full rounded-full bg-primary-600 transition-[width] duration-300" style={{ width: `${progress}%` }} />
              </div>
            </div>

            <section className="rounded-3xl border border-neutral-100 bg-white p-5 shadow-sm sm:p-8">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <p className="text-sm font-semibold text-neutral-500">Question {currentQuestionIndex + 1}</p>
                <span className="rounded-full bg-neutral-100 px-3 py-1.5 text-xs font-bold uppercase tracking-[0.1em] text-neutral-600">{QUESTION_TYPE_LABELS[currentQuestion.question_type]}</span>
              </div>
              <h2 className="mt-4 text-xl font-bold leading-8 text-dark sm:text-2xl">{currentQuestion.question_text}</h2>
              <MediaGallery media={currentQuestion.media} />
              {currentQuestion.question_type === 'diagram_based' && <DeterministicDiagram config={currentRenderConfig} />}

              {currentQuestion.question_type === 'case_study' && typeof currentRenderConfig.passage === 'string' && (
                <div className="mt-7 rounded-2xl border border-primary-100 bg-primary-50/70 p-5">
                  <p className="text-xs font-bold uppercase tracking-[0.12em] text-primary-700">Read the case</p>
                  <p className="mt-2 text-sm leading-7 text-neutral-700">{currentRenderConfig.passage}</p>
                </div>
              )}

              {currentQuestion.question_type === 'assertion_reason' && (typeof currentRenderConfig.assertion === 'string' || typeof currentRenderConfig.reason === 'string') && (
                <div className="mt-7 grid gap-3 sm:grid-cols-2">
                  {typeof currentRenderConfig.assertion === 'string' && <div className="rounded-2xl border border-neutral-200 bg-neutral-50 p-4"><p className="text-xs font-bold uppercase tracking-[0.12em] text-primary-700">Assertion</p><p className="mt-2 text-sm font-semibold leading-6 text-dark">{currentRenderConfig.assertion}</p></div>}
                  {typeof currentRenderConfig.reason === 'string' && <div className="rounded-2xl border border-neutral-200 bg-neutral-50 p-4"><p className="text-xs font-bold uppercase tracking-[0.12em] text-primary-700">Reason</p><p className="mt-2 text-sm font-semibold leading-6 text-dark">{currentRenderConfig.reason}</p></div>}
                </div>
              )}

              {currentQuestion.question_type === 'matching' && (
                <div className="mt-7 space-y-3">
                  {(Array.isArray(currentRenderConfig.prompts) ? currentRenderConfig.prompts : []).filter(isRecord).map((prompt) => {
                    const promptId = String(prompt.id || '')
                    const matchingAnswer = isRecord(currentAnswer) ? currentAnswer : {}
                    return (
                      <div key={promptId} className="flex flex-col gap-2 rounded-2xl border border-neutral-200 bg-neutral-50 p-4 sm:flex-row sm:items-center sm:justify-between">
                        <span className="text-sm font-bold text-dark">{String(prompt.text || promptId)}</span>
                        <select
                          value={typeof matchingAnswer[promptId] === 'string' ? String(matchingAnswer[promptId]) : ''}
                          onChange={(event) => setAnswers((current) => ({ ...current, [currentQuestion.id]: { ...matchingAnswer, [promptId]: event.target.value } }))}
                          className="min-h-11 rounded-xl border border-neutral-200 bg-white px-3 text-sm font-semibold text-dark outline-none focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 sm:max-w-xs"
                        >
                          <option value="">Choose a match</option>
                          {(Array.isArray(currentRenderConfig.choices) ? currentRenderConfig.choices : []).filter(isRecord).map((choice) => <option key={String(choice.id)} value={String(choice.id)}>{String(choice.text || choice.id)}</option>)}
                        </select>
                      </div>
                    )
                  })}
                </div>
              )}

              {currentQuestion.question_type === 'numerical' && (
                <div className="mt-7 max-w-md">
                  <label htmlFor="practice-numerical-answer" className="text-sm font-bold text-neutral-700">Your answer</label>
                  <div className="mt-2 flex items-center rounded-2xl border border-neutral-200 bg-white px-4 focus-within:border-primary-500 focus-within:ring-2 focus-within:ring-primary-500/20">
                    <input id="practice-numerical-answer" type="number" inputMode="decimal" value={typeof currentAnswer === 'number' || typeof currentAnswer === 'string' ? String(currentAnswer) : ''} onChange={(event) => setAnswers((current) => ({ ...current, [currentQuestion.id]: event.target.value }))} placeholder="Enter a number" className="min-h-14 min-w-0 flex-1 bg-transparent text-lg font-semibold text-dark outline-none" />
                    {typeof currentRenderConfig.unit === 'string' && <span className="border-l border-neutral-200 pl-4 text-sm font-bold text-neutral-500">{currentRenderConfig.unit}</span>}
                  </div>
                </div>
              )}

              {currentQuestion.question_type === 'fill_blank' && (
                <div className="mt-7">
                  <label htmlFor="practice-fill-blank-answer" className="text-sm font-bold text-neutral-700">Your answer</label>
                  <input id="practice-fill-blank-answer" type="text" value={typeof currentAnswer === 'string' ? currentAnswer : ''} onChange={(event) => setAnswers((current) => ({ ...current, [currentQuestion.id]: event.target.value }))} placeholder="Type the missing word" className="mt-2 min-h-14 w-full rounded-2xl border border-neutral-200 bg-white px-4 text-base font-semibold text-dark outline-none focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20" />
                </div>
              )}

              {['mcq_single', 'mcq_multi', 'true_false', 'assertion_reason', 'case_study', 'diagram_based'].includes(currentQuestion.question_type) && (
                <div className="mt-7 grid gap-3">
                  {currentQuestion.options.map((option) => {
                    const isMultiple = currentQuestion.question_type === 'mcq_multi'
                    const selectedOptions = Array.isArray(currentAnswer) ? currentAnswer.map(String) : []
                    const isSelected = isMultiple ? selectedOptions.includes(option.id) : currentAnswer === option.id
                    const updateSelection = () => {
                      if (!isMultiple) {
                        setAnswers((current) => ({ ...current, [currentQuestion.id]: option.id }))
                        return
                      }
                      const next = new Set(selectedOptions)
                      if (next.has(option.id)) next.delete(option.id)
                      else next.add(option.id)
                      setAnswers((current) => ({ ...current, [currentQuestion.id]: Array.from(next) }))
                    }
                    return (
                      <button
                        key={option.id}
                        type="button"
                        onClick={updateSelection}
                        aria-pressed={isSelected}
                        className={`flex min-h-14 items-start gap-3 rounded-2xl border px-4 py-4 text-left text-sm transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 sm:text-base ${isSelected ? 'border-primary-500 bg-primary-50 text-primary-900 shadow-sm' : 'border-neutral-200 bg-white text-dark hover:border-primary-300 hover:bg-neutral-50'}`}
                      >
                        <span className={`flex h-7 w-7 shrink-0 items-center justify-center text-sm font-bold ${isMultiple ? 'rounded-lg' : 'rounded-full'} ${isSelected ? 'bg-primary-600 text-white' : 'bg-neutral-100 text-neutral-600'}`}>{isMultiple && isSelected ? '✓' : option.id}</span>
                        <span className="min-w-0 flex-1 pt-0.5">{option.text}</span>
                        <MediaGallery media={option.media} compact />
                      </button>
                    )
                  })}
                </div>
              )}
              <div className="mt-6 flex flex-col gap-2 border-t border-neutral-100 pt-4 sm:flex-row sm:items-center sm:justify-between">
                <button
                  type="button"
                  onClick={() => void flagCurrentQuestion()}
                  disabled={Boolean(flaggedQuestionIds[currentQuestion.id]) || flaggingQuestionId === currentQuestion.id}
                  aria-label="Flag this question for review"
                  className="inline-flex min-h-9 w-fit items-center gap-2 rounded-lg px-2 text-xs font-semibold text-neutral-500 transition-colors hover:bg-neutral-50 hover:text-dark focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 disabled:cursor-default disabled:text-primary-700"
                >
                  {flaggingQuestionId === currentQuestion.id ? <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" /> : <Flag className="h-3.5 w-3.5" aria-hidden="true" />}
                  {flaggedQuestionIds[currentQuestion.id] ? 'Flagged for review' : 'Flag question'}
                </button>
                {flagMessage && <p className="text-xs font-semibold text-neutral-500" role="status" aria-live="polite">{flagMessage}</p>}
              </div>
            </section>

            <div className="flex items-center justify-between gap-3">
              <button
                type="button"
                onClick={goToPreviousQuestion}
                disabled={currentQuestionIndex === 0 || loading}
                className="inline-flex min-h-11 items-center gap-2 rounded-xl px-3 text-sm font-semibold text-neutral-600 transition-colors hover:bg-white hover:text-dark disabled:cursor-not-allowed disabled:opacity-40"
              >
                <ArrowLeft className="h-4 w-4" aria-hidden="true" />
                Back
              </button>
              <button
                type="button"
                onClick={goToNextQuestion}
                disabled={loading}
                className="inline-flex min-h-11 items-center gap-2 rounded-xl bg-dark px-5 py-3 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-neutral-800 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading && <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />}
                {isLastQuestion ? 'Finish test' : 'Next question'}
                {!loading && <ArrowRight className="h-4 w-4" aria-hidden="true" />}
              </button>
            </div>
          </div>
        ) : lastResult && reviewSummary ? (
          <div className="space-y-6">
            <section className="rounded-3xl border border-neutral-100 bg-white p-6 shadow-sm sm:p-8">
              <div className="flex flex-col gap-6 sm:flex-row sm:items-start sm:justify-between">
                <div>
                  <p className="text-sm font-semibold uppercase tracking-[0.16em] text-primary-600">{submittedByTimeout ? "Time's up" : 'Test complete'}</p>
                  <h1 className="mt-3 text-3xl font-bold text-dark sm:text-4xl">Your practice review</h1>
                  <p className="mt-3 max-w-xl text-neutral-600">See what you got right, understand what to revisit, and keep your momentum going.</p>
                </div>
                <div className="rounded-2xl bg-primary-50 px-5 py-4 text-left sm:min-w-36 sm:text-center">
                  <p className="text-3xl font-bold text-primary-700">{lastResult.percentage}%</p>
                  <p className="mt-1 text-sm font-semibold text-primary-700">{lastResult.block_score}/{lastResult.block_total} correct</p>
                </div>
              </div>

              <div className="mt-7 flex flex-col gap-3 sm:flex-row">
                <button
                  type="button"
                  onClick={start}
                  className="inline-flex min-h-12 items-center justify-center gap-2 rounded-xl bg-primary-600 px-5 py-3 font-semibold text-white shadow-sm transition-colors hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
                >
                  Practice again
                  <ArrowRight className="h-4 w-4" aria-hidden="true" />
                </button>
                <button
                  type="button"
                  onClick={() => navigate('/leaderboard')}
                  className="inline-flex min-h-12 items-center justify-center gap-2 rounded-xl border border-neutral-200 bg-white px-5 py-3 font-semibold text-dark transition-colors hover:bg-neutral-50 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
                >
                  <Trophy className="h-4 w-4" aria-hidden="true" />
                  Compare with peers
                </button>
              </div>
            </section>

            <section className="rounded-3xl border border-primary-100 bg-primary-50/70 p-6 sm:p-8">
              <div className="flex items-start gap-4">
                <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-white text-primary-700 shadow-sm">
                  <BookOpenCheck className="h-5 w-5" aria-hidden="true" />
                </div>
                <div>
                  <p className="text-sm font-semibold uppercase tracking-[0.14em] text-primary-700">What to focus on next</p>
                  <h2 className="mt-2 text-xl font-bold text-dark">{reviewSummary.title}</h2>
                  <p className="mt-2 leading-7 text-neutral-700">{reviewSummary.body}</p>
                  {reviewSummary.focusAreas.length > 0 && (
                    <div className="mt-4 flex flex-wrap gap-2">
                      {reviewSummary.focusAreas.map((focusArea) => (
                        <span key={focusArea} className="rounded-full bg-white px-3 py-1.5 text-sm font-semibold text-primary-800 shadow-sm">
                          {focusArea}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </section>

            <section className="space-y-4" aria-labelledby="question-review-heading">
              <div className="flex items-end justify-between gap-4">
                <div>
                  <p className="text-sm font-semibold uppercase tracking-[0.14em] text-neutral-500">Question-by-question</p>
                  <h2 id="question-review-heading" className="mt-2 text-2xl font-bold text-dark">Review your answers</h2>
                </div>
                <p className="text-sm font-semibold text-neutral-500">{reviewQuestions.length} questions</p>
              </div>

              <div className="space-y-3">
                {reviewQuestions.map((question, index) => {
                  const result = lastResult.results.find((item) => item.question_id === question.id)
                  const isCorrect = Boolean(result?.is_correct)
                  return (
                    <article key={question.id} className="rounded-2xl border border-neutral-100 bg-white p-5 shadow-sm sm:p-6">
                      <div className="flex items-start gap-3">
                        <div className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full ${isCorrect ? 'bg-primary-50 text-primary-700' : 'bg-amber-50 text-amber-700'}`}>
                          {isCorrect ? <CheckCircle2 className="h-5 w-5" aria-hidden="true" /> : <AlertTriangle className="h-5 w-5" aria-hidden="true" />}
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <p className="text-sm font-semibold text-neutral-500">Question {index + 1}</p>
                            <span className={`text-sm font-bold ${isCorrect ? 'text-primary-700' : 'text-amber-700'}`}>
                              {isCorrect ? 'Correct' : 'Review this one'}
                            </span>
                          </div>
                          <h3 className="mt-3 text-base font-bold leading-7 text-dark sm:text-lg">{question.question_text}</h3>

                          <div className="mt-4 grid gap-3 sm:grid-cols-2">
                            <div className="rounded-xl bg-neutral-50 p-3">
                              <p className="text-xs font-bold uppercase tracking-[0.12em] text-neutral-500">Your answer</p>
                              <p className="mt-1 text-sm leading-6 text-dark">{result?.answer_summary || getAnswerText(question, result?.selected_option_id)}</p>
                            </div>
                            {!isCorrect && (
                              <div className="rounded-xl bg-primary-50 p-3">
                                <p className="text-xs font-bold uppercase tracking-[0.12em] text-primary-700">Correct answer</p>
                                <p className="mt-1 text-sm leading-6 text-primary-950">{result?.correct_answer_summary || getAnswerText(question, result?.correct_option_id)}</p>
                              </div>
                            )}
                          </div>

                          {result?.explanation && (
                            <div className="mt-4 border-l-2 border-primary-300 pl-3">
                              <p className="text-xs font-bold uppercase tracking-[0.12em] text-neutral-500">Why</p>
                              <p className="mt-1 text-sm leading-6 text-neutral-700">{result.explanation}</p>
                            </div>
                          )}
                        </div>
                      </div>
                    </article>
                  )
                })}
              </div>
            </section>

            <div className="flex justify-center pb-4">
              <button
                type="button"
                onClick={start}
                className="inline-flex min-h-12 items-center justify-center gap-2 rounded-xl bg-primary-600 px-6 py-3 font-semibold text-white shadow-sm transition-colors hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
              >
                <RotateCcw className="h-4 w-4" aria-hidden="true" />
                Start another practice test
              </button>
            </div>
          </div>
        ) : (
          <div className="mx-auto max-w-3xl space-y-6">
            <section className="overflow-hidden rounded-3xl border border-neutral-100 bg-white shadow-sm">
              <div className="border-b border-neutral-100 bg-cream px-6 py-7 sm:px-10 sm:py-9">
                <p className="text-sm font-semibold uppercase tracking-[0.16em] text-primary-600">Practice setup</p>
                <h1 className="mt-3 text-3xl font-bold tracking-tight text-dark sm:text-4xl">Choose what to practise</h1>
                <p className="mt-3 max-w-xl text-base leading-7 text-neutral-600">Set up a short session for Class 8. You can change the focus without changing your subject or chapter.</p>
              </div>

              <div className="space-y-8 p-6 sm:p-10">
                <fieldset>
                  <legend className="text-sm font-bold text-dark">Subject</legend>
                  <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
                    {SUBJECTS.map((subject) => {
                      const SubjectIcon = subject.icon
                      const isSelected = subject.id === selectedSubjectId
                      return (
                        <button
                          key={subject.id}
                          type="button"
                          onClick={() => subject.available && setSelectedSubjectId(subject.id)}
                          disabled={!subject.available}
                          aria-pressed={isSelected}
                          className={`relative min-h-24 rounded-2xl border px-3 py-4 text-left transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 ${isSelected ? 'border-primary-500 bg-primary-50 text-primary-950 shadow-sm' : subject.available ? 'border-neutral-200 bg-white text-dark hover:border-primary-300 hover:bg-primary-50/50' : 'cursor-not-allowed border-neutral-100 bg-neutral-50 text-neutral-400'}`}
                        >
                          <SubjectIcon className={`h-5 w-5 ${isSelected ? 'text-primary-700' : 'text-neutral-400'}`} aria-hidden="true" />
                          <span className="mt-3 block text-sm font-bold">{subject.label}</span>
                          <span className={`mt-1 block text-xs font-semibold ${isSelected ? 'text-primary-700' : 'text-neutral-400'}`}>{subject.meta}</span>
                          {!subject.available && <span className="absolute right-2 top-2 rounded-full bg-white px-2 py-1 text-[10px] font-bold uppercase tracking-[0.08em] text-neutral-400 shadow-sm">Soon</span>}
                        </button>
                      )
                    })}
                  </div>
                </fieldset>

                <div className="border-t border-neutral-100 pt-7">
                  <label htmlFor="practice-chapter" className="text-sm font-bold text-dark">Chapter</label>
                  <select
                    id="practice-chapter"
                    value={selectedChapterId}
                    onChange={(event) => setSelectedChapterId(event.target.value)}
                    disabled={chaptersLoading || loading || chapters.length === 0}
                    className="mt-3 min-h-12 w-full rounded-xl border border-neutral-200 bg-white px-3 text-sm font-semibold text-dark outline-none transition-colors focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20"
                  >
                    {chapters.map((chapter) => (
                      <option key={chapter.id} value={chapter.id}>
                        Chapter {chapter.sequence_number} · {chapter.title}
                      </option>
                    ))}
                  </select>
                  <p className="mt-2 max-w-2xl text-sm leading-6 text-neutral-500">{selectedChapter?.description || 'Select a chapter to begin.'}</p>
                </div>

                <fieldset className="border-t border-neutral-100 pt-7">
                  <legend className="text-sm font-bold text-dark">How do you want to practise?</legend>
                  <div className="mt-3 grid gap-3 sm:grid-cols-3">
                    {TEST_MODES.map((mode) => {
                      const isSelected = mode.id === selectedMode
                      return (
                        <button
                          key={mode.id}
                          type="button"
                          onClick={() => setSelectedMode(mode.id)}
                          aria-pressed={isSelected}
                          className={`min-h-32 rounded-2xl border p-4 text-left transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 ${isSelected ? 'border-primary-500 bg-primary-50 text-primary-950 shadow-sm' : 'border-neutral-200 bg-white text-dark hover:border-primary-300 hover:bg-neutral-50'}`}
                        >
                          <div className="flex items-start justify-between gap-2">
                            <span className="text-sm font-bold">{mode.label}</span>
                            {isSelected && <span className="rounded-full bg-primary-600 px-2 py-1 text-[10px] font-bold uppercase tracking-[0.08em] text-white">Selected</span>}
                          </div>
                          <p className="mt-3 text-sm leading-6 text-neutral-600">{mode.description}</p>
                          <p className={`mt-3 text-xs font-bold uppercase tracking-[0.1em] ${isSelected ? 'text-primary-700' : 'text-neutral-400'}`}>{mode.helper}</p>
                        </button>
                      )
                    })}
                  </div>
                </fieldset>

                <div className="flex flex-col gap-4 border-t border-neutral-100 pt-7 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <p className="text-sm font-bold text-dark">10 questions · 10 minutes</p>
                    <p className="mt-1 text-sm text-neutral-500">{selectedSubject.label} · {selectedModeOption.label}</p>
                  </div>
                  <button
                    type="button"
                    onClick={start}
                    disabled={loading || chaptersLoading || chapters.length === 0 || !selectedSubject.available}
                    className="inline-flex min-h-12 items-center justify-center gap-2 rounded-xl bg-primary-600 px-5 py-3 font-semibold text-white shadow-sm transition-colors hover:bg-primary-700 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {loading ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> : <ArrowRight className="h-4 w-4" aria-hidden="true" />}
                    Start {selectedModeOption.label}
                  </button>
                </div>
              </div>
            </section>

          </div>
        )}
      </main>
    </div>
  )
}
