import React, { useMemo, useState } from 'react'
import {
  ArrowLeft,
  ArrowRight,
  Check,
  CheckCircle2,
  Circle,
  Eye,
  FileText,
  RotateCcw,
  Sparkles,
} from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { Header } from '../components/layout/Header'

type DemoKind =
  | 'mcq_single'
  | 'mcq_multi'
  | 'true_false'
  | 'numerical'
  | 'fill_blank'
  | 'assertion_reason'
  | 'matching'
  | 'case_study'
  | 'diagram_based'

type DemoQuestion = {
  id: string
  kind: DemoKind
  label: string
  prompt: string
  options?: string[]
  assertion?: string
  reason?: string
  passage?: string
  statements?: string[]
  matches?: string[]
}

type DemoAnswer = {
  selected?: string
  selectedMany?: string[]
  text?: string
  number?: string
  matches?: Record<string, string>
}

const DEMO_QUESTIONS: DemoQuestion[] = [
  {
    id: 'mcq-single',
    kind: 'mcq_single',
    label: 'MCQ · single answer',
    prompt: 'Which part of a plant mainly absorbs water and minerals from the soil?',
    options: ['Leaf', 'Root', 'Flower', 'Fruit'],
  },
  {
    id: 'mcq-multi',
    kind: 'mcq_multi',
    label: 'MCQ · multiple answers',
    prompt: 'Which two actions help make an experiment a fair test?',
    options: ['Change one variable at a time', 'Record observations', 'Change every condition together', 'Ignore unexpected results'],
  },
  {
    id: 'true-false',
    kind: 'true_false',
    label: 'True / false',
    prompt: 'A closed electric circuit is needed for current to flow continuously.',
    options: ['True', 'False'],
  },
  {
    id: 'numerical',
    kind: 'numerical',
    label: 'Numerical answer',
    prompt: 'A runner covers 240 metres in 30 seconds. What is the runner’s average speed?',
  },
  {
    id: 'fill-blank',
    kind: 'fill_blank',
    label: 'Fill in the blank',
    prompt: 'The change of water vapour into liquid water is called ________.',
  },
  {
    id: 'assertion-reason',
    kind: 'assertion_reason',
    label: 'Assertion · reason',
    prompt: 'Choose the relationship between the assertion and the reason.',
    assertion: 'Assertion: A metal spoon feels colder than a wooden spoon in the same room.',
    reason: 'Reason: Metal conducts heat away from your hand faster than wood.',
    options: [
      'Both are true, and the reason explains the assertion.',
      'Both are true, but the reason does not explain the assertion.',
      'The assertion is true, but the reason is false.',
      'The assertion is false, but the reason is true.',
    ],
  },
  {
    id: 'matching',
    kind: 'matching',
    label: 'Match the following',
    prompt: 'Match each component with the job it performs in a simple circuit.',
    statements: ['Cell', 'Switch', 'Bulb'],
    matches: ['Supplies electrical energy', 'Opens or closes the path', 'Converts energy into light'],
  },
  {
    id: 'case-study',
    kind: 'case_study',
    label: 'Case study',
    prompt: 'Which conclusion is best supported by the investigation?',
    passage: 'A group placed identical ice cubes in metal, plastic, and wooden cups. They kept all cups in the same room and recorded how long each cube took to melt. The ice in the metal cup melted first.',
    options: ['Metal transferred heat to the ice fastest.', 'The metal cup contained more ice.', 'Wood always makes ice colder.', 'The room temperature changed for only one cup.'],
  },
  {
    id: 'diagram-based',
    kind: 'diagram_based',
    label: 'Diagram-based',
    prompt: 'Which switch position will complete the circuit shown below?',
    options: ['Switch A', 'Switch B', 'Both switches', 'Neither switch'],
  },
]

const KIND_LABELS: Record<DemoKind, string> = {
  mcq_single: 'Single choice',
  mcq_multi: 'Multiple choice',
  true_false: 'True / false',
  numerical: 'Numerical',
  fill_blank: 'Fill in the blank',
  assertion_reason: 'Assertion / reason',
  matching: 'Matching',
  case_study: 'Case study',
  diagram_based: 'Diagram-based',
}

function isAnswered(answer?: DemoAnswer) {
  return Boolean(
    answer?.selected ||
      answer?.selectedMany?.length ||
      answer?.text?.trim() ||
      answer?.number?.trim() ||
      Object.keys(answer?.matches || {}).length,
  )
}

function ChoiceButton({
  label,
  marker,
  selected,
  multiple = false,
  onClick,
}: {
  label: string
  marker: string
  selected: boolean
  multiple?: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={selected}
      className={`flex min-h-14 w-full items-center gap-3 rounded-2xl border px-4 py-3 text-left transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 ${selected ? 'border-primary-500 bg-primary-50 text-primary-950' : 'border-neutral-200 bg-white text-dark hover:border-primary-300 hover:bg-neutral-50'}`}
    >
      <span className={`flex h-8 w-8 shrink-0 items-center justify-center border text-xs font-bold ${multiple ? 'rounded-lg' : 'rounded-full'} ${selected ? 'border-primary-600 bg-primary-600 text-white' : 'border-neutral-300 bg-neutral-50 text-neutral-600'}`}>
        {selected ? <Check className="h-4 w-4" aria-hidden="true" /> : marker}
      </span>
      <span className="text-sm font-semibold leading-6 sm:text-base">{label}</span>
    </button>
  )
}

function CircuitDiagram() {
  return (
    <div className="overflow-hidden rounded-2xl border border-neutral-200 bg-[#fbfcfa] p-4 sm:p-6">
      <svg viewBox="0 0 620 210" role="img" aria-label="A battery, two switches, and a bulb connected by wires" className="h-auto w-full">
        <path d="M74 60H160M74 150H160M160 60V83M160 127V150M460 60H546V150H460M160 60H230M292 60H360M422 60H460M160 150H230M292 150H360M422 150H460" fill="none" stroke="#374151" strokeWidth="5" strokeLinecap="round" />
        <line x1="230" y1="42" x2="230" y2="78" stroke="#2563eb" strokeWidth="7" strokeLinecap="round" />
        <line x1="252" y1="34" x2="252" y2="86" stroke="#2563eb" strokeWidth="7" strokeLinecap="round" />
        <line x1="360" y1="42" x2="360" y2="78" stroke="#2563eb" strokeWidth="7" strokeLinecap="round" />
        <line x1="382" y1="34" x2="382" y2="86" stroke="#2563eb" strokeWidth="7" strokeLinecap="round" />
        <path d="M230 150L274 124" fill="none" stroke="#374151" strokeWidth="5" strokeLinecap="round" />
        <circle cx="230" cy="150" r="7" fill="#374151" />
        <circle cx="292" cy="150" r="7" fill="#374151" />
        <circle cx="360" cy="150" r="7" fill="#374151" />
        <circle cx="422" cy="150" r="7" fill="#374151" />
        <circle cx="390" cy="60" r="30" fill="#fff7d6" stroke="#d97706" strokeWidth="5" />
        <path d="M378 50L402 70M402 50L378 70" stroke="#d97706" strokeWidth="4" strokeLinecap="round" />
        <text x="228" y="112" fill="#4b5563" fontSize="16" fontWeight="700">A</text>
        <text x="354" y="112" fill="#4b5563" fontSize="16" fontWeight="700">B</text>
        <text x="198" y="190" fill="#4b5563" fontSize="15" fontWeight="700">Cell</text>
        <text x="374" y="108" fill="#92400e" fontSize="14" fontWeight="700">Bulb</text>
      </svg>
      <p className="mt-3 text-center text-xs font-semibold text-neutral-500">Illustration preview · labels and media can be stored with the question</p>
    </div>
  )
}

export function QuestionTypePreviewPage() {
  const navigate = useNavigate()
  const [currentIndex, setCurrentIndex] = useState(0)
  const [answers, setAnswers] = useState<Record<string, DemoAnswer>>({})
  const current = DEMO_QUESTIONS[currentIndex]
  const currentAnswer = answers[current.id] || {}
  const answeredCount = useMemo(() => Object.values(answers).filter(isAnswered).length, [answers])
  const progress = ((currentIndex + 1) / DEMO_QUESTIONS.length) * 100

  const updateAnswer = (patch: DemoAnswer) => {
    setAnswers((previous) => ({
      ...previous,
      [current.id]: { ...previous[current.id], ...patch },
    }))
  }

  const selectSingle = (option: string) => updateAnswer({ selected: option })

  const toggleMultiple = (option: string) => {
    const selected = new Set(currentAnswer.selectedMany || [])
    if (selected.has(option)) selected.delete(option)
    else selected.add(option)
    updateAnswer({ selectedMany: Array.from(selected) })
  }

  const renderQuestion = () => {
    if (current.kind === 'mcq_single' || current.kind === 'true_false') {
      return (
        <div className="grid gap-3">
          {current.options?.map((option, index) => (
            <ChoiceButton key={option} label={option} marker={String.fromCharCode(65 + index)} selected={currentAnswer.selected === option} onClick={() => selectSingle(option)} />
          ))}
        </div>
      )
    }

    if (current.kind === 'mcq_multi') {
      return (
        <div className="grid gap-3">
          {current.options?.map((option, index) => (
            <ChoiceButton key={option} label={option} marker={String.fromCharCode(65 + index)} multiple selected={currentAnswer.selectedMany?.includes(option) || false} onClick={() => toggleMultiple(option)} />
          ))}
        </div>
      )
    }

    if (current.kind === 'numerical') {
      return (
        <div className="max-w-md">
          <label htmlFor="numerical-answer" className="text-sm font-bold text-neutral-700">Your answer</label>
          <div className="mt-2 flex items-center rounded-2xl border border-neutral-200 bg-white px-4 focus-within:border-primary-500 focus-within:ring-2 focus-within:ring-primary-500/20">
            <input id="numerical-answer" type="number" inputMode="decimal" value={currentAnswer.number || ''} onChange={(event) => updateAnswer({ number: event.target.value })} placeholder="Enter a number" className="min-h-14 min-w-0 flex-1 bg-transparent text-lg font-semibold text-dark outline-none" />
            <span className="border-l border-neutral-200 pl-4 text-sm font-bold text-neutral-500">m/s</span>
          </div>
          <p className="mt-2 text-xs font-semibold text-neutral-500">The evaluator can allow a numeric tolerance.</p>
        </div>
      )
    }

    if (current.kind === 'fill_blank') {
      return (
        <div>
          <label htmlFor="fill-blank-answer" className="text-sm font-bold text-neutral-700">Your answer</label>
          <input id="fill-blank-answer" type="text" value={currentAnswer.text || ''} onChange={(event) => updateAnswer({ text: event.target.value })} placeholder="Type the missing word" className="mt-2 min-h-14 w-full rounded-2xl border border-neutral-200 bg-white px-4 text-base font-semibold text-dark outline-none transition focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20" />
          <p className="mt-2 text-xs font-semibold text-neutral-500">Exact accepted-answer matching with normalized spacing</p>
        </div>
      )
    }

    if (current.kind === 'assertion_reason') {
      return (
        <div className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-2xl border border-neutral-200 bg-neutral-50 p-4"><p className="text-xs font-bold uppercase tracking-[0.12em] text-primary-700">Assertion</p><p className="mt-2 text-sm font-semibold leading-6 text-dark">{current.assertion}</p></div>
            <div className="rounded-2xl border border-neutral-200 bg-neutral-50 p-4"><p className="text-xs font-bold uppercase tracking-[0.12em] text-primary-700">Reason</p><p className="mt-2 text-sm font-semibold leading-6 text-dark">{current.reason}</p></div>
          </div>
          <div className="grid gap-3">{current.options?.map((option, index) => <ChoiceButton key={option} label={option} marker={String.fromCharCode(65 + index)} selected={currentAnswer.selected === option} onClick={() => selectSingle(option)} />)}</div>
        </div>
      )
    }

    if (current.kind === 'matching') {
      return (
        <div className="space-y-3">
          {current.statements?.map((statement, index) => (
            <div key={statement} className="flex flex-col gap-2 rounded-2xl border border-neutral-200 bg-neutral-50 p-4 sm:flex-row sm:items-center sm:justify-between">
              <span className="text-sm font-bold text-dark">{index + 1}. {statement}</span>
              <select value={currentAnswer.matches?.[statement] || ''} onChange={(event) => updateAnswer({ matches: { ...currentAnswer.matches, [statement]: event.target.value } })} className="min-h-11 rounded-xl border border-neutral-200 bg-white px-3 text-sm font-semibold text-dark outline-none focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 sm:max-w-xs">
                <option value="">Choose a match</option>
                {current.matches?.map((match) => <option key={match} value={match}>{match}</option>)}
              </select>
            </div>
          ))}
        </div>
      )
    }

    if (current.kind === 'case_study') {
      return (
        <div className="space-y-5">
          <div className="rounded-2xl border border-primary-100 bg-primary-50/70 p-5"><p className="text-xs font-bold uppercase tracking-[0.12em] text-primary-700">Read the case</p><p className="mt-2 text-sm leading-7 text-neutral-700">{current.passage}</p></div>
          <div className="grid gap-3">{current.options?.map((option, index) => <ChoiceButton key={option} label={option} marker={String.fromCharCode(65 + index)} selected={currentAnswer.selected === option} onClick={() => selectSingle(option)} />)}</div>
        </div>
      )
    }

    return (
      <div className="space-y-5">
        <CircuitDiagram />
        <div className="grid gap-3">{current.options?.map((option, index) => <ChoiceButton key={option} label={option} marker={String.fromCharCode(65 + index)} selected={currentAnswer.selected === option} onClick={() => selectSingle(option)} />)}</div>
      </div>
    )
  }

  const goTo = (index: number) => setCurrentIndex(Math.min(Math.max(index, 0), DEMO_QUESTIONS.length - 1))
  const restart = () => {
    setAnswers({})
    setCurrentIndex(0)
  }

  return (
    <div className="min-h-screen bg-cream font-sans">
      <Header />
      <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6 sm:py-10 lg:px-8">
        <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <button type="button" onClick={() => navigate('/practice')} className="mb-4 inline-flex items-center gap-2 text-sm font-semibold text-neutral-500 transition-colors hover:text-dark"><ArrowLeft className="h-4 w-4" aria-hidden="true" />Back to practice</button>
            <div className="flex flex-wrap items-center gap-2"><span className="inline-flex items-center gap-2 rounded-full bg-primary-50 px-3 py-1.5 text-xs font-bold uppercase tracking-[0.12em] text-primary-700"><Eye className="h-3.5 w-3.5" aria-hidden="true" />Local preview</span><span className="text-sm font-semibold text-neutral-500">No answers are saved</span></div>
            <h1 className="mt-3 text-3xl font-bold tracking-tight text-dark sm:text-4xl">Question type playground</h1>
            <p className="mt-3 max-w-2xl text-base leading-7 text-neutral-600">Walk through the different question shapes before we commit them to the learner test experience. Try selecting, typing, matching, and reading the diagram.</p>
          </div>
          <div className="flex items-center gap-3 rounded-2xl border border-neutral-200 bg-white px-4 py-3 shadow-sm"><Sparkles className="h-5 w-5 text-primary-600" aria-hidden="true" /><div><p className="text-xs font-bold uppercase tracking-[0.12em] text-neutral-500">Preview test</p><p className="mt-1 text-sm font-bold text-dark">{DEMO_QUESTIONS.length} renderers · {answeredCount} tried</p></div></div>
        </div>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_280px]">
          <section>
            <div className="mb-4 rounded-2xl border border-neutral-100 bg-white p-4 shadow-sm sm:p-5"><div className="flex items-center justify-between gap-4 text-sm font-semibold text-neutral-500"><span>Question {currentIndex + 1} of {DEMO_QUESTIONS.length}</span><span>{Math.round(progress)}% complete</span></div><div className="mt-3 h-2 overflow-hidden rounded-full bg-neutral-100"><div className="h-full rounded-full bg-primary-600 transition-[width] duration-300" style={{ width: `${progress}%` }} /></div></div>

            <article className="rounded-3xl border border-neutral-100 bg-white p-5 shadow-sm sm:p-8">
              <div className="flex flex-wrap items-start justify-between gap-4"><div><p className="text-sm font-semibold text-neutral-500">Question {currentIndex + 1}</p><h2 className="mt-3 max-w-3xl text-xl font-bold leading-8 text-dark sm:text-2xl">{current.prompt}</h2></div><span className="rounded-full bg-neutral-100 px-3 py-1.5 text-xs font-bold uppercase tracking-[0.1em] text-neutral-600">{KIND_LABELS[current.kind]}</span></div>
              <div className="mt-7">{renderQuestion()}</div>
              <div className="mt-7 flex items-center justify-between gap-3 border-t border-neutral-100 pt-5"><button type="button" onClick={() => goTo(currentIndex - 1)} disabled={currentIndex === 0} className="inline-flex min-h-11 items-center gap-2 rounded-xl px-3 text-sm font-semibold text-neutral-600 transition hover:bg-neutral-50 disabled:cursor-not-allowed disabled:opacity-40"><ArrowLeft className="h-4 w-4" aria-hidden="true" />Previous</button>{currentIndex === DEMO_QUESTIONS.length - 1 ? <button type="button" onClick={restart} className="inline-flex min-h-11 items-center gap-2 rounded-xl bg-primary-600 px-5 py-3 text-sm font-semibold text-white transition hover:bg-primary-700"><RotateCcw className="h-4 w-4" aria-hidden="true" />Restart preview</button> : <button type="button" onClick={() => goTo(currentIndex + 1)} className="inline-flex min-h-11 items-center gap-2 rounded-xl bg-dark px-5 py-3 text-sm font-semibold text-white transition hover:bg-neutral-800">Next type<ArrowRight className="h-4 w-4" aria-hidden="true" /></button>}</div>
            </article>
          </section>

          <aside className="h-fit rounded-3xl border border-neutral-100 bg-white p-4 shadow-sm lg:sticky lg:top-6"><div className="flex items-center justify-between gap-3 px-2 pb-3"><div><p className="text-xs font-bold uppercase tracking-[0.12em] text-neutral-500">Renderers</p><p className="mt-1 text-sm font-bold text-dark">Jump to a type</p></div><FileText className="h-5 w-5 text-neutral-400" aria-hidden="true" /></div><nav aria-label="Question type preview" className="space-y-1">{DEMO_QUESTIONS.map((question, index) => { const answered = isAnswered(answers[question.id]); const active = index === currentIndex; return <button key={question.id} type="button" onClick={() => goTo(index)} className={`flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left transition-colors ${active ? 'bg-primary-50 text-primary-900' : 'text-neutral-600 hover:bg-neutral-50'}`}><span className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-bold ${active ? 'bg-primary-600 text-white' : answered ? 'bg-primary-100 text-primary-700' : 'bg-neutral-100 text-neutral-500'}`}>{answered && !active ? <CheckCircle2 className="h-4 w-4" aria-hidden="true" /> : index + 1}</span><span className="min-w-0 flex-1"><span className="block truncate text-sm font-semibold">{question.label}</span><span className="mt-0.5 block text-xs text-neutral-400">{KIND_LABELS[question.kind]}</span></span>{active ? <Circle className="h-3.5 w-3.5 fill-primary-600 text-primary-600" aria-hidden="true" /> : null}</button> })}</nav><div className="mt-4 rounded-2xl bg-neutral-50 p-3 text-xs leading-5 text-neutral-500"><span className="font-bold text-neutral-700">Prototype note:</span> These are representative layouts for the deterministic learner bank. This page does not write test attempts.</div></aside>
        </div>
      </main>
    </div>
  )
}
