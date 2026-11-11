// Review-only component. Not registered in App or production sample routes.
export interface MixedReviewPaper {
  format: '10-objective-5-subjective-v1'
  status: 'review-only'
  grade: number
  title: string
  objective: {position: number; bank_id: string; content_hash: string; prompt: string; options: {label: string; text: string}[]; answer: string}[]
  subjective: {position: number; prompt: string; marks: number; model_answer: string; marking_criteria: string[]; source_url: string; source_page: number}[]
}
export function MixedSamplePaperReview({paper}: {paper: MixedReviewPaper}) {
  if (paper.status !== 'review-only' || paper.format !== '10-objective-5-subjective-v1' || paper.objective.length !== 10 || paper.subjective.length !== 5) {
    return <p role="alert">The mixed paper is incomplete and cannot be shown.</p>
  }
  return <main className="mixed-review">
    <p>REVIEW ONLY - NOT PUBLISHED OR APPROVED</p>
    <h1>{paper.title}</h1>
    <p>Class {paper.grade}. 15 questions: 10 objective and 5 subjective. Subjective questions are for written practice; no automatic grading is claimed.</p>
    <h2>Section A: 10 objective questions</h2>
    <ol>{paper.objective.map(q=><li key={q.bank_id}><p>{q.prompt}</p><ul>{q.options.map(o=><li key={o.label}>{o.label}. {o.text}</li>)}</ul></li>)}</ol>
    <h2>Section B: 5 subjective questions</h2>
    <ol start={11}>{paper.subjective.map(q=><li key={q.position}><p>{q.prompt} ({q.marks} marks)</p><a href={q.source_url} target="_blank" rel="noreferrer">Source PDF, page {q.source_page}</a></li>)}</ol>
    <details><summary>Objective answer key</summary><ol>{paper.objective.map(q=><li key={q.bank_id}>{q.answer}</li>)}</ol></details>
    <details><summary>Subjective model answers and marking guidance</summary><ol start={11}>{paper.subjective.map(q=><li key={q.position}><p>{q.model_answer}</p><ul>{q.marking_criteria.map((c,i)=><li key={i}>{c}</li>)}</ul></li>)}</ol></details>
  </main>
}
