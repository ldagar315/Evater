import { Download, FileText } from 'lucide-react'

const papers = [
  { title: 'Light', name: 'Mirrors and lenses', href: '/1-class-8-light-sample-paper.pdf' },
  { title: 'Electricity', name: 'Magnetic and heating effects', href: '/2-class-8-electricity-sample-paper.pdf' },
  { title: 'Pythagoras', name: 'The Baudhayana-Pythagoras theorem', href: '/3-class-8-pythagoras-sample-paper.pdf' },
]

/** PDFs ship in the same immutable frontend deploy as this announcement. */
export function SamplePaperDownloads() {
  return <section aria-labelledby="sample-paper-heading" className="mx-auto my-5 w-full max-w-6xl px-4 sm:px-6">
    <div className="rounded-2xl border border-primary-200 bg-primary-50 p-5 sm:p-7">
      <div className="flex items-start gap-3">
        <FileText aria-hidden="true" className="mt-1 h-6 w-6 shrink-0 text-primary-700" />
        <div className="min-w-0">
          <p className="text-xs font-bold uppercase tracking-wide text-primary-700">New · Class 8</p>
          <h2 id="sample-paper-heading" className="mt-1 text-xl font-bold leading-snug text-dark sm:text-2xl">Sample papers are live</h2>
          <p className="mt-2 text-sm leading-6 text-neutral-700">Download watermarked PDFs with 10 objective and 5 written-response questions, plus answers and marking guidance. No sign-in needed.</p>
        </div>
      </div>
      <div className="mt-5 grid gap-3 sm:grid-cols-3">
        {papers.map(paper => <a key={paper.href} href={paper.href} download className="flex min-h-14 items-center justify-between gap-3 rounded-xl border border-primary-200 bg-white px-4 py-3 text-primary-800 transition-colors hover:bg-primary-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary-700">
          <span className="min-w-0"><span className="block text-sm font-bold">Download {paper.title}</span><span className="mt-1 block text-xs leading-5 text-neutral-600">{paper.name} · PDF</span></span>
          <Download aria-hidden="true" className="h-5 w-5 shrink-0" />
        </a>)}
      </div>
      <p className="mt-3 text-xs leading-5 text-neutral-500">Original Evater practice papers, not official NCERT papers. 15 questions · 22 marks each.</p>
    </div>
  </section>
}
