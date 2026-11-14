import { Link, useParams } from 'react-router-dom'
import Markdown from 'react-markdown'
import { Header } from '../components/layout/Header'
import { Footer } from '../components/layout/Footer'
import { Seo } from '../components/seo/Seo'
import lenses from '../../../content/reviewed/class8-2026-11-14/lenses-guide.md?raw'
import power from '../../../content/reviewed/class8-2026-11-14/power-play-guide.md?raw'
import microbes from '../../../content/reviewed/class8-2026-11-14/invisible-living-world-guide.md?raw'

export const reviewedGuidePreviews = [
  { slug: 'lenses', title: 'Lenses: predict the image before you look', scope: 'A concept guide within Light, not the complete chapter.', body: lenses },
  { slug: 'power-play', title: 'Power Play: make the multiplication visible', scope: 'Selected exponent and scientific-notation ideas, not every chapter outcome.', body: power },
  { slug: 'invisible-living-world', title: 'The Invisible Living World: connect cells, microbes and everyday changes', scope: 'A selective guide, excluding some levels-of-organisation and microalgae details.', body: microbes },
]

/** Review surface only. The import and route are removed from production builds. */
export function ReviewedContentPreviewPage() {
  const { slug } = useParams<{ slug?: string }>()
  const guide = reviewedGuidePreviews.find(item => item.slug === slug)
  return (
    <div className="min-h-screen bg-cream text-dark">
      <Seo title="Reviewed Class 8 content preview" description="Development-only review of proposed original guides." path="/reviewed-content/" robots="noindex, nofollow" />
      <Header />
      <main className="mx-auto max-w-4xl px-4 py-8 sm:px-6 sm:py-12">
        <div className="mb-8 rounded-xl border border-primary-200 bg-white p-5">
          <p className="text-sm font-bold text-primary-800">Review preview - not published</p>
          <p className="mt-2 text-sm leading-6">Owner publication decision pending. These guides are not in the live blog catalogue. Question import and practice availability are separate.</p>
          <nav aria-label="Guide previews" className="mt-4 flex flex-col gap-2 sm:flex-row sm:flex-wrap">
            {reviewedGuidePreviews.map(item => <Link key={item.slug} className="min-h-11 rounded-md border border-primary-200 px-3 py-2 text-sm font-semibold underline" to={`/reviewed-content/${item.slug}`}>{item.slug.replace(/-/g, ' ')}</Link>)}
          </nav>
        </div>
        {guide ? <article className="rounded-2xl bg-white p-5 sm:p-9">
          <p className="mb-5 text-sm leading-6 text-neutral-600">{guide.scope}</p>
          <div className="prose prose-neutral max-w-none break-words prose-headings:text-dark prose-a:break-all prose-a:text-primary-700"><Markdown>{guide.body}</Markdown></div>
          <aside className="mt-8 rounded-lg bg-cream p-4 text-sm leading-6">Practice pending publication. No exact-concept test is advertised by this preview.</aside>
        </article> : <section className="rounded-2xl bg-white p-6"><h1 className="text-2xl font-bold">{slug ? 'Preview not found' : 'Choose a reviewed guide'}</h1><p className="mt-3 leading-7">Use the links above to review the proposed student copy. No content has been published.</p></section>}
      </main>
      <Footer />
    </div>
  )
}
