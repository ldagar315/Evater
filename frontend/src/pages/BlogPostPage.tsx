import React, { useEffect, useMemo, useState } from 'react'
import { ArrowLeft, ArrowRight, Calendar, Check, ChevronRight, Clock, Home, Share2, Tag } from 'lucide-react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import Markdown from 'react-markdown'
import { Header } from '../components/layout/Header'
import { Footer } from '../components/layout/Footer'
import { BlogCard } from '../components/blog/BlogCard'
import { Seo } from '../components/seo/Seo'
import { SITE_URL } from '../components/seo/seoConfig'
import { Button } from '../components/ui/Button'
import { Card, CardContent } from '../components/ui/Card'
import { useAuthContext } from '../contexts/AuthContext'
import { getPostBySlug, getRelatedPosts } from '../data/blogPosts'
import { BlogPost } from '../types/blog'

const formatDate = (dateString: string) => new Date(dateString).toLocaleDateString('en-US', {
  year: 'numeric',
  month: 'long',
  day: 'numeric',
})

const stripLeadingMarkdownTitle = (content: string) => content.replace(/^#\s+.+(?:\r?\n){1,2}/, '')

const slugify = (value: string) => value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '')

const getTextContent = (children: React.ReactNode): string => React.Children.toArray(children).map((child) => {
  if (typeof child === 'string' || typeof child === 'number') return String(child)
  if (React.isValidElement(child)) return getTextContent(child.props.children)
  return ''
}).join('')

const getHeadings = (content: string) => Array.from(content.matchAll(/^##\s+(.+)$/gm)).map((match, index) => ({
  title: match[1].trim(),
  id: `${slugify(match[1])}-${index}`,
}))

export function BlogPostPage() {
  const { slug } = useParams<{ slug: string }>()
  const navigate = useNavigate()
  const { user } = useAuthContext()
  const [post, setPost] = useState<BlogPost | null>(null)
  const [relatedPosts, setRelatedPosts] = useState<BlogPost[]>([])
  const [loading, setLoading] = useState(true)
  const [shareStatus, setShareStatus] = useState<'idle' | 'copied' | 'error'>('idle')

  useEffect(() => {
    const foundPost = slug ? getPostBySlug(slug) : undefined
    setPost(foundPost || null)
    setRelatedPosts(foundPost ? getRelatedPosts(foundPost) : [])
    setLoading(false)
  }, [slug])

  const articleContent = useMemo(() => post ? stripLeadingMarkdownTitle(post.content) : '', [post])
  const headings = useMemo(() => getHeadings(articleContent), [articleContent])

  const handleShare = async () => {
    if (!post) return

    if (navigator.share) {
      try {
        await navigator.share({ title: post.title, text: post.excerpt, url: window.location.href })
        setShareStatus('copied')
        window.setTimeout(() => setShareStatus('idle'), 2500)
        return
      } catch (error) {
        if (error instanceof DOMException && error.name === 'AbortError') return
      }
    }

    try {
      await navigator.clipboard?.writeText(window.location.href)
      setShareStatus('copied')
    } catch {
      setShareStatus('error')
    }
    window.setTimeout(() => setShareStatus('idle'), 2500)
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-cream">
        <Header />
        <main className="mx-auto flex min-h-[50vh] max-w-3xl items-center justify-center px-4 py-16 text-center">
          <div>
            <div className="mx-auto mb-4 h-10 w-10 animate-spin rounded-full border-4 border-primary-100 border-t-primary-600" />
            <p className="font-semibold text-neutral-600">Loading article…</p>
          </div>
        </main>
        {!user && <Footer />}
      </div>
    )
  }

  if (!post) {
    return (
      <div className="min-h-screen bg-cream">
        <Header />
        <main className="mx-auto flex min-h-[60vh] max-w-2xl items-center justify-center px-4 py-16">
          <Card className="w-full text-center">
            <CardContent className="p-10 sm:p-14">
              <p className="mb-3 text-xs font-extrabold uppercase tracking-[0.16em] text-primary-700">404 article</p>
              <h1 className="text-3xl font-extrabold text-dark">This guide has moved</h1>
              <p className="mx-auto mt-3 max-w-md leading-7 text-neutral-500">The article may have been unpublished, or the URL may be incomplete.</p>
              <Button onClick={() => navigate('/blog')} className="mt-7">Back to the blog</Button>
            </CardContent>
          </Card>
        </main>
        {!user && <Footer />}
      </div>
    )
  }

  const path = `/blog/${post.slug}`
  const articleSchema = {
    '@context': 'https://schema.org',
    '@graph': [
      {
        '@type': 'Article',
        headline: post.title,
        description: post.seo?.meta_description || post.excerpt,
        image: [post.featured_image],
        datePublished: post.published_date,
        dateModified: post.updated_date || post.published_date,
        author: { '@type': 'Person', name: post.author.name },
        publisher: { '@type': 'Organization', name: 'Evater', url: SITE_URL },
        mainEntityOfPage: { '@type': 'WebPage', '@id': `${SITE_URL}${path}` },
        articleSection: post.category,
        keywords: post.seo?.keywords?.join(', '),
      },
      {
        '@type': 'BreadcrumbList',
        itemListElement: [
          { '@type': 'ListItem', position: 1, name: 'Home', item: SITE_URL },
          { '@type': 'ListItem', position: 2, name: 'Blog', item: `${SITE_URL}/blog` },
          { '@type': 'ListItem', position: 3, name: post.title, item: `${SITE_URL}${path}` },
        ],
      },
      ...(post.faqs?.length ? [{
        '@type': 'FAQPage',
        mainEntity: post.faqs.map((faq) => ({
          '@type': 'Question',
          name: faq.question,
          acceptedAnswer: { '@type': 'Answer', text: faq.answer },
        })),
      }] : []),
    ],
  }

  const markdownComponents = {
    h2: ({ children }: { children?: React.ReactNode }) => {
      const title = getTextContent(children)
      const heading = headings.find((item) => item.title === title)
      return <h2 id={heading?.id || slugify(title)}>{children}</h2>
    },
    h3: ({ children }: { children?: React.ReactNode }) => <h3>{children}</h3>,
    a: ({ href, children }: { href?: string; children?: React.ReactNode }) => {
      if (href?.startsWith('/')) return <Link to={href}>{children}</Link>
      return <a href={href} target="_blank" rel="noreferrer">{children}</a>
    },
  }

  return (
    <div className="min-h-screen bg-cream font-sans">
      <Seo
        title={post.seo?.meta_title || `${post.title} | Evater Blog`}
        description={post.seo?.meta_description || post.excerpt}
        path={path}
        image={post.featured_image}
        type="article"
        publishedTime={post.published_date}
        modifiedTime={post.updated_date || post.published_date}
        author={post.author.name}
        keywords={post.seo?.keywords}
        jsonLd={articleSchema}
      />
      <Header />

      <main>
        <nav aria-label="Breadcrumb" className="border-b border-neutral-200 bg-white">
          <div className="mx-auto flex max-w-7xl items-center gap-2 px-4 py-4 text-sm text-neutral-500 sm:px-6 lg:px-8">
            <Link to="/" className="inline-flex min-h-10 items-center gap-2 rounded-lg px-2 hover:bg-primary-50 hover:text-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500">
              <Home className="h-4 w-4" aria-hidden="true" /> <span className="sr-only sm:not-sr-only">Home</span>
            </Link>
            <ChevronRight className="h-4 w-4" aria-hidden="true" />
            <Link to="/blog" className="rounded-lg px-2 py-2 hover:bg-primary-50 hover:text-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500">Blog</Link>
            <ChevronRight className="h-4 w-4 shrink-0" aria-hidden="true" />
            <span className="truncate rounded-lg px-2 py-2 font-semibold text-dark" aria-current="page">{post.title}</span>
          </div>
        </nav>

        <header className="relative isolate overflow-hidden bg-[#173b38] text-white">
          <div className="absolute inset-0 bg-gradient-to-br from-primary-900/60 via-[#173b38]/90 to-dark/95" />
          <img src={post.featured_image} alt="" className="absolute inset-0 h-full w-full object-cover opacity-20 mix-blend-screen" width={1400} height={900} />
          <div className="relative mx-auto max-w-4xl px-4 py-16 sm:px-6 sm:py-20 lg:px-8 lg:py-24">
            <Link to={`/blog?category=${encodeURIComponent(post.category.toLowerCase().replace(/[^a-z0-9]+/g, '-'))}`} className="inline-flex min-h-10 items-center rounded-full border border-secondary-300/40 bg-secondary-300/15 px-4 text-xs font-extrabold uppercase tracking-[0.16em] text-secondary-100 hover:bg-secondary-300/25 focus:outline-none focus:ring-2 focus:ring-secondary-300">
              {post.category}
            </Link>
            <h1 className="mt-7 max-w-4xl text-4xl font-extrabold leading-[1.08] tracking-[-0.04em] sm:text-5xl lg:text-6xl">{post.title}</h1>
            <p className="mt-6 max-w-3xl text-lg leading-8 text-primary-50/75">{post.excerpt}</p>
            <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-4 text-sm font-semibold text-white/70">
              <span className="flex items-center gap-2"><Calendar className="h-4 w-4 text-secondary-200" aria-hidden="true" /><time dateTime={post.published_date}>{formatDate(post.published_date)}</time></span>
              <span className="flex items-center gap-2"><Clock className="h-4 w-4 text-secondary-200" aria-hidden="true" />{post.read_time} min read</span>
              <span>By {post.author.name}</span>
            </div>
          </div>
        </header>

        <section className="relative mx-auto grid max-w-7xl gap-10 px-4 py-12 sm:px-6 sm:py-16 lg:grid-cols-[15rem_minmax(0,1fr)] lg:px-8 lg:py-20">
          {headings.length > 0 && (
            <aside className="h-fit rounded-2xl border border-neutral-200 bg-white p-5 lg:sticky lg:top-24">
              <p className="mb-3 text-xs font-extrabold uppercase tracking-[0.16em] text-primary-700">In this guide</p>
              <ol className="space-y-2">
                {headings.map((heading, index) => (
                  <li key={heading.id}>
                    <a href={`#${heading.id}`} className="flex gap-2 rounded-lg px-2 py-2 text-sm font-semibold leading-5 text-neutral-600 hover:bg-primary-50 hover:text-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500">
                      <span className="text-primary-600">{String(index + 1).padStart(2, '0')}</span>{heading.title}
                    </a>
                  </li>
                ))}
              </ol>
            </aside>
          )}

          <article className="min-w-0 rounded-3xl border border-neutral-200 bg-white p-6 shadow-sm sm:p-10 lg:p-14">
            <div className="prose prose-lg max-w-none prose-headings:font-extrabold prose-headings:tracking-tight prose-headings:text-dark prose-h2:scroll-mt-28 prose-p:leading-8 prose-p:text-neutral-600 prose-a:font-bold prose-a:text-primary-700 prose-a:no-underline hover:prose-a:underline prose-strong:text-dark prose-li:text-neutral-600 prose-blockquote:border-primary-500 prose-blockquote:bg-primary-50 prose-blockquote:text-neutral-700 prose-img:rounded-2xl">
              <Markdown components={markdownComponents}>{articleContent}</Markdown>
            </div>

            {post.faqs && post.faqs.length > 0 && (
              <section aria-labelledby="faq-heading" className="mt-12 border-t border-neutral-200 pt-10">
                <p className="mb-2 text-xs font-extrabold uppercase tracking-[0.16em] text-primary-700">Quick answers</p>
                <h2 id="faq-heading" className="text-2xl font-extrabold tracking-tight text-dark">Frequently asked questions</h2>
                <div className="mt-6 space-y-5">
                  {post.faqs.map((faq) => (
                    <details key={faq.question} className="group rounded-2xl border border-neutral-200 bg-cream/60 p-5">
                      <summary className="cursor-pointer list-none pr-6 font-extrabold text-dark marker:hidden group-open:text-primary-700">{faq.question}</summary>
                      <p className="mt-3 leading-7 text-neutral-600">{faq.answer}</p>
                    </details>
                  ))}
                </div>
              </section>
            )}

            {post.tags.length > 0 && (
              <div className="mt-12 flex flex-wrap items-center gap-2 border-t border-neutral-200 pt-8">
                <Tag className="mr-1 h-4 w-4 text-neutral-400" aria-hidden="true" />
                {post.tags.map((tag) => <span key={tag} className="rounded-full border border-neutral-200 bg-neutral-50 px-3 py-1 text-sm font-semibold text-neutral-600">#{tag}</span>)}
              </div>
            )}

            <div className="mt-10 flex flex-col gap-6 border-t border-neutral-200 pt-8 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <p className="text-sm font-semibold text-neutral-500">Written by</p>
                <p className="mt-1 font-extrabold text-dark">{post.author.name}</p>
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <Button variant="outline" size="sm" onClick={handleShare} aria-label="Share this article">
                  {shareStatus === 'copied' ? <Check className="mr-2 h-4 w-4" aria-hidden="true" /> : <Share2 className="mr-2 h-4 w-4" aria-hidden="true" />}
                  {shareStatus === 'copied' ? 'Link copied' : shareStatus === 'error' ? 'Copy unavailable' : 'Share article'}
                </Button>
                <span className="sr-only" aria-live="polite">{shareStatus === 'copied' ? 'Article link copied.' : ''}</span>
              </div>
            </div>
          </article>
        </section>

        {relatedPosts.length > 0 && (
          <section aria-labelledby="related-heading" className="border-t border-neutral-200 bg-white py-16 sm:py-20">
            <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
              <div className="mb-8 flex items-end justify-between gap-4">
                <div>
                  <p className="mb-2 text-xs font-extrabold uppercase tracking-[0.16em] text-primary-700">Keep learning</p>
                  <h2 id="related-heading" className="text-3xl font-extrabold tracking-tight text-dark">More like this</h2>
                </div>
                <Link to="/blog" className="hidden min-h-11 items-center gap-1 rounded-xl px-2 text-sm font-extrabold text-primary-700 hover:bg-primary-50 sm:inline-flex">All articles <ArrowRight className="h-4 w-4" aria-hidden="true" /></Link>
              </div>
              <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
                {relatedPosts.map((relatedPost) => <BlogCard key={relatedPost.id} post={relatedPost} />)}
              </div>
            </div>
          </section>
        )}

        <section className="bg-[#173b38] py-16 text-white sm:py-20">
          <div className="mx-auto flex max-w-5xl flex-col gap-8 px-4 sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:px-8">
            <div>
              <p className="mb-3 text-xs font-extrabold uppercase tracking-[0.16em] text-secondary-200">Turn reading into progress</p>
              <h2 className="max-w-2xl text-3xl font-extrabold tracking-tight sm:text-4xl">Try a practice set, then learn from the mistakes.</h2>
              <p className="mt-3 max-w-2xl leading-7 text-primary-50/70">Evater helps you move from “I read it” to “I can use it.”</p>
            </div>
            <div className="flex shrink-0 flex-wrap gap-3">
              <Link to={user ? '/home' : '/auth'} className="inline-flex min-h-12 items-center justify-center rounded-xl bg-secondary-300 px-5 text-sm font-extrabold text-dark transition-colors hover:bg-secondary-200 focus:outline-none focus:ring-2 focus:ring-secondary-200 focus:ring-offset-2 focus:ring-offset-[#173b38]">{user ? 'Go to dashboard' : 'Get started free'}</Link>
              <Link to="/blog" className="inline-flex min-h-12 items-center justify-center rounded-xl border border-white/30 px-5 text-sm font-extrabold text-white transition-colors hover:bg-white/10 focus:outline-none focus:ring-2 focus:ring-white">Read more articles <ArrowLeft className="ml-2 h-4 w-4 rotate-180" aria-hidden="true" /></Link>
            </div>
          </div>
        </section>
      </main>

      {!user && <Footer />}
    </div>
  )
}
