import React, { useEffect, useMemo, useState } from 'react'
import { ArrowRight, BookOpen, Check, Search, Sparkles, X } from 'lucide-react'
import { Link, useSearchParams } from 'react-router-dom'
import { Header } from '../components/layout/Header'
import { Footer } from '../components/layout/Footer'
import { BlogCard } from '../components/blog/BlogCard'
import { Seo } from '../components/seo/Seo'
import { SITE_URL } from '../components/seo/seoConfig'
import { Card, CardContent } from '../components/ui/Card'
import { useAuthContext } from '../contexts/AuthContext'
import { blogCategories, blogPosts } from '../data/blogPosts'

const postsPerPage = 9

export function BlogPage() {
  const { user } = useAuthContext()
  const [searchParams, setSearchParams] = useSearchParams()
  const [currentPage, setCurrentPage] = useState(1)
  const searchQuery = searchParams.get('q') || ''
  const selectedCategory = searchParams.get('category') || ''

  const updateFilters = (nextQuery: string, nextCategory = selectedCategory) => {
    const nextParams = new URLSearchParams()
    if (nextQuery.trim()) nextParams.set('q', nextQuery.trim())
    if (nextCategory) nextParams.set('category', nextCategory)
    setSearchParams(nextParams)
    setCurrentPage(1)
  }

  useEffect(() => {
    setCurrentPage(1)
  }, [searchQuery, selectedCategory])

  const filteredPosts = useMemo(() => {
    const normalizedQuery = searchQuery.trim().toLowerCase()
    const normalizedCategory = selectedCategory.trim().toLowerCase()

    return blogPosts
      .filter((post) => post.status === 'published')
      .filter((post) => !normalizedCategory || post.category.toLowerCase() === normalizedCategory || post.category.toLowerCase().replace(/[^a-z0-9]+/g, '-') === normalizedCategory)
      .filter((post) => {
        if (!normalizedQuery) return true
        return [post.title, post.excerpt, post.content, post.category, ...post.tags]
          .some((value) => value.toLowerCase().includes(normalizedQuery))
      })
      .sort((a, b) => new Date(b.published_date).getTime() - new Date(a.published_date).getTime())
  }, [searchQuery, selectedCategory])

  const featuredPost = useMemo(
    () => filteredPosts.find((post) => post.is_featured) || filteredPosts[0],
    [filteredPosts],
  )
  const showFeatured = !searchQuery && !selectedCategory && currentPage === 1 && Boolean(featuredPost)
  const totalPages = Math.ceil(filteredPosts.length / postsPerPage)
  const startIndex = (currentPage - 1) * postsPerPage
  const paginatedPosts = filteredPosts.slice(startIndex, startIndex + postsPerPage)
  const latestPosts = showFeatured && featuredPost
    ? paginatedPosts.filter((post) => post.id !== featuredPost.id)
    : paginatedPosts

  const pageDescription = searchQuery || selectedCategory
    ? `Browse Evater articles${selectedCategory ? ` in ${selectedCategory}` : ''}${searchQuery ? ` matching “${searchQuery}”` : ''}.`
    : 'Practical study strategies, Class 8 Science revision plans, and feedback-led learning ideas for students in Classes 6–10.'

  const collectionSchema = {
    '@context': 'https://schema.org',
    '@type': 'CollectionPage',
    name: 'Evater Learning Journal',
    description: pageDescription,
    url: `${SITE_URL}/blog`,
    mainEntity: {
      '@type': 'ItemList',
      itemListElement: filteredPosts.map((post, index) => ({
        '@type': 'ListItem',
        position: index + 1,
        url: `${SITE_URL}/blog/${post.slug}`,
        name: post.title,
      })),
    },
  }

  return (
    <div className="min-h-screen bg-cream font-sans">
      <Seo
        title="Study Tips, Class 8 Science & Exam Prep | Evater Blog"
        description="Practical study strategies, Class 8 Science revision plans, and feedback-led learning ideas for students in Classes 6–10."
        path="/blog"
        jsonLd={collectionSchema}
        keywords={['Class 8 Science', 'study tips for students', 'exam preparation', 'NCERT revision']}
      />
      <Header />

      <main>
        <section className="relative isolate overflow-hidden border-b border-neutral-200 bg-[#173b38] text-white">
          <div className="absolute -right-24 -top-32 h-80 w-80 rounded-full border-[36px] border-secondary-300/20" aria-hidden="true" />
          <div className="absolute -bottom-32 left-[-4rem] h-72 w-72 rounded-full bg-primary-400/10 blur-2xl" aria-hidden="true" />
          <div className="relative mx-auto grid max-w-7xl gap-12 px-4 py-16 sm:px-6 sm:py-20 lg:grid-cols-[1.1fr_0.9fr] lg:items-end lg:px-8 lg:py-24">
            <div>
              <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-secondary-300/30 bg-secondary-300/10 px-3 py-1.5 text-xs font-extrabold uppercase tracking-[0.16em] text-secondary-200">
                <Sparkles className="h-3.5 w-3.5" aria-hidden="true" /> Evater learning journal
              </div>
              <h1 className="max-w-3xl text-4xl font-extrabold leading-[1.05] tracking-[-0.04em] sm:text-5xl lg:text-7xl">
                Study smarter. Make progress visible.
              </h1>
              <p className="mt-6 max-w-2xl text-base leading-8 text-primary-50/75 sm:text-lg">
                Clear, practical ideas for students in Classes 6–10: NCERT revision plans, Science explanations, and better ways to learn from every attempt.
              </p>
            </div>

            <form
              role="search"
              onSubmit={(event) => event.preventDefault()}
              className="rounded-3xl border border-white/15 bg-white/10 p-4 shadow-2xl shadow-black/10 backdrop-blur-sm sm:p-5"
            >
              <label htmlFor="blog-search" className="mb-3 block text-sm font-bold text-white/80">
                What are you working on?
              </label>
              <div className="relative">
                <Search className="pointer-events-none absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-primary-100/60" aria-hidden="true" />
                <input
                  id="blog-search"
                  type="search"
                  value={searchQuery}
                  onChange={(event) => updateFilters(event.target.value)}
                  placeholder="Try “Class 8 Science”"
                  className="min-h-14 w-full rounded-2xl border border-white/15 bg-white px-12 pr-12 text-base font-semibold text-dark outline-none ring-primary-300 placeholder:text-neutral-400 focus:ring-4"
                />
                {searchQuery && (
                  <button
                    type="button"
                    onClick={() => {
                      setSearchParams(new URLSearchParams())
                      setCurrentPage(1)
                    }}
                    className="absolute right-3 top-1/2 flex h-9 w-9 -translate-y-1/2 items-center justify-center rounded-xl text-neutral-500 hover:bg-neutral-100 hover:text-dark focus:outline-none focus:ring-2 focus:ring-primary-500"
                    aria-label="Clear search"
                  >
                    <X className="h-4 w-4" aria-hidden="true" />
                  </button>
                )}
              </div>
              <p className="mt-3 text-xs leading-5 text-primary-50/60">Search titles, topics, tags, and article content.</p>
            </form>
          </div>
        </section>

        <section className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8 lg:py-16">
          <div className="mb-12 flex flex-col gap-6 border-b border-neutral-200 pb-8 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <p className="mb-3 text-xs font-extrabold uppercase tracking-[0.16em] text-primary-700">Browse by intent</p>
              <h2 className="text-3xl font-extrabold tracking-tight text-dark sm:text-4xl">Find your next useful idea</h2>
            </div>
            <nav aria-label="Blog categories" className="flex flex-wrap gap-2 lg:max-w-2xl lg:justify-end">
              <Link
                to="/blog"
                className={`inline-flex min-h-11 items-center gap-2 rounded-full border px-4 text-sm font-bold transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 ${!selectedCategory ? 'border-dark bg-dark text-white' : 'border-neutral-300 bg-white text-neutral-600 hover:border-primary-300 hover:text-primary-700'}`}
              >
                {!selectedCategory && <Check className="h-4 w-4" aria-hidden="true" />} All articles
              </Link>
              {blogCategories.map((category) => {
                const active = selectedCategory.toLowerCase() === category.name.toLowerCase() || selectedCategory.toLowerCase() === category.slug
                return (
                  <Link
                    key={category.id}
                    to={`/blog?category=${encodeURIComponent(category.slug)}`}
                    className={`inline-flex min-h-11 items-center rounded-full border px-4 text-sm font-bold transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 ${active ? 'border-primary-700 bg-primary-700 text-white' : 'border-neutral-300 bg-white text-neutral-600 hover:border-primary-300 hover:text-primary-700'}`}
                  >
                    {category.name} <span className="ml-1.5 text-xs opacity-70">{category.post_count}</span>
                  </Link>
                )
              })}
            </nav>
          </div>

          {showFeatured && featuredPost && (
            <section aria-labelledby="featured-heading" className="mb-16">
              <div className="mb-6 flex items-end justify-between gap-4">
                <div>
                  <p className="mb-2 text-xs font-extrabold uppercase tracking-[0.16em] text-primary-700">Start here</p>
                  <h2 id="featured-heading" className="text-2xl font-extrabold tracking-tight text-dark sm:text-3xl">Featured guide</h2>
                </div>
                <Link to={`/blog/${featuredPost.slug}`} className="hidden min-h-11 items-center gap-1 rounded-xl px-2 text-sm font-bold text-primary-700 hover:bg-primary-50 sm:inline-flex">
                  Read the guide <ArrowRight className="h-4 w-4" aria-hidden="true" />
                </Link>
              </div>
              <BlogCard post={featuredPost} featured />
            </section>
          )}

          <section aria-labelledby="articles-heading">
            <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <p className="mb-2 text-xs font-extrabold uppercase tracking-[0.16em] text-primary-700">{searchQuery || selectedCategory ? 'Your results' : 'Keep exploring'}</p>
                <h2 id="articles-heading" className="text-2xl font-extrabold tracking-tight text-dark sm:text-3xl">
                  {searchQuery || selectedCategory ? 'Search results' : 'Latest articles'}
                </h2>
              </div>
              <p className="text-sm font-semibold text-neutral-500" aria-live="polite">
                Showing {latestPosts.length} of {filteredPosts.length} {filteredPosts.length === 1 ? 'article' : 'articles'}
              </p>
            </div>

            {latestPosts.length === 0 ? (
              <Card className="border-2 border-dashed border-neutral-200 bg-white shadow-none">
                <CardContent className="p-10 text-center sm:p-16">
                  <BookOpen className="mx-auto mb-5 h-12 w-12 text-primary-300" aria-hidden="true" />
                  <h3 className="text-xl font-extrabold text-dark">No articles found</h3>
                  <p className="mx-auto mt-2 max-w-md leading-7 text-neutral-500">Try a broader topic, or clear the filters to browse every Evater learning guide.</p>
                  <button
                    type="button"
                    onClick={() => updateFilters('')}
                    className="mt-6 inline-flex min-h-11 items-center rounded-xl px-4 text-sm font-extrabold text-primary-700 hover:bg-primary-50 focus:outline-none focus:ring-2 focus:ring-primary-500"
                  >
                    Clear filters
                  </button>
                </CardContent>
              </Card>
            ) : (
              <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
                {latestPosts.map((post) => <BlogCard key={post.id} post={post} />)}
              </div>
            )}
          </section>

          {totalPages > 1 && (
            <nav aria-label="Blog pagination" className="mt-12 flex items-center justify-center gap-2">
              {Array.from({ length: totalPages }, (_, index) => index + 1).map((page) => (
                <button
                  key={page}
                  type="button"
                  onClick={() => setCurrentPage(page)}
                  className={`flex min-h-11 min-w-11 items-center justify-center rounded-xl text-sm font-bold focus:outline-none focus:ring-2 focus:ring-primary-500 ${currentPage === page ? 'bg-dark text-white' : 'border border-neutral-200 bg-white text-neutral-600 hover:border-primary-300 hover:text-primary-700'}`}
                  aria-current={currentPage === page ? 'page' : undefined}
                >
                  {page}
                </button>
              ))}
            </nav>
          )}
        </section>
      </main>

      {!user && <Footer />}
    </div>
  )
}
