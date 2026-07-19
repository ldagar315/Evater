import React, { useState } from 'react'
import { ArrowUpRight, Clock } from 'lucide-react'
import { Link } from 'react-router-dom'
import { BlogPost } from '../../types/blog'

interface BlogCardProps {
  post: BlogPost
  featured?: boolean
}

const formatDate = (dateString: string) => new Date(dateString).toLocaleDateString('en-US', {
  month: 'short',
  day: 'numeric',
  year: 'numeric',
})

export function BlogCard({ post, featured = false }: BlogCardProps) {
  const [imageSource, setImageSource] = useState(post.featured_image)

  return (
    <article className={`group overflow-hidden rounded-3xl border border-neutral-200 bg-white shadow-sm transition-all duration-300 hover:-translate-y-1 hover:border-primary-200 hover:shadow-xl hover:shadow-primary-900/5 ${featured ? 'lg:grid lg:grid-cols-[1.08fr_0.92fr]' : 'flex h-full flex-col'}`}>
      <Link
        to={`/blog/${post.slug}`}
        className={`relative block overflow-hidden bg-neutral-100 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-primary-500 ${featured ? 'min-h-[18rem] lg:min-h-[27rem]' : 'aspect-[16/10]'}`}
        aria-label={`Read ${post.title}`}
      >
        <img
          src={imageSource}
          alt=""
          className="h-full w-full object-cover transition duration-700 group-hover:scale-105"
          loading={featured ? 'eager' : 'lazy'}
          decoding="async"
          width={1400}
          height={900}
          onError={() => setImageSource('/Evater_logo_2.png')}
        />
        <div className="absolute inset-0 bg-gradient-to-t from-dark/50 via-transparent to-transparent opacity-70" />
        {featured && (
          <span className="absolute left-5 top-5 rounded-full bg-secondary-400 px-3 py-1 text-xs font-extrabold uppercase tracking-[0.16em] text-dark shadow-lg">
            Editor's pick
          </span>
        )}
      </Link>

      <div className={`flex flex-1 flex-col ${featured ? 'justify-center p-6 sm:p-8 lg:p-10' : 'p-6'}`}>
        <div className="mb-4 flex flex-wrap items-center gap-x-3 gap-y-2 text-xs font-bold uppercase tracking-[0.12em] text-primary-700">
          <span>{post.category}</span>
          <span className="h-1 w-1 rounded-full bg-neutral-300" aria-hidden="true" />
          <time dateTime={post.published_date}>{formatDate(post.published_date)}</time>
        </div>

        <h3 className={`${featured ? 'text-2xl sm:text-3xl lg:text-4xl' : 'text-xl'} font-extrabold leading-tight tracking-tight text-dark`}>
          <Link to={`/blog/${post.slug}`} className="rounded-md outline-none transition-colors hover:text-primary-700 focus:ring-2 focus:ring-primary-500">
            {post.title}
          </Link>
        </h3>

        <p className={`${featured ? 'mt-5 text-base sm:text-lg' : 'mt-3 text-sm'} line-clamp-3 leading-relaxed text-neutral-600`}>
          {post.excerpt}
        </p>

        <div className="mt-auto flex items-center justify-between gap-4 pt-6">
          <div className="flex items-center gap-2 text-sm text-neutral-500">
            <Clock className="h-4 w-4 text-primary-600" aria-hidden="true" />
            <span>{post.read_time} min read</span>
          </div>
          <Link
            to={`/blog/${post.slug}`}
            className="inline-flex min-h-11 items-center gap-1 rounded-xl px-2 text-sm font-extrabold text-dark transition-colors hover:text-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500"
          >
            Read article
            <ArrowUpRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" aria-hidden="true" />
          </Link>
        </div>
      </div>
    </article>
  )
}
