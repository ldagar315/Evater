import { useEffect } from 'react'
import { SITE_URL } from './seoConfig'

interface SeoProps {
  title: string
  description: string
  path: string
  image?: string
  type?: 'website' | 'article'
  publishedTime?: string
  modifiedTime?: string
  author?: string
  keywords?: string[]
  jsonLd?: Record<string, unknown> | Record<string, unknown>[]
}

function upsertMeta(attribute: 'name' | 'property', key: string, content: string) {
  let element = document.head.querySelector<HTMLMetaElement>(`meta[${attribute}="${key}"]`)

  if (!element) {
    element = document.createElement('meta')
    element.setAttribute(attribute, key)
    element.dataset.seoManaged = 'true'
    document.head.appendChild(element)
  }

  element.content = content
}

function upsertCanonical(href: string) {
  let element = document.head.querySelector<HTMLLinkElement>('link[rel="canonical"]')

  if (!element) {
    element = document.createElement('link')
    element.rel = 'canonical'
    element.dataset.seoManaged = 'true'
    document.head.appendChild(element)
  }

  element.href = href
}

function removeMeta(attribute: 'name' | 'property', key: string) {
  document.head.querySelector(`meta[${attribute}="${key}"]`)?.remove()
}

export function Seo({
  title,
  description,
  path,
  image = `${SITE_URL}/Evater_logo_2.png`,
  type = 'website',
  publishedTime,
  modifiedTime,
  author,
  keywords = [],
  jsonLd,
}: SeoProps) {
  useEffect(() => {
    const canonicalUrl = `${SITE_URL}${path.startsWith('/') ? path : `/${path}`}`
    const previousTitle = document.title
    document.title = title

    upsertMeta('name', 'description', description)
    upsertMeta('name', 'robots', 'index, follow, max-image-preview:large')
    upsertMeta('property', 'og:title', title)
    upsertMeta('property', 'og:description', description)
    upsertMeta('property', 'og:type', type)
    upsertMeta('property', 'og:url', canonicalUrl)
    upsertMeta('property', 'og:image', image)
    upsertMeta('property', 'og:site_name', 'Evater')
    upsertMeta('name', 'twitter:card', 'summary_large_image')
    upsertMeta('name', 'twitter:title', title)
    upsertMeta('name', 'twitter:description', description)
    upsertMeta('name', 'twitter:image', image)

    if (keywords.length > 0) {
      upsertMeta('name', 'keywords', keywords.join(', '))
    } else {
      removeMeta('name', 'keywords')
    }

    if (publishedTime) upsertMeta('property', 'article:published_time', publishedTime)
    else removeMeta('property', 'article:published_time')
    if (modifiedTime) upsertMeta('property', 'article:modified_time', modifiedTime)
    else removeMeta('property', 'article:modified_time')
    if (author) upsertMeta('property', 'article:author', author)
    else removeMeta('property', 'article:author')
    upsertCanonical(canonicalUrl)

    const previousJsonLd = document.head.querySelector('script[data-seo-jsonld]')
    previousJsonLd?.remove()

    if (jsonLd) {
      const script = document.createElement('script')
      script.type = 'application/ld+json'
      script.dataset.seoJsonld = 'true'
      script.textContent = JSON.stringify(jsonLd)
      document.head.appendChild(script)
    }

    return () => {
      document.title = previousTitle
      document.head.querySelector('script[data-seo-jsonld]')?.remove()
    }
  }, [author, description, image, jsonLd, keywords, modifiedTime, path, publishedTime, title, type])

  return null
}
