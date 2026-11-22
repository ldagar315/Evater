import { youtubeVideoId } from './resourceVideo'

type ResourceVideoEmbedProps = {
  videoUrl: string
  title: string
}

/** Keep the embedded lesson primary, with an independent route if the player fails. */
export function ResourceVideoEmbed({ videoUrl, title }: ResourceVideoEmbedProps) {
  const videoId = youtubeVideoId(videoUrl)
  if (!videoId) return null

  return <span className="not-prose my-5 block">
    <iframe
      className="aspect-video w-full rounded-xl border-0 bg-neutral-900"
      src={`https://www.youtube-nocookie.com/embed/${videoId}`}
      title={`YouTube video ${videoId} for ${title}`}
      loading="lazy"
      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
      referrerPolicy="strict-origin-when-cross-origin"
      allowFullScreen
    />
    <span className="mt-2 block text-sm text-neutral-600">
      Video not loading?{' '}
      <a className="font-medium text-teal-700 underline underline-offset-2 hover:text-teal-900" href={videoUrl} target="_blank" rel="noopener noreferrer">
        Watch on YouTube<span className="sr-only"> (opens in a new tab)</span>
      </a>
    </span>
  </span>
}
