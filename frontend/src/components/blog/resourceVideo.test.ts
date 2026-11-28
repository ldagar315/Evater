import { describe, expect, it } from 'vitest'
import { youtubeVideoId } from './resourceVideo'

describe('resource video allowlist', () => {
  it('accepts the supported YouTube URL forms', () => {
    for (const url of ['https://www.youtube.com/watch?v=abcdefghijk&t=20', 'https://youtu.be/abcdefghijk', 'https://youtube.com/embed/abcdefghijk', 'https://m.youtube.com/shorts/abcdefghijk']) expect(youtubeVideoId(url)).toBe('abcdefghijk')
  })
  it('does not turn other resources or malformed URLs into embeds', () => {
    for (const url of ['https://bbc.co.uk/video/abcdefghijk', 'https://youtube.com.evil.test/watch?v=abcdefghijk', 'javascript:alert(1)', 'http://youtube.com/watch?v=abcdefghijk', 'https://youtube.com/watch?v=bad', '/papers/']) expect(youtubeVideoId(url)).toBeNull()
  })
})
