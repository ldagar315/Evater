import { describe, expect, it } from 'vitest'
import { resourceGuides } from './resourceGuides'
import { youtubeVideoId } from '../components/blog/resourceVideo'

const content = (id: string) => resourceGuides.find(p => p.id === id)!.content
const topic = (text: string, n: number) => text.split(`## ${n}.`)[1].split('\n## ')[0]

const additions = [
  ['resource-science-12', '2lqhJNgn_Wg', 'Food Chains & Food Webs', 'FuseSchool - Global Education'],
  ['resource-science-13', 'o8K0RTYM6Mc', 'Real World: Earth Systems', 'NASA eClips'],
] as const

describe('two source-fit ecosystem and Earth-system video placements', () => {
  for (const [id, videoId, title, publisher] of additions) {
    it(`places ${videoId} only in the intended second topic`, () => {
      const text = content(id)
      const selected = topic(text, 2)
      const url = `https://www.youtube.com/watch?v=${videoId}`
      expect(selected).toContain(`### Watch: ${title}`)
      expect(selected).toContain(publisher)
      expect(selected).toContain(`[Open the video](${url})`)
      expect(selected.match(/\[Open the video\]/g)).toHaveLength(1)
      expect(text.match(new RegExp(videoId, 'g'))).toHaveLength(1)
      expect(youtubeVideoId(url)).toBe(videoId)
      expect(selected).toContain('### Understand it')
      expect(selected).toContain('### Try it yourself')
      expect(selected).toContain('**Check:**')
    })
  }

  it('keeps four original teaching/check sections and revision in both lessons', () => {
    for (const [id] of additions) {
      const text = content(id)
      expect(text.match(/^## \d+\./gm)).toHaveLength(4)
      expect(text.match(/### Try it yourself/g)).toHaveLength(4)
      expect(text.match(/\*\*Check:\*\*/g)).toHaveLength(4)
      expect(text).toContain('## Your five-minute revision')
      expect(text).toContain('[Try an Evater sample paper](/papers/)')
      expect(text).not.toMatch(/supporting evidence|transcript|playback verified/i)
      const externalLinks = [...text.matchAll(/\]\((https?:[^)]+)\)/g)].map(m => m[1])
      for (const url of externalLinks) expect(youtubeVideoId(url)).not.toBeNull()
    }
  })

  it('changes video-bearing coverage, not the number of teaching topics', () => {
    const topics = resourceGuides.flatMap(p => p.content.split(/^## /m).slice(1).filter(t => /^\d+\./.test(t)))
    expect(topics).toHaveLength(113)
    expect(topics.filter(t => t.includes('[Open the video]'))).toHaveLength(78)
    expect(topics.filter(t => !t.includes('[Open the video]'))).toHaveLength(35)
    for (const text of topics) expect((text.match(/\[Open the video\]/g) || []).length).toBeLessThanOrEqual(1)
  })
})
