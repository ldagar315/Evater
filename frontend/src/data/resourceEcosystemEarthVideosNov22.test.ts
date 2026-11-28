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
    const separation = content('resource-science-8').includes('[Open the video](https://www.youtube.com/watch?v=vi_SJBnxmHo)') ? 1 : 0
    const area = content('resource-maths-14').includes("[Open the video](https://www.youtube.com/watch?v=iNZZIb_-PA8)") ? 1 : 0
    const binomial = content('resource-maths-6').includes("[Open the video](https://www.youtube.com/watch?v=Xy8NKUoyy98)") ? 1 : 0
    const force = content('resource-science-5').includes('[Open the video](https://www.youtube.com/watch?v=SYU1TKQru1c)') ? 1 : 0
    const composition = content('resource-science-8').includes('[Open the video](https://www.youtube.com/watch?v=nUzOXy9V-K0)') ? 1 : 0
    const alloy = content('resource-science-8').includes('[Open the video](https://www.youtube.com/watch?v=KgUmNQD6m5Q)') ? 1 : 0
    const massWeight = content('resource-science-5').includes('[Open the video](https://www.youtube.com/watch?v=rFdbY_V7vIo)') ? 1 : 0
    const pythagoras0 = content('resource-maths-9').includes("[Open the video](https://www.youtube.com/watch?v=UKaoHO4ClS4)") ? 1 : 0
    const pythagoras1 = content('resource-maths-9').includes("[Open the video](https://www.youtube.com/watch?v=Eo04qjLRiSY)") ? 1 : 0
    const rotation = content('resource-science-11').includes('[Open the video](https://www.youtube.com/watch?v=l64YwNl1wr0)') ? 1 : 0
    const soil = content('resource-science-12').includes('[Open the video](https://www.youtube.com/watch?v=K7YpPgbmSew)') ? 1 : 0
    const digitalRoot = content('resource-maths-5').includes('[Open the video](https://www.youtube.com/watch?v=dTXMhOGI9Z8)') ? 1 : 0
    const variables = content('resource-science-1').includes('[Open the video](https://www.youtube.com/watch?v=BXFU86GNmrg)') ? 1 : 0
    const leap = content('resource-science-11').includes('[Open the video](https://www.youtube.com/watch?v=yd1i3vkkh-0)') ? 1 : 0
    const cyclone = content('resource-science-6').includes('[Open the video](https://www.youtube.com/watch?v=xKubdY2mHXc)') ? 1 : 0
    const disease = content('resource-science-3').includes('[Open the video](https://www.youtube.com/watch?v=dbd5iydu3EY)') ? 1 : 0
    const kite = content('resource-maths-4').includes('[Open the video](https://www.youtube.com/watch?v=-IRXWg2oLY4)') ? 1 : 0
    const gas = content('resource-science-9').includes('[Open the video](https://www.youtube.com/watch?v=wK4reyh86w0)') ? 1 : 0
    const microbe = content('resource-science-2').includes('[Open the video](https://www.youtube.com/watch?v=4HdDlpAB2hU)') ? 1 : 0
    const expansion = content('resource-maths-6').includes('[Open the video](https://www.youtube.com/watch?v=3NHSwiv_pSE)') ? 1 : 0
    const parity = content('resource-maths-5').includes('[Open the video](https://www.youtube.com/watch?v=D0CkdFSWLOA)') ? 1 : 0
    const remainders = content('resource-maths-5').includes('[Open the video](https://www.youtube.com/watch?v=P1qyjdh_sIw)') ? 1 : 0
    const solution = content('resource-science-9').includes('[Open the video](https://www.youtube.com/watch?v=kYgGI9Ox3gU)') ? 1 : 0
    const diagonals = content('resource-maths-4').includes('[Open the video](https://www.youtube.com/watch?v=PC9mbAwE3zk)') ? 1 : 0
    const profit = content('resource-maths-8').includes('[Open the video](https://www.youtube.com/watch?v=aB_aXOWI_FI)') ? 1 : 0
    const habitat = content('resource-science-13').includes('[Open the video](https://www.youtube.com/watch?v=bs9e6ovISbs)') ? 1 : 0
    const multiples = content('resource-maths-5').includes('[Open the video](https://www.youtube.com/watch?v=nCq3rxCG2MA)') ? 1 : 0
    const calendarGrid = content('resource-maths-13').includes('[Open the video](https://www.youtube.com/watch?v=7VMsMtk3QdI)') ? 1 : 0
    const digits = content('resource-maths-5').includes('[Open the video](https://www.youtube.com/watch?v=_dT97MUYuzs)') ? 1 : 0
    const lunar = content('resource-science-11').includes('[Open the video](https://www.youtube.com/watch?v=-T97d6dsfhE)') ? 1 : 0
    const reproduction = content('resource-science-13').includes('[Open the video](https://www.youtube.com/watch?v=p4652vo5Oic)') ? 1 : 0
    const cubes = content('resource-maths-9').includes('[Open the video](https://www.youtube.com/watch?v=1BSFyEIY2BY)') ? 1 : 0
    const products = content('resource-maths-13').includes('[Open the video](https://www.youtube.com/watch?v=0Ho8Az8y4Mw)') ? 1 : 0
    const signs = content('resource-science-3').includes('[Open the video](https://www.youtube.com/watch?v=LhrbUNp2gi0)') ? 1 : 0
    const heating = content('resource-science-4').includes('[Open the video](https://www.youtube.com/watch?v=zzmHjg0PORY)') ? 1 : 0
    expect(topics.filter(t => t.includes('[Open the video]'))).toHaveLength(78 + separation + area + binomial + force + composition + alloy + massWeight + pythagoras0 + pythagoras1 + rotation + soil + digitalRoot + variables + leap + cyclone + disease + kite + gas + microbe + expansion + parity + remainders + solution + diagonals + profit + habitat + multiples + calendarGrid + heating + digits + lunar + reproduction + cubes + products + signs)
    expect(topics.filter(t => !t.includes('[Open the video]'))).toHaveLength(35 - separation - area - binomial - force - composition - alloy - massWeight - pythagoras0 - pythagoras1 - rotation - soil - digitalRoot - variables - leap - cyclone - disease - kite - gas - microbe - expansion - parity - remainders - solution - diagonals - profit - habitat - multiples - calendarGrid - heating - digits - lunar - reproduction - cubes - products - signs)
    for (const text of topics) expect((text.match(/\[Open the video\]/g) || []).length).toBeLessThanOrEqual(1)
  })
})
