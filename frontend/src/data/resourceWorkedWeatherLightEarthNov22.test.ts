import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
const weather = readFileSync(new URL('./resourceGuides/12-science-6.md', import.meta.url), 'utf8')
const light = readFileSync(new URL('./resourceGuides/16-science-10.md', import.meta.url), 'utf8')
const earth = readFileSync(new URL('./resourceGuides/18-science-13.md', import.meta.url), 'utf8')
const block = (s: string, n: number) => s.split(`## ${n}.`)[1].split('\n## ')[0]
describe('six Science reasoning and safety transfers', () => {
  it('separates higher-to-lower pressure flow from the existence of pressure', () => {
    const flow = (sea: number, land: number) => sea > land ? 'sea to land' : sea < land ? 'land to sea' : 'no difference'
    expect(flow(1014, 1008)).toBe('sea to land')
    expect(flow(1006, 1011)).toBe('land to sea')
    expect(flow(1010, 1010)).toBe('no difference')
    expect(1010).toBeGreaterThan(0)
    for (const text of ['does not leave a vacuum', 'ignoring other weather and rotation', 'they do not mean zero pressure']) expect(block(weather, 2)).toContain(text)
  })
  it('restarts a fictional lightning minimum from the LAST thunder and preserves shelter limits', () => {
    const thunderMinutes = [14 * 60 + 6, 14 * 60 + 14, 14 * 60 + 21]
    const earliest = Math.max(...thunderMinutes) + 30
    expect([Math.floor(earliest / 60), earliest % 60]).toEqual([14, 51])
    expect(14 * 60 + 44).toBeLessThan(earliest)
    for (const text of ['14:51, not 14:44', 'not an automatic all-clear', 'Rain stopping does not show', 'isolated tree, open shed or covered porch is not safe shelter', 'light travels much faster than sound']) expect(block(weather, 3)).toContain(text)
  })
  it('connects cyclone mechanism to warnings without making a lull an all-clear', () => {
    for (const text of ['condenses, it releases heat', "Earth's rotation", 'Warm water alone does not guarantee', 'official evacuation', 'Calm during a cyclone may be the eye', 'wait for the official all-clear']) expect(block(weather, 4)).toContain(text)
    expect(weather).not.toMatch(/keep (?:doors|windows).*open|make emergency repairs|crouch down/i)
  })
  it('independently checks representative refraction directions and normal incidence', () => {
    const refractedDegrees = (n1: number, n2: number, degrees: number) => Math.asin(n1 / n2 * Math.sin(degrees * Math.PI / 180)) * 180 / Math.PI
    expect(refractedDegrees(1, 1.5, 30)).toBeLessThan(30)
    expect(refractedDegrees(1.5, 1, 20)).toBeGreaterThan(20)
    expect(refractedDegrees(1, 1.5, 0)).toBe(0)
    expect(1 / 1.5).not.toBe(1)
    for (const text of ['without bending, but its speed changes', 'C bends away from the normal', 'primarily uses refraction', 'Never use a lens to look at the Sun']) expect(block(light, 3)).toContain(text)
  })
  it('computes moving-object plane-mirror distance and rejects a screen image behind the mirror', () => {
    for (const [start, movement] of [[18, 5], [20, 3], [11, 4]]) {
      const after = start - movement
      expect(2 * start - 2 * after).toBe(2 * movement)
    }
    expect((18 - 5) * 2).toBe(26)
    for (const text of ['13 cm behind', 'decreased by 10 cm', 'cannot catch this virtual image', 'extensions, not actual light', 'severe eye injury or fire']) expect(block(light, 4)).toContain(text)
  })
  it('keeps hypothetical habitability evidence distinct from a detection of life', () => {
    for (const text of ['usable energy', 'still depend on resources', 'not proof of an inhabited world', 'Neither world is proven inhabited', 'checking non-living explanations', 'not a confirmed discovery']) expect(block(earth, 1)).toContain(text)
    expect(block(earth, 1)).not.toMatch(/all life needs oxygen|all organisms need sunlight|life has been found on another planet/i)
  })
  it('retains current video inventory, try/check structure and revision', () => {
    for (const [s, videos] of [[weather, 3], [light, 4], [earth, 1]] as const) {
      const added = s === earth && s.includes('[Open the video](https://www.youtube.com/watch?v=o8K0RTYM6Mc)')
      expect((s.match(/\[Open the video\]/g) || []).length).toBe(videos + Number(s === earth && s.includes('[Open the video](https://www.youtube.com/watch?v=p4652vo5Oic)')) + Number(s === earth && s.includes('[Open the video](https://www.youtube.com/watch?v=bs9e6ovISbs)')) + Number(added) + Number(s === weather && s.includes('[Open the video](https://www.youtube.com/watch?v=xKubdY2mHXc)')))
      if (s === earth) {
        const urls = [...s.matchAll(/\[Open the video\]\(([^)]+)\)/g)].map(m => m[1])
        expect(urls).toEqual(['https://www.youtube.com/watch?v=Bsp5JYNMAQE', ...(added ? ['https://www.youtube.com/watch?v=o8K0RTYM6Mc'] : []), ...(s.includes('[Open the video](https://www.youtube.com/watch?v=p4652vo5Oic)') ? ['https://www.youtube.com/watch?v=p4652vo5Oic'] : []), ...(s.includes('[Open the video](https://www.youtube.com/watch?v=bs9e6ovISbs)') ? ['https://www.youtube.com/watch?v=bs9e6ovISbs'] : [])])
      }
      expect((s.match(/### Try it yourself/g) || []).length).toBe(4)
      expect((s.match(/\*\*Check:\*\*/g) || []).length).toBe(4)
      expect(s).toContain('## Your five-minute revision')
    }
  })
})
