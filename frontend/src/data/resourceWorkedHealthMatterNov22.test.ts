import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
const health = readFileSync(new URL('./resourceGuides/10-science-3.md', import.meta.url), 'utf8')
const matter = readFileSync(new URL('./resourceGuides/14-science-8.md', import.meta.url), 'utf8')
const section = (text: string, n: number) => text.split(`## ${n}.`)[1].split('\n## ')[0]

describe('six health and matter reasoning lessons', () => {
  it('has teaching, a distinct transfer question and a reasoned check in all six topics', () => {
    for (const [text, topics] of [[health, [1, 2, 3, 4]], [matter, [3, 4]]] as const) {
      for (const n of topics) {
        const block = section(text, n)
        expect(block).toMatch(/### (Understand it|Think it through)/)
        expect(block).toContain('### Try it yourself')
        expect(block).toContain('**Check:**')
        expect(block.split('**Check:**')[1].trim().split(/\s+/).length).toBeGreaterThan(30)
      }
    }
  })
  it('keeps physical, mental and social wellbeing separate from appearance or supplement choices', () => {
    const block = section(health, 1)
    for (const text of ['physical, mental and social', 'Do not judge health by appearance', 'choose supplements on your own', 'too little evidence']) expect(block).toContain(text)
    expect(block).toContain('The lunches can support health without solving every problem')
  })
  it('distinguishes a reported symptom, a measured sign and an unproved cause claim', () => {
    const block = section(health, 2)
    for (const text of ['38.5°C', 'Feeling dizzy is a reported symptom', 'measured high temperature is a sign', 'cause claim', 'not for diagnosing yourself or another person']) expect(block).toContain(text)
    expect(block).toContain('Neither a symptom nor a sign alone identifies the disease')
  })
  it('matches prevention to the stipulated route without treating all illness as contagious or blameworthy', () => {
    const block = section(health, 3)
    for (const text of ['Safe drinking water addresses the stated route for A', 'mosquito breeding or exposure address B', 'does not tell us one universal prevention method', 'any person is to blame']) expect(block).toContain(text)
    expect(block).toContain('not automatically caught by sitting beside someone')
  })
  it('preserves vaccine limits, bacterial resistance and clinician-only antibiotic decisions', () => {
    const block = section(health, 4)
    for (const text of ['protection takes time and is not perfect', 'not the viruses that cause colds', 'bacteria become harder to treat', 'Do not self-prescribe antibiotics or share medicines', 'qualified clinician decides']) expect(block).toContain(text)
    expect(block).toContain('does not guarantee protection from every germ')
    expect(block).toContain('not a person\'s body becoming resistant')
  })
  it('separates pure elements and compounds from uniform mixtures', () => {
    const block = section(matter, 3)
    for (const text of ['Silver is an element', 'Pure carbon dioxide is a compound', 'Air is a mixture of gases', 'uniform appearance alone cannot distinguish']) expect(block).toContain(text)
    expect(block).toContain('having two elements does not make it a mixture')
    expect(block).toContain('not instructions to heat or electrolyse substances at home')
  })
  it('uses independent alloy, elemental, mineral and multi-mineral-rock transfer cases', () => {
    const block = section(matter, 4)
    for (const text of ['brass, a mixture', 'Pure aluminium is an element', 'Quartz is a compound mineral', 'multi-mineral rock is a mixture', 'not every mineral must be a compound']) expect(block).toContain(text)
    expect(block).toContain('at least one metal')
    expect(block).toContain('Do not classify a material from shininess alone')
  })
  it('retains existing embeds and does not add reading redirects or unsafe experiments', () => {
    expect((health.match(/\[Open the video\]/g) || []).length).toBe(2)
    expect((matter.match(/\[Open the video\]/g) || []).length).toBe(1)
    for (const text of [health, matter]) {
      expect(text).not.toMatch(/https?:\/\/(?:www\.)?(?:who\.int|cdc\.gov|openstax\.org)/)
      expect(text).not.toContain('supporting evidence')
      expect(text).toContain('## Your five-minute revision')
    }
  })
})
