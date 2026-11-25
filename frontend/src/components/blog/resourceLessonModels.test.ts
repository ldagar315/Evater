import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import { resourceGuides } from '../../data/resourceGuides'
const source = readFileSync(new URL('../../pages/BlogPostPage.tsx',import.meta.url),'utf8')
const mappings = [
  ['ncert-class-8-maths-power-play-study-guide','class-8-maths-power-play-study-resources','PowerGrowthVisual'],
  ['ncert-class-8-maths-area-study-guide','class-8-maths-area-study-resources','AreaHeightVisual'],
  ['ncert-class-8-maths-square-and-cube-study-guide','class-8-maths-a-square-and-a-cube-study-resources','CubeVolumeSimulation'],
  ['ncert-class-8-maths-baudhayana-pythagoras-theorem','class-8-maths-the-baudhayana-pythagoras-theorem-study-resources','PythagorasSimulation'],
  ['ncert-class-8-science-exploring-forces-study-guide','class-8-science-exploring-forces-study-resources','ForceSimulation'],
]
describe('reuse existing models on matching resource lessons',()=>{
  for(const [original,resource,component] of mappings) it(`keeps ${original} and adds ${resource}`,()=>{
    expect(source).toContain(`post.slug === '${original}' && <${component} />`)
    expect(source).toContain(`post.slug === '${resource}' && <${component} />`)
    expect(resourceGuides.filter(g=>g.slug===resource)).toHaveLength(1)
    const guide=resourceGuides.find(g=>g.slug===resource)!
    expect(guide.content).toContain('Try it yourself')
    expect(guide.content).toContain('Check:')
    expect(guide.content).toContain('Your five-minute revision')
    expect(guide.content).toContain('[Open the video]')
  })
  it('does not widen model mounting to every lesson',()=>{
    expect(mappings).toHaveLength(5)
    expect(new Set(mappings.map(m=>m[1])).size).toBe(5)
    expect(source).not.toContain("post.id.startsWith('resource-') && <ForceSimulation")
  })
})

it('keeps SVG pattern ids distinct from Pythagoras input ids',()=>{
 const model=readFileSync(new URL('./PythagorasSimulation.tsx',import.meta.url),'utf8')
 expect(model).toContain('id={`${id}-pattern-b`}')
 expect(model).toContain('id={`${id}-b`} data-side=')
 expect(model).toContain('url(#${id}-pattern-')
})
