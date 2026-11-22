import {describe,expect,it} from 'vitest'
import {resourceGuides} from './resourceGuides'
const guide=(id:number)=>resourceGuides.find(p=>p.id===`resource-science-${id}`)!.content
const topic=(id:number,n:number)=>guide(id).split(/^## /m).find(s=>s.startsWith(n+'.'))!
describe('six worked Science topics, November20',()=>{
 it('keeps six independent activities and existing video inventory',()=>{
  for(const [id,n]of [[6,1],[8,1],[8,2],[10,1],[10,2],[11,4]]){const s=topic(id,n);expect(s).toContain('### Try it yourself');expect(s).toContain('**Check:**')}
  for(const [id,videos]of [[6,3],[10,4]])expect((guide(id).match(/\[Open the video\]/g)||[]).length).toBe(videos)
  const skyKnown=['https://www.youtube.com/watch?v=l64YwNl1wr0','https://www.youtube.com/watch?v=Jip3BbZBpsM']
  const skySelected=skyKnown.filter((url,i)=>i===1||guide(11).includes(`[Open the video](${url})`))
  expect([...guide(11).matchAll(/\[Open the video\]\(([^)]+)\)/g)].map(m=>m[1])).toEqual(skySelected)
  for(const url of skySelected)expect(topic(11,skyKnown.indexOf(url)+1)).toContain(`[Open the video](${url})`)
  const known=["https://www.youtube.com/watch?v=vi_SJBnxmHo", "https://www.youtube.com/watch?v=nUzOXy9V-K0", "https://www.youtube.com/watch?v=Vt7lN4QPU0k", "https://www.youtube.com/watch?v=KgUmNQD6m5Q"]
  const selected=known.filter((url,i)=>i===2||guide(8).includes(`[Open the video](${url})`))
  expect([...guide(8).matchAll(/\[Open the video\]\(([^)]+)\)/g)].map(m=>m[1])).toEqual(selected)
 })
 it('compares ratios rather than force or area alone',()=>{
  expect(60/.30).toBe(200);expect(60/.10).toBe(600);expect(120/.60).toBe(200)
  const pressureA=90/.30,pressureB=120/.60
  expect(pressureA).toBe(300);expect(pressureB).toBe(200);expect(pressureA).toBeGreaterThan(pressureB)
  const s=topic(6,1);expect(s).toContain('perpendicular force');expect(s).toContain('Both force and area matter')
 })
 it('tracks which components are separated and preserves practical safety',()=>{
  const mixture=['sand','salt','water'];const residue=mixture.filter(x=>x==='sand');const filtrate=mixture.filter(x=>x!=='sand')
  expect(residue).toEqual(['sand']);expect(filtrate).toEqual(['salt','water']);expect(filtrate.length).toBeGreaterThan(1)
  const s=topic(8,1);expect(s).toContain('The clear filtrate is still a mixture');expect(s).toContain('not an instruction to heat mixtures at home');expect(s).toContain('teacher\'s approved setup and supervision')
  expect(s).toContain('evaporating the filtrate is not needed just to obtain sand')
 })
 it('distinguishes atom count from kind of atom, and separation from decomposition',()=>{
  const kindCount=(atoms:string[])=>new Set(atoms).size
  expect(kindCount(['O','O'])).toBe(1);expect(kindCount(['N','N'])).toBe(1);expect(kindCount(['H','H','O'])).toBe(2);expect(kindCount(['C','O','O'])).toBe(2)
  const s=topic(8,2);expect(s).toContain('only one kind of atom');expect(s).toContain('Having two atoms does not by itself make something a compound');expect(s).toContain('fixed ratio');expect(s).toContain('not decomposition of sugar into its elements')
 })
 it('uses the normal for reflection angles, including normal incidence',()=>{
  expect(90-30).toBe(60);expect(90-25).toBe(65)
  for(let surface=0;surface<=90;surface++){const incidence=90-surface,reflection=incidence;expect(reflection+surface).toBe(90)}
  const s=topic(10,1);expect(s).toContain('Both angles are 65°');expect(s).toContain('both angles are 0°');expect(s).toContain('returns along the same line')
 })
 it('uses combined observations and avoids a universal concave-image rule',()=>{
  const s=topic(10,2);expect(s).toContain('bringing it sufficiently close');expect(s).toContain('Moving it farther away can give an inverted image');expect(s).toContain('ordinary object in front');expect(s).toContain('A is plane, B is convex, and C is concave');expect(s).toContain('orientation alone cannot identify the mirror');expect(s).toContain('not a task to focus sunlight')
  expect(guide(10)).toContain('Never look at the Sun through a lens or magnifying glass')
 })
 it('checks Gregorian decisions independently against standard calendar dates',()=>{
  const leap=(y:number)=>y%4===0&&(y%100!==0||y%400===0)
  for(let y=1600;y<=2500;y++){const feb=new Date(Date.UTC(y,2,0)).getUTCDate();expect(leap(y)).toBe(feb===29)}
  expect([2028,1900,2000,2100,2400,2027,2032].map(leap)).toEqual([true,false,true,false,true,false,true])
  const s=topic(11,4);expect(s).toContain('Other calendars need their own rules');expect(s).toContain('2100 is not a leap year');expect(s).toContain('2400 is a leap year')
 })
})
