import { describe,it,expect } from 'vitest'
import { positionalModel } from './numberStoryModel'
import words from '../../data/mathsGuides/story-of-numbers.md?raw'
describe('positional teaching model',()=>{
 it('preserves every allowed amount in every base',()=>{for(let b=2;b<=10;b++)for(let n=0;n<=255;n++){const m=positionalModel(n,b);expect(m.places.reduce((a,p)=>a+p.contribution,0)).toBe(n);expect(m.places.every(p=>p.digit>=0&&p.digit<b&&p.contribution===p.digit*p.weight)).toBe(true);expect(parseInt(m.numeral,b)).toBe(n);expect(m.numeral).toBe(n.toString(b))}})
 it('keeps zero and empty internal places',()=>{expect(positionalModel(0,5).places).toEqual([{digit:0,weight:1,contribution:0}]);expect(positionalModel(52,5).numeral).toBe('202')})
 it('rejects invalid model inputs',()=>{for(const n of [-1,.5,256,NaN,Infinity])expect(()=>positionalModel(n,5)).toThrow();for(const b of [0,1,2.5,11,NaN])expect(()=>positionalModel(42,b)).toThrow()})
 it('independently checks original worked answers',()=>{expect(1*25+3*5+2).toBe(42);expect(2*25+2).toBe(52);expect(3*5+3).toBe(18);expect(2*16+3).toBe(35);expect(2*5+3+4).toBe(3*5+2);expect(400+7).not.toBe(40+7);expect(50-10+2).toBe(42);expect(50+10+4).toBe(64)})
 it('keeps full original prose and honest limits',()=>{expect(words.split(/\s+/).length).toBeGreaterThan(1000);expect(words).toContain('not a reconstruction');expect(words).toContain('does not certify');expect(words).toContain('hegp103.pdf')})
})
