import { describe,it,expect } from 'vitest'
import { ratioScaling,ratioFactor } from './ratioModel'
import { readFileSync } from 'node:fs'
describe('proportional rectangles',()=>{
 it('recognises equivalent ratios including fractional scaling',()=>{for(const [w,h] of [[30,20],[60,40],[90,60]])expect(ratioScaling(w,h)).toMatchObject({ratio:'3:2',proportional:true});expect(ratioFactor(30,60)).toBe('1/2');expect(ratioFactor(90,60)).toBe('3/2')})
 it('rejects equal-addition reasoning',()=>{expect(ratioScaling(80,60)).toMatchObject({ratio:'4:3',proportional:false,widthChange:20,heightChange:20});expect(ratioScaling(40,20).proportional).toBe(false)})
 it('uses exact cross products for all supported pairs',()=>{for(let w=20;w<=100;w+=5)for(let h=20;h<=100;h+=5){const m=ratioScaling(w,h);expect(m.proportional).toBe(w*2===h*3);const [a,b]=m.ratio.split(':').map(Number);expect(a*h).toBe(b*w);expect(m.widthFactor).toBeCloseTo(w/60);expect(m.heightFactor).toBeCloseTo(h/40)}})
 it('rejects invalid or unsupported dimensions',()=>{for(const n of [0,-20,19,21,105,NaN,Infinity,20.5]){expect(()=>ratioScaling(n,40)).toThrow(RangeError);expect(()=>ratioScaling(60,n)).toThrow(RangeError)}})
 it('covers minimum and maximum square bounds',()=>{for(const n of [20,100])expect(ratioScaling(n,n)).toMatchObject({ratio:'1:1',proportional:false})})
 it('independently checks the transfer question',()=>{expect(90/60).toBe(45/30);expect(60/45).not.toBe(45/30)})
 it('mounts only on the matching lesson and namespaces patterns',()=>{const s=readFileSync(new URL('../../pages/BlogPostPage.tsx',import.meta.url),'utf8');expect(s.match(/<RatioScaling \/>/g)).toHaveLength(1);expect(s).toContain("post.slug === 'class-8-maths-proportional-reasoning-1-study-resources'");const c=readFileSync(new URL('./RatioScaling.tsx',import.meta.url),'utf8');expect(c).toContain('useId()');expect(c).toContain('${id}-grid-${i}')})
})
