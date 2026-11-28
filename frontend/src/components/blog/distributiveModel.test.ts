import {describe,it,expect} from 'vitest'
import {distributiveArea} from './distributiveModel'
import {readFileSync} from 'node:fs'
describe('positive-length distributive area',()=>{
 it('accounts for all four regions in the default',()=>{expect(distributiveArea(5,3,2)).toMatchObject({width:8,height:7,parts:[25,15,10,6],total:56,middle:5,missing:25,square:false})})
 it('conserves area for all324supported combinations',()=>{for(let x=2;x<=10;x++)for(let a=1;a<=6;a++)for(let b=1;b<=6;b++){const m=distributiveArea(x,a,b);expect(m.parts.reduce((s,n)=>s+n,0)).toBe(m.total);expect(m.total).toBe(x*x+(a+b)*x+a*b);expect(m.total-x*x-a*b).toBe(m.missing);expect(m.width*m.height).toBe(m.total)}})
 it('shows two equal cross-products for a square',()=>{const m=distributiveArea(4,3,3);expect(m.square).toBe(true);expect(m.parts[1]).toBe(m.parts[2]);expect(m.missing).toBe(24);expect(m.total).toBe(49)})
 it('covers smallest and largest dimensions',()=>{expect(distributiveArea(2,1,1).total).toBe(9);expect(distributiveArea(10,6,6).total).toBe(256)})
 it('rejects invalid values including zero and negative lengths',()=>{for(const n of [-1,0,1,11,NaN,Infinity,2.5])expect(()=>distributiveArea(n,2,3)).toThrow(RangeError);for(const n of [-1,0,7,NaN,Infinity,1.5]){expect(()=>distributiveArea(5,n,2)).toThrow(RangeError);expect(()=>distributiveArea(5,2,n)).toThrow(RangeError)}})
 it('independently checks the transfer example',()=>{expect((3+4)*(3+1)).toBe(28);expect(3*3+4).toBe(13);expect(4*3+3).toBe(15)})
 it('mounts once and explains the domain boundary',()=>{const s=readFileSync(new URL('../../pages/BlogPostPage.tsx',import.meta.url),'utf8');expect(s.match(/<DistributiveArea \/>/g)).toHaveLength(1);expect(s).toContain("post.slug === 'class-8-maths-we-distribute-yet-things-multiply-study-resources'");const c=readFileSync(new URL('./DistributiveArea.tsx',import.meta.url),'utf8');expect(c).toContain('does not represent negative lengths');expect(c).toContain('useId()')})
})
