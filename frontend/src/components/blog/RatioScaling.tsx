import { useId, useState } from 'react'
import { ratioFactor, ratioScaling } from './ratioModel'
const button = 'min-h-11 rounded-xl border border-teal-700 bg-white px-4 py-2 text-sm font-bold text-teal-900 hover:bg-teal-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-700'
export function RatioScaling() {
  const id = useId()
  const [width,setWidth] = useState(90), [height,setHeight] = useState(60), [answer,setAnswer] = useState(false)
  const m = ratioScaling(width,height)
  const preset = (w:number,h:number) => { setWidth(w);setHeight(h) }
  return <section data-ratio-scaling="true" aria-labelledby={`${id}-title`} className="my-10 rounded-2xl border border-teal-200 bg-teal-50/50 p-4 sm:p-6">
    <p className="text-xs font-extrabold uppercase tracking-widest text-teal-700">Explore proportional scaling</p>
    <h2 id={`${id}-title`} className="mt-2 text-2xl font-extrabold text-dark">Same shape, or just the same change?</h2>
    <p className="mt-3 text-sm leading-6">Start with a rectangle 60 units wide and 40 units high. Predict which changes keep its proportions. Each small grid square is 10 by 10 units. The two panels use the same scale, so size and shape can both be compared.</p>
    <div className="mt-5 grid gap-5 md:grid-cols-2">
      {[{name:'Original',w:60,h:40,fill:'#475569'},{name:'Your rectangle',w:width,h:height,fill:'#0f766e'}].map((r,i)=><figure key={r.name} className="rounded-xl border border-teal-100 bg-white p-3">
        <figcaption className="mb-3 text-sm font-bold">{r.name}: {r.w} × {r.h} units</figcaption>
        <svg data-ratio-panel={i} viewBox="0 0 120 120" role="img" aria-label={`${r.name}, width ${r.w}, height ${r.h} units, on a zero to 100 unit grid.`} className="mx-auto w-full max-w-80">
          <defs><pattern id={`${id}-grid-${i}`} width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 H 0 V 10" fill="none" stroke="#d4d4d4" strokeWidth="0.5" /></pattern></defs>
          <rect x="10" y="10" width="100" height="100" fill={`url(#${id}-grid-${i})`} stroke="#737373" strokeWidth="0.5" />
          <rect data-ratio-rectangle={i} x="10" y={110-r.h} width={r.w} height={r.h} fill={r.fill} fillOpacity="0.22" stroke={r.fill} strokeWidth="1" />
          <path d={`M 10 ${110-r.h} L ${10+r.w} 110 M 10 110 L ${10+r.w} ${110-r.h}`} fill="none" stroke={r.fill} strokeWidth="0.8" />
          <text x="10" y="118" fontSize="4">0</text><text x="100" y="118" fontSize="4">100 units</text>
        </svg>
        <p className="mt-2 text-sm">Width : height = {r.w}:{r.h}{i===0?' = 3:2':` = ${m.ratio}`}</p>
      </figure>)}
    </div>
    <div className="my-5 grid gap-5 sm:grid-cols-2">{[{name:'Width',value:width,set:setWidth},{name:'Height',value:height,set:setHeight}].map(c=><label key={c.name} htmlFor={`${id}-${c.name}`} className="block text-sm font-bold">{c.name}: {c.value} units<input id={`${id}-${c.name}`} type="range" min="20" max="100" step="5" value={c.value} onChange={e=>c.set(Number(e.target.value))} className="mt-3 min-h-11 w-full accent-teal-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-700" /></label>)}</div>
    <div className="flex flex-wrap gap-3"><button type="button" className={button} onClick={()=>preset(30,20)}>Halve both dimensions</button><button type="button" className={button} onClick={()=>preset(80,60)}>Add 20 to both</button><button type="button" className={button} onClick={()=>preset(90,60)}>Reset: multiply both by 3/2</button></div>
    <div data-ratio-result="true" aria-live="polite" aria-atomic="true" className="mt-5 space-y-2 rounded-xl bg-white p-4 text-sm leading-7">
      <p><strong>{m.proportional?'Same proportions':'Different proportions'}: {m.ratio} {m.proportional?'=':'≠'} 3:2.</strong></p>
      <p>Width factor: {width}/60 = {ratioFactor(width,60)}. Height factor: {height}/40 = {ratioFactor(height,40)}.</p>
      <p>{m.proportional?'Both dimensions use the same multiplier, so the rectangle keeps its shape.':'The multipliers differ. An equal change in units is not the same as an equal scale factor.'}</p>
      <p>Cross-check: {width} × 40 = {width*40}; {height} × 60 = {height*60}. {m.proportional?'The cross-products match.':'The cross-products do not match.'}</p>
      <p>Width changes by {m.widthChange>0?'+':''}{m.widthChange} units; height by {m.heightChange>0?'+':''}{m.heightChange} units. Ratios compare ordered dimensions, not their difference.</p>
    </div>
    <div className="mt-5 border-t border-teal-200 pt-5"><p className="text-sm leading-6">Before checking: a different rectangle is 45 units wide and 30 high. Which preserves its shape, doubling both dimensions or adding 15 units to both? Explain with factors, not appearance alone.</p><button type="button" className={`${button} mt-3`} aria-expanded={answer} aria-controls={`${id}-answer`} onClick={()=>setAnswer(!answer)}>{answer?'Hide explanation':'Check explanation'}</button>{answer&&<p id={`${id}-answer`} className="mt-3 text-sm leading-7">Doubling gives 90 × 60: both factors are 2 and 90:60 = 3:2. Adding 15 gives 60 × 45: the factors are 4/3 and 3/2, and 60:45 = 4:3. The equal addition changes the proportions. This is a 2D rectangle model, not a test of every feature of a real image.</p>}</div>
  </section>
}
