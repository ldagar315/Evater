import { useState } from 'react'
import { areaShape, powerGrowth } from './mathsChapterModel'

export function PowerGrowthVisual() {
  const [base, setBase] = useState(2), [steps, setSteps] = useState(4)
  const model = powerGrowth(base, steps)
  return <section aria-labelledby="power-growth-title" className="my-8 rounded-2xl border border-primary-200 bg-primary-50 p-5 sm:p-7">
    <h2 id="power-growth-title" className="scroll-mt-24 text-xl font-bold">Try it: add or multiply?</h2>
    <p className="mt-2 text-sm leading-6">Both counters start at 1. One adds the chosen number each step; the other multiplies by it. Predict before changing a control.</p>
    <div className="my-5 grid gap-5 sm:grid-cols-2">
      <label className="block text-sm font-semibold">Number: {base}<input aria-label="Growth number" type="range" min="2" max="5" value={base} onChange={e => setBase(Number(e.target.value))} className="mt-3 block h-8 w-full accent-teal-700" /></label>
      <label className="block text-sm font-semibold">Steps: {steps}<input aria-label="Growth steps" type="range" min="0" max="6" value={steps} onChange={e => setSteps(Number(e.target.value))} className="mt-3 block h-8 w-full accent-teal-700" /></label>
    </div>
    <div aria-live="polite" className="grid gap-3 sm:grid-cols-2">
      <p className="rounded-lg bg-white p-4">Repeated addition<br/><strong>1 + {base} × {steps} = {model.added.toLocaleString('en-IN')}</strong></p>
      <p className="rounded-lg bg-white p-4">Repeated multiplication<br/><strong>{base}<sup>{steps}</sup> = {model.multiplied.toLocaleString('en-IN')}</strong></p>
    </div>
    <ol className="mt-5 space-y-2 text-sm">{model.values.map(v => <li key={v.step} className="flex flex-wrap justify-between gap-2 border-b border-primary-100 pb-2"><span>Step {v.step}</span><span>Add: {v.added.toLocaleString('en-IN')} · Multiply: {v.multiplied.toLocaleString('en-IN')}</span></li>)}</ol>
    <p className="mt-4 text-xs leading-5 text-neutral-600">A mathematical counter model, not a prediction of unlimited physical paper folding. At step 0, neither operation has been applied.</p>
  </section>
}

export function AreaHeightVisual() {
  const [base, setBase] = useState(8), [height, setHeight] = useState(3), [shift, setShift] = useState(2)
  const m = areaShape(base, height, shift), scale = 19, x = 100, y = 190, top = y - m.height * scale
  const points = `${x},${y} ${x+m.base*scale},${y} ${x+(m.base+m.shift)*scale},${top} ${x+m.shift*scale},${top}`
  const hx = x+m.shift*scale
  return <section aria-labelledby="area-height-title" className="my-8 rounded-2xl border border-primary-200 bg-primary-50 p-5 sm:p-7">
    <h2 id="area-height-title" className="scroll-mt-24 text-xl font-bold">Slide the top, keep the area</h2>
    <p className="mt-2 text-sm leading-6">Change the sideways shift while holding base and height fixed. The sloping side changes; the perpendicular height does not.</p>
    <div className="my-5 grid gap-4 sm:grid-cols-3">{([{label:'Base',value:base,min:4,max:12,set:setBase},{label:'Height',value:height,min:2,max:6,set:setHeight},{label:'Sideways shift',value:shift,min:-3,max:3,set:setShift}]).map(c => <label key={c.label} className="block text-sm font-semibold">{c.label}: {c.value}<input aria-label={c.label} type="range" min={c.min} max={c.max} value={c.value} onChange={e => c.set(Number(e.target.value))} className="mt-3 block h-8 w-full accent-teal-700" /></label>)}</div>
    <svg viewBox="0 0 420 250" role="img" aria-label={`Parallelogram with base ${m.base}, perpendicular height ${m.height}, sideways shift ${m.shift}. Area ${m.area} square units.`} className="w-full rounded-xl bg-white">
      <line x1="30" x2="390" y1={y} y2={y} stroke="#b5c6c2" strokeDasharray="4 4" />
      <polygon points={points} fill="#d1eee6" stroke="#176b59" strokeWidth="2" />
      <line x1={hx} x2={hx} y1={top} y2={y} stroke="#b45309" strokeWidth="2" strokeDasharray="5 4" />
      <path d={`M ${hx} ${y-9} h 9 v 9`} fill="none" stroke="#b45309" />
      <text x={hx+10} y={(top+y)/2} fontSize="14" fill="#7c4008">h = {m.height}</text>
      <text x={x+m.base*scale/2} y={y+28} textAnchor="middle" fontSize="14">base = {m.base}</text>
      <text x="20" y="25" fontSize="13">Not to physical scale. Lengths in units.</text>
    </svg>
    <p aria-live="polite" className="mt-4 text-base"><strong>Area = {m.base} × {m.height} = {m.area} square units.</strong> A triangle with this base and height has area {m.triangleArea} square units.</p>
    <p className="mt-3 text-xs leading-5 text-neutral-600">Sloping side ≈ {m.slopingSide.toFixed(2)} units. The dashed line is perpendicular to the base line, even when it falls outside the shape. A 2D area model, not a 3D simulation.</p>
  </section>
}
