import { useState } from 'react'
import { positionalModel } from './numberStoryModel'
export function NumberStoryVisual() {
 const [amount,setAmount]=useState(42),[base,setBase]=useState(5)
 const m=positionalModel(amount,base)
 return <section aria-labelledby="number-story-model-title" className="my-8 rounded-2xl border border-primary-200 bg-primary-50 p-5 sm:p-7">
  <h2 id="number-story-model-title" className="scroll-mt-24 text-xl font-bold">Same amount, different base</h2>
  <p className="mt-2 text-sm leading-6">Keep the amount fixed and change the base. Predict the numeral first, then add the place contributions to check it.</p>
  <div className="my-5 grid gap-5 sm:grid-cols-2">
   <label className="block text-sm font-semibold">Decimal amount: {amount}<input aria-label="Decimal amount" type="range" min="0" max="255" value={amount} onChange={e=>setAmount(Number(e.target.value))} className="mt-3 block h-8 w-full accent-teal-700"/></label>
   <label className="block text-sm font-semibold">Base: {base}<input aria-label="Number base" type="range" min="2" max="10" value={base} onChange={e=>setBase(Number(e.target.value))} className="mt-3 block h-8 w-full accent-teal-700"/></label>
  </div>
  <p aria-live="polite" className="rounded-lg bg-white p-4 text-base"><strong>{m.numeral} in base {base}</strong> represents <strong>{amount} in decimal</strong>.</p>
  <div role="list" aria-label="Place value contributions, largest first" className="my-5 grid grid-cols-2 gap-3 sm:grid-cols-4">{m.places.map(p=><div role="listitem" key={p.weight} className="rounded-xl border border-primary-200 bg-white p-3 text-center"><span className="block text-xs text-neutral-600">Place value {p.weight}</span><strong className="my-2 block text-2xl">{p.digit}</strong><span className="block text-sm">{p.digit} × {p.weight} = {p.contribution}</span></div>)}</div>
  <p className="text-sm leading-6">Check: {m.places.map(p=>p.contribution).join(' + ')} = {amount}. Allowed digits: 0 to {base-1}. Every {base} units of one place exchange for one of the next place.</p>
  <p className="mt-3 text-xs leading-5 text-neutral-600">A modern positional-notation teaching model for whole numbers 0-255. Not a simulation of every historical numeral system. In particular, the chapter's Mayan weights are not uniform powers of 20.</p>
 </section>
}
