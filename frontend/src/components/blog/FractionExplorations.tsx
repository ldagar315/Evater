import { useId, useState } from 'react'
import { fractionNumber as fmt, fractionShare, PART_COUNTS, successiveChanges } from './fractionModel'

const control = 'mt-3 min-h-11 w-full accent-teal-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-teal-700'
const button = 'min-h-11 rounded-xl border border-teal-700 px-4 py-2 text-sm font-bold text-teal-900 hover:bg-teal-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-700'

export function FractionExplorations() {
  const id = useId()
  const [parts, setParts] = useState(20)
  const [shaded, setShaded] = useState(7)
  const [original, setOriginal] = useState(500)
  const [firstRate, setFirstRate] = useState(20)
  const [secondRate, setSecondRate] = useState(-20)
  const [answer, setAnswer] = useState(false)
  const f = fractionShare(shaded, parts), m = successiveChanges(original, firstRate, secondRate)
  const columns = parts === 40 ? 8 : parts === 25 || parts === 20 || parts === 10 || parts === 5 ? 5 : parts === 8 ? 4 : parts
  const rates = (n: number) => `${n > 0 ? '+' : ''}${n}%`
  return <div data-fraction-explorations="true" className="my-10 space-y-8">
    <section aria-labelledby={`${id}-share`} className="rounded-2xl border border-teal-200 bg-teal-50/50 p-4 sm:p-6">
      <p className="text-xs font-extrabold uppercase tracking-widest text-teal-700">Explore equal parts</p>
      <h2 id={`${id}-share`} className="mt-2 text-2xl font-extrabold text-dark">One share, three ways to write it</h2>
      <p className="mt-3 text-sm leading-6">The entire grid is one whole. Every cell has equal area. Predict the percentage, then change the shaded parts. This model shows shares from 0% to 100%; percentages can be greater than 100% in other contexts.</p>
      <div className="mt-5 grid gap-6 md:grid-cols-2 md:items-center">
        <div className="rounded-xl border border-teal-100 bg-white p-4">
          <div role="img" aria-label={`${shaded} of ${parts} equal parts shaded, ${fmt(f.percent)} percent; ${f.remaining} unshaded.`} data-fraction-grid="true" className="mx-auto grid max-w-80 gap-1" style={{gridTemplateColumns:`repeat(${columns}, minmax(0, 1fr))`}}>
            {Array.from({length:parts},(_,i)=><span key={i} data-shaded={i<shaded ? 'true' : 'false'} className={`aspect-square border-2 border-teal-800 ${i<shaded ? 'bg-teal-700' : 'bg-white'}`} aria-hidden="true" />)}
          </div>
          <p className="mt-3 text-center text-sm">Filled teal: shaded. White: unshaded.</p>
        </div>
        <div className="space-y-5">
          <label className="block text-sm font-bold" htmlFor={`${id}-parts`}>Equal parts in the whole
            <select id={`${id}-parts`} value={parts} onChange={e=>{const next=Number(e.target.value);setShaded(Math.round(shaded/parts*next));setParts(next)}} className="mt-3 min-h-12 w-full rounded-xl border border-teal-500 bg-white px-3">{PART_COUNTS.map(n=><option key={n} value={n}>{n} equal parts</option>)}</select>
          </label>
          <p className="text-xs leading-5 text-neutral-600">Changing the partition keeps the nearest possible share. Use the shaded-parts slider to choose an exact count.</p>
          <label className="block text-sm font-bold" htmlFor={`${id}-shaded`}>Shaded parts: {shaded} of {parts}<input id={`${id}-shaded`} type="range" min="0" max={parts} step="1" value={shaded} onChange={e=>setShaded(Number(e.target.value))} className={control} /></label>
          <div data-fraction-result="true" aria-live="polite" aria-atomic="true" className="rounded-xl bg-white p-4 text-sm leading-7">
            <p><strong>Shaded: {shaded}/{parts} = {fmt(f.decimal)} = {fmt(f.percent)}%</strong></p>
            <p>Unshaded: {f.remaining}/{parts} = {fmt(f.complementDecimal)} = {fmt(f.complementPercent)}%</p>
            <p>The shares add to one whole, or 100%.</p>
          </div>
          <button type="button" className={button} onClick={()=>{setParts(20);setShaded(7)}}>Reset to 7 of 20</button>
        </div>
      </div>
    </section>
    <section aria-labelledby={`${id}-changes`} className="rounded-2xl border border-teal-200 bg-teal-50/50 p-4 sm:p-6">
      <p className="text-xs font-extrabold uppercase tracking-widest text-teal-700">Explore changing bases</p>
      <h2 id={`${id}-changes`} className="mt-2 text-2xl font-extrabold text-dark">Does a rise cancel an equal fall?</h2>
      <p className="mt-3 text-sm leading-6">Use a hypothetical quantity measured in units. The second percentage acts on the result of the first step, not on the original. Negative rates mean decreases. These are 2D mathematical models, not prices, forecasts or financial advice.</p>
      <div className="my-5 grid gap-5 sm:grid-cols-3">
        <label htmlFor={`${id}-original`} className="text-sm font-bold">Original: {original} units<input id={`${id}-original`} type="range" min="100" max="1000" step="50" value={original} onChange={e=>setOriginal(Number(e.target.value))} className={control} /></label>
        <label htmlFor={`${id}-first`} className="text-sm font-bold">First change: {rates(firstRate)}<input id={`${id}-first`} type="range" min="-100" max="100" step="5" value={firstRate} onChange={e=>setFirstRate(Number(e.target.value))} className={control} /></label>
        <label htmlFor={`${id}-second`} className="text-sm font-bold">Second change: {rates(secondRate)}<input id={`${id}-second`} type="range" min="-100" max="100" step="5" value={secondRate} onChange={e=>setSecondRate(Number(e.target.value))} className={control} /></label>
      </div>
      <figure className="space-y-4 rounded-xl bg-white p-4" aria-label="Three bars on one common scale">
        {[{label:'Original',value:original},{label:'After first change',value:m.first},{label:'After second change',value:m.final}].map((row,i)=><div key={row.label}>
          <p className="mb-2 text-sm font-bold">{row.label}: {fmt(row.value)} units</p>
          <div className="h-6 rounded border border-neutral-300 bg-neutral-50" aria-hidden="true"><div data-percent-bar={i} data-value={row.value} className={`h-full ${i===0?'bg-slate-600':i===1?'bg-teal-700':'bg-indigo-700'}`} style={{width:`${row.value/m.scaleMax*100}%`}} /></div>
        </div>)}
        <figcaption className="text-xs leading-5 text-neutral-600">All three bars share a zero and scale. The longest is {fmt(m.scaleMax)} units, not necessarily 100 units. The scale adjusts when you change a control; compare bars within one view.</figcaption>
      </figure>
      <div data-percent-result="true" aria-live="polite" aria-atomic="true" className="mt-5 space-y-2 rounded-xl bg-white p-4 text-sm leading-7">
        <p>First: {original} × (1 {firstRate<0?'−':'+'} {Math.abs(firstRate)}/100) = <strong>{fmt(m.first)} units</strong>. Change: {fmt(m.firstChange)} units.</p>
        <p>Second: {fmt(m.first)} × (1 {secondRate<0?'−':'+'} {Math.abs(secondRate)}/100) = <strong>{fmt(m.final)} units</strong>. Change: {fmt(m.secondChange)} units.</p>
        <p><strong>Net: {fmt(m.netChange)} units, or {fmt(m.netPercent)}% of the original {original}.</strong></p>
        {m.first===0 && <p>The first step reached zero. Any percentage of zero is zero, so the second step stays at zero.</p>}
      </div>
      <button type="button" className={`${button} mt-4`} onClick={()=>{setOriginal(500);setFirstRate(20);setSecondRate(-20)}}>Reset: +20%, then −20%</button>
      <div className="mt-5 border-t border-teal-200 pt-5">
        <p className="text-sm leading-6">Before checking: start with 400 units, increase by 25%, then decrease by 20%. Do you return to 400? Explain why these unequal rates can undo one another.</p>
        <button type="button" className={`${button} mt-3`} aria-expanded={answer} aria-controls={`${id}-answer`} onClick={()=>setAnswer(!answer)}>{answer?'Hide explanation':'Check explanation'}</button>
        {answer&&<p id={`${id}-answer`} className="mt-3 text-sm leading-7">Yes. 25% of 400 is 100, giving 500. Then 20% of 500 is 100, giving 400. The multipliers 1.25 and 0.8 multiply to 1. Equal percentage increases and decreases instead use different amounts: +20% then −20% multiplies by 1.2 × 0.8 = 0.96, a net 4% decrease.</p>}
      </div>
    </section>
  </div>
}
