import { useId, useState } from 'react'
import { meanBalance } from './meanModel'
export function MeanBalance() {
 const id=useId(),[last,setLast]=useState(10),[shared,setShared]=useState(false),[show,setShow]=useState(false),m=meanBalance(last)
 const x=(n:number)=>42+n*13, y=(n:number)=>194-n*7
 return <section data-mean-balance aria-labelledby={`${id}-title`} className="my-8 rounded-2xl border border-teal-200 bg-teal-50 p-5 sm:p-7">
 <p className="text-xs font-bold uppercase tracking-widest text-teal-800">Explore the mean</p><h2 id={`${id}-title`} className="mt-2 text-xl font-bold">One value moves. Where does the mean go?</h2>
 <p className="mt-3 text-sm leading-6">Keep 4, 6 and 8 fixed. Change the fourth value, not the number of values. The mean is the total shared equally among all four entries.</p>
 <label className="mt-4 block text-sm font-semibold">Fourth value: {last}<input aria-label="Fourth value" type="range" min="0" max="20" step="1" value={last} onChange={e=>setLast(Number(e.target.value))} className="mt-2 block h-8 w-full accent-teal-700"/></label>
 <div aria-live="polite" className="my-4 rounded-xl bg-white p-4 text-sm leading-6"><p><strong>Mean = ({m.values.join(' + ')}) ÷ 4 = {m.total} ÷ 4 = {m.mean}.</strong></p><p>Compared with the starting mean 7: {m.change>0?'+':''}{m.change}. A change of 1 in one value changes the mean by 1 ÷ 4 = 0.25.</p></div>
 <h3 className="font-bold">Balance the distances</h3><p className="mt-2 text-sm leading-6">Each row is one entry on the same 0–20 number scale. The dashed line marks the mean. Coloured segments show the distances from each value to the mean, not distances between entries.</p>
 <svg data-mean-dots viewBox="0 0 350 230" role="img" aria-label={`Values ${m.values.join(', ')}. Mean ${m.mean}. Total distance below the mean ${m.below}; above ${m.above}.`} className="mt-3 w-full rounded-xl bg-white">
 <text x="175" y="20" textAnchor="middle" fontSize="16">Value (units), fixed scale 0–20</text>
 {[0,5,10,15,20].map(n=><g key={n}><line x1={x(n)} x2={x(n)} y1="35" y2="182" stroke="#e5e7eb"/><text x={x(n)} y="201" textAnchor="middle" fontSize="16">{n}</text></g>)}
 <line data-mean-line x1={x(m.mean)} x2={x(m.mean)} y1="35" y2="182" stroke="#764d89" strokeWidth="2" strokeDasharray="5 3"/>
 {m.values.map((n,i)=><g key={i}><text x="14" y={57+i*37} fontSize="16">{'ABCD'[i]}</text><line x1={x(n)} x2={x(m.mean)} y1={53+i*37} y2={53+i*37} stroke={n<m.mean?'#176b59':'#315f9b'} strokeWidth="3"/><circle data-mean-dot cx={x(n)} cy={53+i*37} r="5" fill={i===3?'#9c3e27':'#173e4a'}/></g>)}
 <text x="175" y="220" textAnchor="middle" fontSize="16">Mean = {m.mean} (dashed line)</text></svg>
 <p className="mt-3 text-sm leading-6">Below the mean: {m.below} units in total. Above the mean: {m.above} units in total. The totals balance, even when the mean is not one of the values or halfway between the smallest and largest values.</p>
 <h3 className="mt-5 font-bold">Share the same total equally</h3><button type="button" aria-pressed={shared} onClick={()=>setShared(!shared)} className="mt-2 min-h-11 rounded-lg border border-teal-700 bg-white px-3 text-sm font-bold text-teal-900">{shared?'Show original amounts':'Show equal shares'}</button>
 <svg data-mean-bars viewBox="0 0 350 240" role="img" aria-label={shared?`Equal shares: four bars of ${m.mean} units, total ${m.total}.`:`Original amounts: ${m.values.join(', ')} units, total ${m.total}.`} className="mt-3 w-full rounded-xl bg-white">
 <text x="175" y="20" textAnchor="middle" fontSize="16">{shared?'Equal shares':'Original amounts'} (units)</text>
 {[0,5,10,15,20].map(n=><g key={n}><line x1="40" x2="320" y1={y(n)} y2={y(n)} stroke="#e5e7eb"/><text x="30" y={y(n)+4} textAnchor="end" fontSize="16">{n}</text></g>)}
 {m.values.map((n,i)=><g key={i}><rect data-share-bar x={61+i*65} y={y(shared?m.mean:n)} width="34" height={(shared?m.mean:n)*7} fill={shared?'#176b59':i===3?'#9c3e27':'#315f9b'}/><text x={78+i*65} y={y(shared?m.mean:n)-6} textAnchor="middle" fontSize="16">{shared?m.mean:n}</text><text x={78+i*65} y="215" textAnchor="middle" fontSize="16">{'ABCD'[i]}</text></g>)}
 <text x="175" y="233" textAnchor="middle" fontSize="16">Total stays {m.total}; four entries stay four.</text></svg>
 <p className="mt-2 text-xs leading-5 text-neutral-600">These are abstract amounts in units, so a fractional equal share is allowed. Neither graph changes its scale. This explores changing one existing value, not adding or removing an entry.</p>
 <button type="button" onClick={()=>{setLast(10);setShared(false);setShow(false)}} className="mt-4 min-h-11 rounded-lg border border-teal-700 bg-white px-3 text-sm font-bold text-teal-900">Reset the exploration</button>
 <p className="mt-5 text-sm leading-6">Before checking: 3, 5, 7 and 9 have mean 6. Replace 9 with 13. What is the new mean? Does it increase by 4 or by 1? Explain without using the controls.</p><button type="button" aria-expanded={show} aria-controls={`${id}-answer`} onClick={()=>setShow(!show)} className="mt-2 min-h-11 rounded-lg border border-teal-700 bg-white px-3 text-sm font-bold text-teal-900">{show?'Hide explanation':'Check explanation'}</button>{show&&<p id={`${id}-answer`} className="mt-3 text-sm leading-6">The total rises from 24 to 28, but there are still four entries. The mean is 28 ÷ 4 = 7, an increase of 1. The extra 4 units are shared across all four entries. The mean is not 10: adding 4 to just one entry is different from adding 4 to every entry.</p>}
 </section>
}
