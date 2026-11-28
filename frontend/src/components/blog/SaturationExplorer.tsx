import { useId, useState } from 'react'
import { saturationModel } from './saturationModel'

export function SaturationExplorer() {
  const id = useId()
  const [solvent, setSolvent] = useState(2)
  const [added, setAdded] = useState(6)
  const [answer, setAnswer] = useState('')
  const [checked, setChecked] = useState(false)
  const m = saturationModel(solvent, added)
  const range = 'min-h-11 w-full accent-teal-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-700'
  const button = 'min-h-11 rounded-xl border border-teal-700 px-4 py-2 text-sm font-bold text-teal-900 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-700'
  return <section data-saturation-explorer="true" aria-labelledby={`${id}-heading`} className="my-10 min-w-0 rounded-2xl border border-teal-200 bg-teal-50/50 p-4 sm:p-6">
    <p className="text-xs font-extrabold uppercase tracking-widest text-teal-700">Explore a dissolving limit</p>
    <h2 id={`${id}-heading`} className="mt-2 text-2xl font-extrabold text-dark">More solvent. More room to dissolve.</h2>
    <p className="mt-3 text-sm leading-6 text-neutral-600">At one fixed temperature, this fictional solute has a limit of 4 solute units per solvent unit. Change one amount at a time. The numbers are a teaching rule, not measured salt or sugar data.</p>
    <div className="mt-5 grid min-w-0 gap-6 md:grid-cols-2">
      <figure className="min-w-0 rounded-xl bg-white p-3">
        <svg viewBox="0 0 320 320" className="block h-auto w-full" role="img" aria-labelledby={`${id}-title ${id}-desc`}>
          <title id={`${id}-title`}>Solute accounting in a schematic beaker</title>
          <desc id={`${id}-desc`}>{m.dissolved} solute units dissolved and {m.undissolved} units undissolved, out of {added} added. The solvent is shown as a blue region, not individual particles.</desc>
          <path d={`M45 ${250 - solvent * 40} H275 V270 H45 Z`} fill="#cffafe" />
          <path d="M45 55 V270 Q45 280 55 280 H265 Q275 280 275 270 V55" fill="none" stroke="#155e75" strokeWidth="3" />
          {[1,2,3,4].map(n=><g key={n}><path d={`M45 ${250-n*40} H58`} stroke="#155e75"/><text x="30" y={255-n*40} textAnchor="end" fontSize="13">{n}</text></g>)}
          {Array.from({length:m.dissolved},(_,i)=><circle key={i} data-solute="dissolved" cx={75+(i%4)*55} cy={235-Math.floor(i/4)*35} r="7" fill="#0f766e" />)}
          {Array.from({length:m.undissolved},(_,i)=><rect key={i} data-solute="undissolved" x={65+(i%10)*19} y={260-Math.floor(i/10)*15} width="11" height="11" fill="#b45309" />)}
          <text x="160" y="35" textAnchor="middle" fontSize="15" fontWeight="700" fill="#134e4a">Fixed temperature · after mixing</text>
          <text x="160" y="308" textAnchor="middle" fontSize="13" fill="#475569">Blue region = solvent amount (units)</text>
        </svg>
        <figcaption className="text-sm leading-6 text-neutral-600">Green circles: dissolved solute units. Amber squares: extra undissolved solute units. Symbols are accounting markers, not visible real particles; liquid level is schematic, not total solution volume.</figcaption>
      </figure>
      <div className="min-w-0 space-y-5">
        <div><label className="mb-2 block text-sm font-bold" htmlFor={`${id}-solvent`}>Solvent amount: {solvent} {solvent === 1 ? 'unit' : 'units'}</label><input className={range} id={`${id}-solvent`} type="range" min="1" max="4" step="1" value={solvent} onChange={e=>setSolvent(Number(e.target.value))}/></div>
        <div><label className="mb-2 block text-sm font-bold" htmlFor={`${id}-added`}>Solute added: {added} units</label><input className={range} id={`${id}-added`} type="range" min="0" max="20" step="1" value={added} onChange={e=>setAdded(Number(e.target.value))}/></div>
        <div className="rounded-xl bg-white p-4 text-sm leading-7" aria-live="polite" aria-atomic="true" data-saturation-result="true">
          <p>Limit: {solvent} × 4 = <strong>{m.limit} solute units</strong>.</p>
          <p>{m.dissolved} dissolved + {m.undissolved} undissolved = {added} added.</p>
          <p className="mt-2 font-bold">{m.saturated ? 'Saturated: the solution has reached its limit.' : `Unsaturated: room for ${m.room} more solute ${m.room === 1 ? 'unit' : 'units'}.`}</p>
          <p>{m.undissolved > 0 ? 'Extra solid remains outside the solution. More stirring cannot raise this fixed-temperature limit.' : m.saturated ? 'At the exact limit, no extra solid is left. A saturated solution can still look clear.' : 'All the added solute dissolves. It is still present, spread through the solvent.'}</p>
        </div>
        <button className={button} type="button" onClick={()=>{setSolvent(2);setAdded(6)}}>Reset amounts</button>
      </div>
    </div>
    <div className="mt-6 rounded-xl border border-teal-200 bg-white p-4">
      <h3 className="font-extrabold text-dark">Predict a different case</h3>
      <p className="mt-2 text-sm leading-6">At the same temperature, 15 solute units are added to 3 solvent units. Then 1 more solvent unit is added, with no extra solute. How much remains undissolved after each step?</p>
      <fieldset className="mt-3 space-y-2"><legend className="sr-only">Choose the undissolved amounts</legend>{[['3 then 0','3 units, then 0 units'],['3 then 3','3 units, then 3 units'],['0 then 0','0 units, then 0 units']].map(([value,label])=><label key={value} className="flex min-h-11 cursor-pointer items-center gap-3 rounded-lg border border-neutral-200 px-3 py-2 text-sm"><input type="radio" name={`${id}-prediction`} value={value} checked={answer===value} onChange={()=>{setAnswer(value);setChecked(false)}}/>{label}</label>)}</fieldset>
      <button type="button" className={`${button} mt-3 disabled:opacity-50`} disabled={!answer} onClick={()=>setChecked(true)}>Check prediction</button>
      {checked && <p role="status" className="mt-3 text-sm leading-7">{answer==='3 then 0'?'Correct.':'Try again using the two limits.'} First limit: 3 × 4 = 12, leaving 15 − 12 = 3 undissolved. After adding solvent, the limit is 4 × 4 = 16. All 15 units can dissolve, so none remains undissolved and the solution is now unsaturated.</p>}
    </div>
    <p className="mt-4 text-sm leading-6 text-neutral-600">Idealised equilibrium accounting: the solute and solvent stay the same, temperature stays fixed, and enough mixing time has passed. This does not model dissolving speed, heating, gases or real molecular structure. Use the on-screen model only; no tasting or heating is needed.</p>
  </section>
}
