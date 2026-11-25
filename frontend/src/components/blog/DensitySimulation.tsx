import { useId, useState } from 'react'
import { densityModel } from './densityModel'

export function DensitySimulation() {
  const id = useId()
  const [mass, setMass] = useState(120)
  const [volume, setVolume] = useState(200)
  const [liquid, setLiquid] = useState(1)
  const [answer, setAnswer] = useState(false)
  const model = densityModel(mass, volume, liquid)
  const waterline = 100
  const height = 94
  const top = model.state === 'floats' ? waterline - height * (1 - model.submergedFraction) : 140
  const format = (value: number) => Number(value.toFixed(3)).toString()
  const range = 'min-h-11 w-full accent-teal-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-teal-700'
  const preset = (m: number, v: number, l: number) => { setMass(m); setVolume(v); setLiquid(l) }
  return <section data-density-simulation="true" aria-labelledby={`${id}-heading`} className="my-10 scroll-mt-24 rounded-2xl border border-teal-200 bg-teal-50/50 p-4 sm:p-6">
    <p className="text-xs font-extrabold uppercase tracking-widest text-teal-700">Explore a floating model</p>
    <h2 id={`${id}-heading`} className="mt-2 text-2xl font-extrabold text-dark">More mass does not always mean more dense.</h2>
    <p className="mt-3 text-sm leading-6 text-neutral-700">Change the supplied values for a fictional sealed block. Predict the result before comparing its average density with the liquid's density.</p>
    <div className="mt-5 grid min-w-0 gap-6 md:grid-cols-2 md:items-start">
      <figure className="min-w-0 rounded-xl border border-teal-100 bg-white p-3">
        <svg viewBox="0 0 360 310" role="img" aria-labelledby={`${id}-title ${id}-desc`} className="block h-auto w-full">
          <title id={`${id}-title`}>Density comparison for a sealed rigid block</title>
          <desc id={`${id}-desc`}>{model.result}. Block density {format(model.density)} grams per cubic centimetre; liquid density {liquid}. {model.state === 'floats' ? `${format(model.submergedFraction * 100)} percent of its volume is below the surface at rest.` : 'The block is shown fully submerged.'} This is not a motion animation.</desc>
          <path d="M32 55 V260 Q32 275 47 275 H313 Q328 275 328 260 V55" fill="#f0fdfa" stroke="#115e59" strokeWidth="3" />
          <path d="M34 100 H326 V260 Q326 273 313 273 H47 Q34 273 34 260Z" fill="#bae6fd" />
          <path d="M34 100 H326" stroke="#0369a1" strokeWidth="2" strokeDasharray="6 4" />
          <rect x="126" y={top} width="108" height={height} rx="3" fill="#fbbf24" stroke="#92400e" strokeWidth="2" />
          <path d={`M126 ${top} l14 -10 h108 l-14 10 M234 ${top} l14 -10 v94 l-14 10`} fill="none" stroke="#92400e" strokeWidth="2" />
          <text x="180" y={top + 43} textAnchor="middle" fontSize="18" fontWeight="700" fill="#451a03">Sealed block</text>
          <text x="180" y="295" textAnchor="middle" fontSize="16" fill="#0c4a6e">Liquid: {liquid} g/cm³</text>
        </svg>
        <figcaption className="text-sm leading-6 text-neutral-700">Diagram not to scale. For floating equilibrium, the fraction of outer volume submerged is density of block ÷ density of liquid. The block's drawn size stays fixed so you can compare the waterline.</figcaption>
      </figure>
      <div className="min-w-0 space-y-4">
        <div><label htmlFor={`${id}-mass`} className="block text-sm font-bold text-dark">Total mass: {mass} g</label><input id={`${id}-mass`} type="range" min="20" max="600" step="10" value={mass} onChange={e => setMass(Number(e.target.value))} className={range} /></div>
        <div><label htmlFor={`${id}-volume`} className="block text-sm font-bold text-dark">Whole outer volume: {volume} cm³</label><input id={`${id}-volume`} type="range" min="50" max="500" step="10" value={volume} onChange={e => setVolume(Number(e.target.value))} className={range} /></div>
        <div><label htmlFor={`${id}-liquid`} className="block text-sm font-bold text-dark">Supplied liquid density</label><select id={`${id}-liquid`} value={liquid} onChange={e => setLiquid(Number(e.target.value))} className="mt-2 min-h-11 w-full rounded-lg border border-teal-300 bg-white px-3 text-dark"><option value="0.8">0.8 g/cm³</option><option value="1">1 g/cm³ (water model)</option><option value="1.2">1.2 g/cm³</option></select></div>
        <div aria-live="polite" aria-atomic="true" className="rounded-xl bg-white p-4 text-sm leading-7 text-neutral-700">
          <p>Average density = {mass} ÷ {volume} = <strong>{format(model.density)} g/cm³</strong> (rounded where needed).</p>
          <p data-density-result="true" className="font-bold text-teal-900">{model.result}.</p>
          <p>{model.state === 'floats' ? `${format(model.submergedFraction * 100)}% of outer volume is submerged at rest. Buoyancy then balances weight; gravity has not vanished.` : model.state === 'suspended' ? 'Equal densities: buoyancy balances weight when fully submerged. Shown at an arbitrary depth, not resting on a support.' : 'Even fully submerged, the displaced liquid has less mass than the block. Buoyancy is present but is smaller than the block’s weight. Shown before it reaches the bottom.'}</p>
        </div>
      </div>
    </div>
    <div className="mt-5 flex flex-wrap gap-3">{[
      ['Reset: floating block', 120, 200, 1], ['Equal densities', 200, 200, 1], ['Lighter but sinking', 60, 50, 1],
    ].map(([label, m, v, l]) => <button key={label} type="button" onClick={() => preset(Number(m), Number(v), Number(l))} className="min-h-11 rounded-xl border border-teal-700 px-4 py-2 text-sm font-bold text-teal-800 hover:bg-teal-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-700">{label}</button>)}</div>
    <p className="mt-5 text-sm leading-6 text-neutral-700">Model assumptions: a sealed rigid object with no leaking, absorption, dissolving or shape change; a uniform liquid at rest; no surface tension or other support. Total mass includes everything inside the outer volume. This is a digital comparison, not an instruction to put unknown objects in water.</p>
    <div className="mt-5 rounded-xl border border-teal-200 bg-white p-4 text-sm leading-7 text-neutral-700">
      <h3 className="font-bold text-dark">Try without moving the sliders</h3><p>A sealed object has mass 180 g and outer volume 300 cm³. Compare it in liquids with supplied densities 0.8 and 1.2 g/cm³. Does the object's density change? In which liquid is less of its volume submerged at rest?</p>
      <button type="button" aria-expanded={answer} onClick={() => setAnswer(!answer)} className="mt-3 min-h-11 rounded-lg bg-teal-800 px-4 py-2 font-bold text-white">{answer ? 'Hide reasoning' : 'Check reasoning'}</button>
      {answer && <p className="mt-3">Its density stays 180 ÷ 300 = 0.6 g/cm³. It floats in both. The submerged fractions are 0.6 ÷ 0.8 = 0.75 (75%) and 0.6 ÷ 1.2 = 0.5 (50%). Less volume is submerged in the denser liquid. Changing the liquid does not change this sealed object's mass or volume.</p>}
    </div>
  </section>
}
