import { useId, useState } from 'react'
import { contactPressure } from './pressureModel'

export function ContactPressure() {
  const id = useId(), [force, setForce] = useState(60), [area, setArea] = useState(3), [show, setShow] = useState(false)
  const m = contactPressure(force, area)
  const exact = m.denominator === 1 ? `${m.numerator}` : `${m.numerator}/${m.denominator}`
  const label = m.denominator === 1 ? `${exact} Pa` : `${exact} Pa (about ${m.pascals.toFixed(1)} Pa)`
  return <section data-contact-pressure aria-labelledby={`${id}-title`} className="my-8 rounded-2xl border border-teal-200 bg-teal-50 p-5 sm:p-7">
    <p className="text-xs font-bold uppercase tracking-widest text-teal-800">Explore force and contact area</p>
    <h2 id={`${id}-title`} className="mt-2 text-xl font-bold">Does a bigger force always mean bigger pressure?</h2>
    <p className="mt-3 text-sm leading-6">Compare the same reference pad with a pad you control. This ideal model spreads a perpendicular force evenly over the whole contact area. Predict the result before moving a control.</p>
    <div className="my-5 grid gap-5 sm:grid-cols-2">
      <label className="text-sm font-semibold">Perpendicular force: {force} N<input aria-label="Perpendicular force" type="range" min="20" max="200" step="20" value={force} onChange={e=>setForce(Number(e.target.value))} className="mt-2 block h-8 w-full accent-teal-700" /></label>
      <label className="text-sm font-semibold">Contact area: {(area/10).toFixed(1)} m²<input aria-label="Contact area" type="range" min="1" max="10" step="1" value={area} onChange={e=>setArea(Number(e.target.value))} className="mt-2 block h-8 w-full accent-teal-700" /></label>
    </div>
    <div className="grid gap-4 sm:grid-cols-2">{[{name:'Reference pad',f:60,a:3,p:'200 Pa',key:'reference'},{name:'Your pad',f:force,a:area,p:label,key:'current'}].map(p=><figure key={p.key} className="rounded-xl border border-teal-100 bg-white p-3">
      <figcaption className="text-sm font-bold">{p.name}</figcaption>
      <svg data-pressure-pad={p.key} viewBox="0 0 290 205" role="img" aria-label={`${p.name}: ${p.f} newtons over ${(p.a/10).toFixed(1)} square metres. Pressure ${p.p}.`} className="mt-2 w-full">
        <line x1="145" x2="145" y1={95-p.f*.35} y2="95" stroke="#9a3412" strokeWidth="4" />
        <path d="M 137 85 L 145 97 L 153 85" fill="none" stroke="#9a3412" strokeWidth="3" />
        <rect data-pad-rectangle x={145-p.a*11} y="108" width={p.a*22} height="35" fill="#99e0cd" stroke="#176b59" strokeWidth="2" />
        <line x1="20" x2="270" y1="145" y2="145" stroke="#60746e" strokeWidth="2" />
        <text x="145" y="166" textAnchor="middle" fontSize="15">{(p.a/10).toFixed(1)} m² contact area</text>
        <text x="145" y="189" textAnchor="middle" fontSize="15">{p.f} N perpendicular force</text>
      </svg>
      <p className="text-sm leading-6">{p.f} ÷ {(p.a/10).toFixed(1)} = <strong>{p.p}</strong></p>
    </figure>)}</div>
    <p className="mt-3 text-xs leading-5 text-neutral-600">Schematic side view, not a physical-scale drawing. Pad width represents contact area at a fixed depth; both panels use the same area scale and force-arrow scale. Arrows show total force, not pressure. No material deformation or weather is simulated.</p>
    <div aria-live="polite" className="mt-4 rounded-xl bg-white p-4 text-sm leading-6"><p><strong>Your pressure: {label}.</strong> {m.pascals === 200 ? 'Equal to the reference.' : m.pascals > 200 ? 'Greater than the reference.' : 'Less than the reference.'}</p><p>Pressure = force ÷ area. Units: N/m² = Pa. At fixed force, doubling the contact area halves the pressure. Doubling both leaves pressure unchanged.</p></div>
    <div className="mt-4 flex flex-wrap gap-2"><button type="button" onClick={()=>{setForce(60);setArea(3);setShow(false)}} className="min-h-11 rounded-lg border border-teal-700 bg-white px-3 text-sm font-bold text-teal-900">Reset: 60 N, 0.3 m²</button><button type="button" onClick={()=>{setForce(120);setArea(6)}} className="min-h-11 rounded-lg border border-teal-700 bg-white px-3 text-sm font-bold text-teal-900">Double force and area</button></div>
    <p className="mt-5 text-sm leading-6">Before checking: pad A has 80 N over 0.2 m²; pad B has 120 N over 0.6 m². Which has greater pressure? Explain why comparing force alone fails.</p>
    <button type="button" aria-expanded={show} aria-controls={`${id}-answer`} onClick={()=>setShow(!show)} className="mt-2 min-h-11 rounded-lg border border-teal-700 bg-white px-3 text-sm font-bold text-teal-900">{show?'Hide explanation':'Check explanation'}</button>
    {show&&<p id={`${id}-answer`} className="mt-3 text-sm leading-6">A: 80 ÷ 0.2 = 400 Pa. B: 120 ÷ 0.6 = 200 Pa. A has twice the pressure despite the smaller force. Its force acts over one-third of B's area, so both quantities must be compared. This calculation is a model, not advice for testing pressure on a person.</p>}
  </section>
}
