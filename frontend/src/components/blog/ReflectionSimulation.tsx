import { useId, useState } from 'react'
import { reflectionModel } from './reflectionModel'

export function ReflectionSimulation() {
  const id = useId()
  const [angle, setAngle] = useState(45)
  const model = reflectionModel(angle)
  return <section data-reflection-simulation="true" aria-labelledby={`${id}-heading`} className="my-10 scroll-mt-24 rounded-2xl border border-teal-200 bg-teal-50/50 p-4 sm:p-6">
    <p className="text-xs font-extrabold uppercase tracking-widest text-teal-700">Explore a ray diagram</p>
    <h2 id={`${id}-heading`} className="mt-2 text-2xl font-extrabold text-dark">Measure from the normal, not the mirror</h2>
    <p className="mt-3 text-sm leading-6 text-neutral-600">Change the incoming ray's angle. The reflected ray makes the same angle with the normal, a line perpendicular to the mirror at the point where light strikes.</p>
    <div className="mt-5 grid min-w-0 gap-6 md:grid-cols-2 md:items-center">
      <figure className="min-w-0 rounded-xl border border-teal-100 bg-white p-2">
        <div className="flex flex-wrap justify-center gap-x-4 gap-y-1 p-2 text-xs font-bold"><span className="text-teal-800">Solid: incident ray</span><span className="text-orange-800">Dashed: reflected ray</span></div>
        <svg viewBox="0 0 420 310" role="img" aria-labelledby={`${id}-title ${id}-desc`} className="block h-auto w-full">
          <title id={`${id}-title`}>Reflection from a plane mirror</title>
          <desc id={`${id}-desc`}>The incident ray travels towards the mirror, and the reflected ray travels away. Both make {angle} degrees with the normal. {angle === 0 ? 'At zero degrees they share the same line and travel in opposite directions.' : `Each makes ${model.surfaceAngle} degrees with the mirror surface.`}</desc>
          <defs>
            <marker id={`${id}-in`} markerWidth="6" markerHeight="6" refX="4" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L5,3 L0,6 Z" fill="#0f766e" /></marker>
            <marker id={`${id}-out`} markerWidth="6" markerHeight="6" refX="4" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L5,3 L0,6 Z" fill="#9a3412" /></marker>
          </defs>
          <path d="M16 250 H404" stroke="#334155" strokeWidth="4" />
          <path d="M16 258 H404" stroke="#94a3b8" strokeWidth="8" />
          <text x="210" y="294" textAnchor="middle" fontSize="17" fill="#334155">Plane mirror</text>
          <path d="M210 24 V272" stroke="#64748b" strokeDasharray="4 6" strokeWidth="2" />
          <text x="221" y="36" fontSize="16" fill="#475569">Normal</text>
          <path data-incident-ray="true" d={model.incidentPath} stroke="#0f766e" strokeWidth="4" fill="none" />
          <path data-reflected-ray="true" d={model.reflectedPath} stroke="#9a3412" strokeWidth="3" strokeDasharray="9 7" fill="none" />
          <path d={model.incidentArrow} stroke="#0f766e" strokeWidth="3" markerEnd={`url(#${id}-in)`} />
          <path d={model.reflectedArrow} stroke="#9a3412" strokeWidth="3" markerEnd={`url(#${id}-out)`} />
          {angle > 0 && <><path d={model.incidentArc} stroke="#0f766e" strokeWidth="2" fill="none" /><path d={model.reflectedArc} stroke="#9a3412" strokeWidth="2" fill="none" /></>}
          <circle cx="210" cy="250" r="4" fill="#334155" />
        </svg>
        <figcaption className="px-3 pb-3 text-center text-sm font-semibold text-teal-900">Arrows show the direction light travels. The normal is a reference line, not another ray.</figcaption>
      </figure>
      <div className="min-w-0 space-y-5">
        <div><label htmlFor={`${id}-angle`} className="mb-2 block text-sm font-bold text-dark">Angle of incidence: {angle} degrees</label><input id={`${id}-angle`} type="range" min="0" max="75" step="1" value={angle} onChange={event => setAngle(Number(event.target.value))} className="min-h-11 w-full accent-teal-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-teal-700" /></div>
        <div className="rounded-xl bg-white p-4 text-sm leading-7 text-neutral-700" aria-live="polite" aria-atomic="true">
          <p data-reflection-result="true">Angle of incidence = angle of reflection = <strong>{angle} degrees</strong>.</p>
          <p>Angle with the mirror surface: 90 - {angle} = <strong>{model.surfaceAngle} degrees</strong>.</p>
          <p>{angle === 0 ? 'The light returns along the same line. The two rays overlap, but their arrows point in opposite directions.' : 'The two rays and the normal lie in the same plane, shown here as a flat diagram.'}</p>
        </div>
        <button type="button" onClick={() => setAngle(45)} className="min-h-11 rounded-xl border border-teal-700 px-4 text-sm font-bold text-teal-800 hover:bg-teal-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-700">Reset angle</button>
      </div>
    </div>
    <p className="mt-5 text-sm leading-6 text-neutral-600">Try 60 degrees from the normal. Each ray is only 30 degrees from the mirror surface. Then try zero: both angles measured from the normal are zero.</p>
    <p className="mt-3 text-sm leading-6 text-neutral-600">This model shows one ray reflecting from a flat, ideal mirror. It does not model image formation, curved mirrors, refraction or lenses. A 2D diagram keeps the angles in their shared plane clear.</p>
  </section>
}
