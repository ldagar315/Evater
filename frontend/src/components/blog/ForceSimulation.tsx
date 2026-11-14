import { useId, useState } from 'react'
import { forceModel } from './forceModel'

export function ForceSimulation() {
  const id = useId()
  const [right, setRight] = useState(6)
  const [left, setLeft] = useState(4)
  const model = forceModel(right, left)
  const range = 'min-h-11 w-full accent-teal-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-teal-700'
  return <section data-force-simulation="true" aria-labelledby={`${id}-heading`} className="my-10 scroll-mt-24 rounded-2xl border border-teal-200 bg-teal-50/50 p-4 sm:p-6">
    <p className="text-xs font-extrabold uppercase tracking-widest text-teal-700">Explore with a 3D block</p>
    <h2 id={`${id}-heading`} className="mt-2 text-2xl font-extrabold text-dark">Two pushes. One net force.</h2>
    <p className="mt-3 text-sm leading-6 text-neutral-600">Change the pushes on opposite sides of the same block. Compare the arrow lengths, then work out the horizontal net force.</p>
    <div className="mt-5 grid min-w-0 gap-6 md:grid-cols-2 md:items-center">
      <figure className="min-w-0 rounded-xl border border-teal-100 bg-white p-2">
        <svg viewBox="0 0 360 270" role="img" aria-labelledby={`${id}-title ${id}-desc`} className="block h-auto w-full">
          <title id={`${id}-title`}>Opposing pushes on one block</title>
          <desc id={`${id}-desc`}>{right} newtons to the right and {left} newtons to the left. Net force: {model.result} This diagram shows forces, not the block's motion.</desc>
          <ellipse cx="192" cy="216" rx="108" ry="13" fill="#ccfbf1" />
          <g stroke="#115e59" strokeWidth="2" strokeLinejoin="round">
            <path d="M140 105 L220 105 L220 205 L140 205 Z" fill="#5eead4" />
            <path d="M140 105 L170 82 L250 82 L220 105 Z" fill="#ccfbf1" />
            <path d="M220 105 L250 82 L250 182 L220 205 Z" fill="#0f766e" />
          </g>
          <g strokeWidth="6" fill="none" strokeLinecap="round">
            {right > 0 && <g stroke="#0f766e" data-force-arrow="right"><path d={`M${140 - model.rightLength} 154 H140`} /><path d="M132 147 L140 154 L132 161" /></g>}
            {left > 0 && <g stroke="#9a3412" data-force-arrow="left"><path d={`M${220 + model.leftLength} 154 H220`} /><path d="M228 147 L220 154 L228 161" /></g>}
          </g>
          <text x="82" y="131" textAnchor="middle" fontSize="17" fontWeight="700" fill="#115e59">{right} N right</text>
          <text x="282" y="205" textAnchor="middle" fontSize="17" fontWeight="700" fill="#9a3412">{left} N left</text>
          <text x="180" y="250" textAnchor="middle" fontSize="17" fontWeight="700" fill="#134e4a">{model.net === 0 ? 'Balanced: 0 N' : `Net: ${model.magnitude} N ${model.direction}`}</text>
        </svg>
        <figcaption className="px-3 pb-3 text-center text-sm font-semibold text-teal-900">Arrows show force, not speed. The block stays in place so you can compare them.</figcaption>
      </figure>
      <div className="min-w-0 space-y-5">
        <div><label htmlFor={`${id}-right`} className="mb-2 block text-sm font-bold text-dark">Push to the right: {right} N</label><input id={`${id}-right`} type="range" min="0" max="10" step="1" value={right} onChange={event => setRight(Number(event.target.value))} className={range} /></div>
        <div><label htmlFor={`${id}-left`} className="mb-2 block text-sm font-bold text-dark">Push to the left: {left} N</label><input id={`${id}-left`} type="range" min="0" max="10" step="1" value={left} onChange={event => setLeft(Number(event.target.value))} className={range} /></div>
        <div className="rounded-xl bg-white p-4 text-sm leading-7 text-neutral-700" aria-live="polite" aria-atomic="true">
          <p>Take right as positive: {right} - {left} = <strong>{model.net} N</strong>.</p>
          <p data-force-result="true"><strong>{model.result}</strong></p>
          <p>{model.net === 0 ? 'No change in motion is caused by these horizontal forces. Balanced forces do not mean a moving object must stop.' : 'This net force changes the block\'s motion. Its direction is not necessarily the direction the block is already moving.'}</p>
        </div>
        <button type="button" onClick={() => { setRight(6); setLeft(4) }} className="min-h-11 rounded-xl border border-teal-700 px-4 text-sm font-bold text-teal-800 hover:bg-teal-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-700">Reset pushes</button>
      </div>
    </div>
    <p className="mt-5 text-sm leading-6 text-neutral-600">Simplified model: both pushes act on one rigid block along a horizontal line. Friction and turning are ignored; weight and support balance vertically. This is a force diagram, not a prediction of speed or distance.</p>
    <p className="mt-3 text-sm leading-6 text-neutral-600">Try equal pushes, then change just one. Zero net force means no change in motion, not necessarily no motion.</p>
  </section>
}
