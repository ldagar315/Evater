import { useId, useState } from 'react'
import { cubeModel } from './cubeModel'

export function CubeVolumeSimulation() {
  const id = useId()
  const [side, setSide] = useState(3)
  const [layers, setLayers] = useState(3)
  const [angle, setAngle] = useState(45)
  const model = cubeModel(side, layers, angle)
  const controls = 'min-h-11 w-full accent-teal-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-teal-700'
  return (
    <section data-cube-simulation="true" aria-labelledby={`${id}-heading`} className="my-10 overflow-hidden rounded-2xl border border-teal-200 bg-teal-50/50 p-4 sm:p-6">
      <p className="text-xs font-extrabold uppercase tracking-widest text-teal-700">Explore in 3D</p>
      <h2 id={`${id}-heading`} className="mt-2 text-2xl font-extrabold text-dark">Build a cube, one layer at a time</h2>
      <p className="mt-3 text-sm leading-6 text-neutral-600">A square counts unit squares. A cube stacks square layers. Change the side length, then reveal the layers to see why volume is n x n x n.</p>
      <div className="mt-5 grid min-w-0 gap-6 md:grid-cols-2 md:items-center">
        <figure className="min-w-0 rounded-xl border border-teal-100 bg-white p-2">
          <svg viewBox="0 0 340 310" role="img" aria-labelledby={`${id}-model-title ${id}-model-desc`} className="block h-auto w-full">
            <title id={`${id}-model-title`}>{model.side} by {model.side} base with {model.layers} layers</title>
            <desc id={`${id}-model-desc`}>{model.shownVolume} unit cubes are shown. The complete cube contains {model.fullVolume} unit cubes. Rotate the view using the control below.</desc>
            <ellipse cx="170" cy="278" rx="120" ry="14" fill="#ccfbf1" />
            {model.units.map((unit) => <g key={unit.key} data-unit-cube="true" stroke="#115e59" strokeWidth="0.8" strokeLinejoin="round">
              <polygon points={unit.right} fill="#0f766e" />
              <polygon points={unit.front} fill="#2dd4bf" />
              <polygon points={unit.top} fill="#99f6e4" />
            </g>)}
          </svg>
          <figcaption className="px-3 pb-3 text-center text-sm font-semibold text-teal-900">Each small block is one cubic unit.</figcaption>
        </figure>
        <div className="min-w-0 space-y-6">
          <div>
            <label htmlFor={`${id}-side`} className="mb-3 block text-sm font-bold text-dark">Side length: {side} units</label>
            <input id={`${id}-side`} type="range" min="1" max="6" step="1" value={side} onChange={(event) => { const n = Number(event.target.value); setSide(n); setLayers(n) }} className={controls} />
          </div>
          <div>
            <label htmlFor={`${id}-layers`} className="mb-3 block text-sm font-bold text-dark">Layers shown: {layers} of {side}</label>
            <input id={`${id}-layers`} type="range" min="1" max={side} step="1" value={layers} onChange={(event) => setLayers(Number(event.target.value))} className={controls} disabled={side === 1} />
          </div>
          <div>
            <label htmlFor={`${id}-angle`} className="mb-3 block text-sm font-bold text-dark">Rotate view: {angle} degrees</label>
            <input id={`${id}-angle`} type="range" min="20" max="70" step="1" value={angle} onChange={(event) => setAngle(Number(event.target.value))} className={controls} />
          </div>
          <div className="rounded-xl bg-white p-4 text-sm leading-7 text-neutral-700" aria-live="polite" aria-atomic="true">
            <p>One layer: {side} x {side} = <strong>{model.baseArea} square units</strong>.</p>
            <p>Shown: {model.baseArea} x {layers} = <strong>{model.shownVolume} cubic units</strong>.</p>
            <p>Full cube: {side} x {side} x {side} = <strong>{model.fullVolume} cubic units</strong>.</p>
          </div>
          <button type="button" onClick={() => { setSide(3); setLayers(3); setAngle(45) }} className="min-h-11 rounded-xl border border-teal-700 px-4 text-sm font-bold text-teal-800 hover:bg-teal-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-700">Reset model</button>
        </div>
      </div>
      <p className="mt-5 text-sm leading-6 text-neutral-600">Try this: set the side to 4. Each layer has 16 unit cubes, so four layers make 64. If you show only two layers, you have a rectangular prism with 32 unit cubes, not the full cube.</p>
    </section>
  )
}
