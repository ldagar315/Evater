import {useId, useState} from 'react'
import {compressionDrawing, compressionVolumes, type CompressionVolume} from './compressionModel'
export function GasCompression() {
  const id = useId()
  const [volume, setVolume] = useState<CompressionVolume>(100)
  const [show, setShow] = useState(false)
  const model = compressionDrawing(volume)
  return <section data-compression-explorer aria-labelledby={`${id}-title`} className="my-8 rounded-2xl border border-teal-200 bg-teal-50 p-5 sm:p-7">
    <p className="text-xs font-bold uppercase tracking-widest text-teal-800">Explore a sealed gas sample</p>
    <h2 id={`${id}-title`} className="mt-2 text-xl font-bold">Smaller gaps, same particles</h2>
    <p className="mt-3 text-sm leading-6">Choose a smaller container volume. No gas escapes and the gas does not change state. What should change in its particle diagram, and what should stay the same?</p>
    <div className="mt-4 grid gap-2 sm:grid-cols-3" role="group" aria-label="Relative container volume">{compressionVolumes.map(value => <button type="button" key={value} aria-label={`Volume ${value}`} aria-pressed={volume === value} onClick={() => setVolume(value)} className="min-h-11 rounded-lg border border-teal-700 bg-white px-3 py-2 text-sm font-bold text-teal-900">{value} relative units{value === 100 ? ' (start)' : ''}</button>)}</div>
    <svg data-compression-diagram viewBox="0 0 360 230" role="img" aria-label={`Schematic sealed gas: relative volume ${volume}, 16 same-size particle symbols in a smaller space as volume falls.`} className="mt-4 w-full rounded-xl bg-white">
      <rect x="30" y="40" width="300" height="140" fill="none" stroke="#94a3b8" strokeWidth="2" strokeDasharray="5 5"/>
      <rect data-gas-container x={model.left} y={model.top} width={model.width} height={model.height} fill="#f0fdfa" stroke="#166859" strokeWidth="3"/>
      {model.particles.map((p, i) => <circle data-gas-particle key={i} cx={p.x} cy={p.y} r={p.radius} fill="#166859"/>)}
      <g fill="#334155" textAnchor="middle" fontSize="16"><text x="180" y="24">One sealed sample</text><text x="180" y="211">Relative volume: {volume}</text></g>
    </svg>
    <p className="mt-2 text-xs leading-5 text-neutral-600">The solid outline encloses the gas. The dashed outline marks the starting space, not another container of gas. All drawings use the same scale and particle-symbol size. The dots are symbols, not visible grains or a frozen arrangement of real gas particles.</p>
    <div aria-live="polite" className="mt-4 rounded-xl bg-white p-4 text-sm leading-6"><p><strong>{model.count} particle symbols. Same symbol size.</strong></p><p>{volume === 100 ? 'Starting view: use the smaller-volume buttons to compare spacing.' : `The sample occupies ${volume} of the starting 100 relative volume units. The same 16 symbols fit with less space between them overall. None have vanished or shrunk.`}</p></div>
    <p className="mt-4 text-sm leading-6">Gas particles are far apart, so their spacing can decrease when a gas is compressed. Liquid water is much harder to compress under ordinary classroom conditions: its particles are already close together. Practically incompressible does not mean absolutely no change under any pressure.</p>
    <p className="mt-2 text-xs leading-5 text-neutral-600">A finite, not-to-scale teaching diagram, not a physical simulation. Real gas particles keep moving randomly; these still views do not show their paths or speed. No pressure, force, temperature or rate is calculated. On-screen exploration only: do not pressurise sealed containers or use a syringe needle.</p>
    <button type="button" onClick={() => {setVolume(100); setShow(false)}} className="mt-4 min-h-11 rounded-lg border border-teal-700 bg-white px-3 text-sm font-bold text-teal-900">Reset gas view</button>
    <p className="mt-5 text-sm leading-6">Before checking: a different sealed gas sample is drawn with 24 same-size symbols at 120 relative volume units. It is compressed to 80 units with no gas escaping or change of state. Should the next drawing use 16 symbols, 24 smaller symbols, or 24 same-size symbols closer together?</p>
    <button type="button" aria-expanded={show} aria-controls={`${id}-answer`} onClick={() => setShow(!show)} className="mt-2 min-h-11 rounded-lg border border-teal-700 bg-white px-3 text-sm font-bold text-teal-900">{show ? 'Hide explanation' : 'Check explanation'}</button>
    {show && <p id={`${id}-answer`} className="mt-3 text-sm leading-6">Use 24 same-size symbols closer together overall. The volume is smaller, but no gas has escaped. Fewer symbols would wrongly suggest less matter; smaller symbols would wrongly suggest the particles themselves shrank. The reduced spacing explains this change in volume.</p>}
  </section>
}
