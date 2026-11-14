import { useId, useState } from 'react'
import { pythagorasModel } from './pythagorasModel'

export function PythagorasSimulation() {
  const id = useId()
  const [a, setA] = useState(3)
  const [b, setB] = useState(4)
  const m = pythagorasModel(a, b)
  return <section data-pythagoras-simulation="true" aria-labelledby={`${id}-heading`} className="my-10 scroll-mt-24 rounded-2xl border border-teal-200 bg-teal-50/50 p-4 sm:p-6">
    <p className="text-xs font-extrabold uppercase tracking-widest text-teal-700">Explore the square areas</p>
    <h2 id={`${id}-heading`} className="mt-2 text-2xl font-extrabold text-dark">Two square areas add up to the third</h2>
    <p className="mt-3 text-sm leading-6 text-neutral-600">Change the two perpendicular sides of this right-angled triangle. Each square is drawn to scale on one side. The hypotenuse is opposite the right angle.</p>
    <div className="mt-5 grid min-w-0 gap-6 md:grid-cols-2 md:items-center">
      <figure className="min-w-0 rounded-xl border border-teal-100 bg-white p-2">
        <svg viewBox="0 0 480 400" role="img" aria-labelledby={`${id}-title ${id}-desc`} className="block h-auto w-full">
          <title id={`${id}-title`}>Three squares on a right-angled triangle</title>
          <desc id={`${id}-desc`}>Perpendicular sides a = {a} cm and b = {b} cm have square areas {a * a} and {b * b} square cm. The square on the hypotenuse has area {m.cSquared} square cm. The small corner mark shows a right angle.</desc>
          <defs><pattern id={`${id}-b`} width="10" height="10" patternUnits="userSpaceOnUse"><path d="M0 10 L10 0" stroke="#9a3412" strokeWidth="1" /></pattern><pattern id={`${id}-c`} width="12" height="12" patternUnits="userSpaceOnUse"><circle cx="6" cy="6" r="1.5" fill="#4338ca" /></pattern></defs>
          {m.squares.map((square, i) => <polygon key={i} data-square={i} points={m.points(square)} fill={['#ccfbf1', '#ffedd5', '#e0e7ff'][i]} stroke={['#0f766e', '#9a3412', '#4338ca'][i]} strokeWidth="2" />)}
          {m.squares.slice(1).map((square, i) => <polygon key={i} points={m.points(square)} fill={`url(#${id}-${i === 0 ? 'b' : 'c'})`} />)}
          <polygon points={m.points(m.triangle)} fill="#fff" stroke="#334155" strokeWidth="2" />
          <polyline points={m.points(m.rightAngle)} fill="none" stroke="#334155" strokeWidth="2" />
        </svg>
        <figcaption className="space-y-2 px-3 pb-3 text-sm font-semibold"><p className="text-teal-800">Solid teal: a-square, {a * a} square cm</p><p className="text-orange-800">Striped orange: b-square, {b * b} square cm</p><p className="text-indigo-800">Dotted purple: hypotenuse square, {m.cSquared} square cm</p></figcaption>
      </figure>
      <div className="min-w-0 space-y-5">
        <div><label htmlFor={`${id}-a`} className="mb-2 block text-sm font-bold text-dark">Perpendicular side a: {a} cm</label><input id={`${id}-a`} data-side="a" type="range" min="1" max="12" step="1" value={a} onChange={e => setA(Number(e.target.value))} className="min-h-11 w-full accent-teal-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-teal-700" /></div>
        <div><label htmlFor={`${id}-b`} className="mb-2 block text-sm font-bold text-dark">Perpendicular side b: {b} cm</label><input id={`${id}-b`} data-side="b" type="range" min="1" max="12" step="1" value={b} onChange={e => setB(Number(e.target.value))} className="min-h-11 w-full accent-teal-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-teal-700" /></div>
        <div className="rounded-xl bg-white p-4 text-sm leading-7 text-neutral-700" aria-live="polite" aria-atomic="true"><p>a squared + b squared = c squared</p><p data-pythagoras-result="true"><strong>{a * a} + {b * b} = {m.cSquared} square cm</strong></p><p>Hypotenuse c = square root of {m.cSquared}{m.exactHypotenuse ? ` = ${m.c} cm.` : `, approximately ${m.c.toFixed(3)} cm.`}</p><p>The areas add. The side lengths do not: a + b is not c.</p></div>
        <button type="button" onClick={() => { setA(3); setB(4) }} className="min-h-11 rounded-xl border border-teal-700 px-4 text-sm font-bold text-teal-800 hover:bg-teal-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-700">Reset to 3 and 4</button>
      </div>
    </div>
    <p className="mt-5 text-sm leading-6 text-neutral-600">Try a = 5 and b = 12. Then try two sides of 1 cm: the hypotenuse is not a whole number, but its square still has area 2 square cm.</p>
    <p className="mt-3 text-sm leading-6 text-neutral-600">This 2D model always keeps a right angle. It illustrates the relationship, not a proof for every triangle. The drawing rescales to fit; compare the squares within one view, not their screen size across different settings.</p>
  </section>
}
