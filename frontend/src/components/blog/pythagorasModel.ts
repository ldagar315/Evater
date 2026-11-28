export type Point = { x: number; y: number }
export function pythagorasModel(a: number, b: number) {
  if (![a, b].every(n => Number.isInteger(n) && n >= 1 && n <= 12)) throw new RangeError('Side lengths must be whole numbers from 1 to 12')
  const cSquared = a * a + b * b
  const c = Math.sqrt(cSquared)
  const A = { x: 0, y: 0 }, B = { x: a, y: 0 }, C = { x: 0, y: -b }
  const squares = [
    [A, B, { x: a, y: a }, { x: 0, y: a }],
    [A, C, { x: -b, y: -b }, { x: -b, y: 0 }],
    [B, C, { x: b, y: -b - a }, { x: a + b, y: -a }],
  ]
  const scale = Math.min(400 / (a + 2 * b), 340 / (2 * a + b))
  const plot = (p: Point) => ({ x: 240 + (p.x - a / 2) * scale, y: 200 + (p.y + b / 2) * scale })
  const points = (ps: Point[]) => ps.map(p => { const q = plot(p); return `${q.x},${q.y}` }).join(' ')
  const corner = Math.min(a, b) * 0.2
  return { a, b, c, cSquared, squares, scale, plot, points, triangle: [A, B, C], rightAngle: [{ x: 0, y: -corner }, { x: corner, y: -corner }, { x: corner, y: 0 }], exactHypotenuse: Number.isInteger(c) }
}
