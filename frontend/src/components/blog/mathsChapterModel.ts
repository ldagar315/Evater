export function powerGrowth(base: number, steps: number) {
  const b = Number.isFinite(base) ? Math.max(2, Math.min(5, Math.round(base))) : 2
  const n = Number.isFinite(steps) ? Math.max(0, Math.min(6, Math.round(steps))) : 0
  return { base: b, steps: n, added: 1 + b * n, multiplied: b ** n, values: Array.from({ length: n + 1 }, (_, i) => ({ step: i, added: 1 + b * i, multiplied: b ** i })) }
}
export function areaShape(base: number, height: number, shift: number) {
  const clamp = (x: number, fallback: number, lo: number, hi: number) => Number.isFinite(x) ? Math.max(lo, Math.min(hi, Math.round(x))) : fallback
  const b = clamp(base, 8, 4, 12), h = clamp(height, 3, 2, 6), s = clamp(shift, 0, -3, 3)
  return { base: b, height: h, shift: s, area: b * h, triangleArea: b * h / 2, slopingSide: Math.hypot(s, h) }
}
