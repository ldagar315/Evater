function dimension(n: number) { return Number.isInteger(n) && n >= 20 && n <= 100 && n % 5 === 0 }
export function ratioGcd(a: number, b: number): number { while (b) { const r = a % b; a = b; b = r } return a }
export function ratioScaling(width: number, height: number) {
  if (!dimension(width) || !dimension(height)) throw new RangeError('Use dimensions from 20 to 100 in steps of 5.')
  const gcd = ratioGcd(width, height)
  return { width, height, ratio: `${width / gcd}:${height / gcd}`, widthFactor: width / 60, heightFactor: height / 40, proportional: width * 40 === height * 60, widthChange: width - 60, heightChange: height - 40 }
}
export function ratioFactor(n: number, base: number) { const g = ratioGcd(n, base); return n % base === 0 ? String(n / base) : `${n / g}/${base / g}` }
