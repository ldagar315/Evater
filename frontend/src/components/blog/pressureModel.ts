export function contactPressure(force: number, areaTenths: number) {
  if (!Number.isInteger(force) || force < 20 || force > 200 || force % 20 !== 0 || !Number.isInteger(areaTenths) || areaTenths < 1 || areaTenths > 10) throw new RangeError('Use force 20 to 200 N in steps of 20 and area 0.1 to 1.0 square metres.')
  const numerator = force * 10
  const divisor = (a: number, b: number): number => b === 0 ? a : divisor(b, a % b)
  const g = divisor(numerator, areaTenths)
  return { force, areaTenths, area: areaTenths / 10, numerator: numerator / g, denominator: areaTenths / g, pascals: numerator / areaTenths, baselineRatio: numerator / areaTenths / 200 }
}
