export const PART_COUNTS = [2, 4, 5, 8, 10, 20, 25, 40] as const

export function fractionShare(shaded: number, parts: number) {
  if (!PART_COUNTS.some(n => n === parts) || !Number.isInteger(shaded) || shaded < 0 || shaded > parts) throw new RangeError('Choose a supported whole and a valid number of parts.')
  return { shaded, parts, remaining: parts - shaded, decimal: shaded / parts, percent: shaded * 100 / parts, complementDecimal: (parts - shaded) / parts, complementPercent: (parts - shaded) * 100 / parts }
}

export function successiveChanges(original: number, firstRate: number, secondRate: number) {
  if (!Number.isInteger(original) || original < 100 || original > 1000 || [firstRate, secondRate].some(n => !Number.isInteger(n) || n < -100 || n > 100)) throw new RangeError('Choose an original from 100 to 1000 and rates from -100 to 100.')
  // Integer numerators avoid accumulating decimal rounding between the two steps.
  const firstHundredths = original * (100 + firstRate)
  const finalTenThousandths = firstHundredths * (100 + secondRate)
  const first = firstHundredths / 100
  const final = finalTenThousandths / 10000
  return { original, first, final, firstChange: original * firstRate / 100, secondChange: firstHundredths * secondRate / 10000, netChange: (finalTenThousandths - original * 10000) / 10000, netPercent: ((100 + firstRate) * (100 + secondRate) - 10000) / 100, scaleMax: Math.max(original, first, final) }
}

/** All supported outputs terminate in at most four decimal places. */
export function fractionNumber(value: number): string {
  return Number(value.toFixed(4)).toLocaleString('en-IN', { maximumFractionDigits: 4 })
}
