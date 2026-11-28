export function meanBalance(last: number) {
  if (!Number.isInteger(last) || last < 0 || last > 20) throw new RangeError('Use a whole number from 0 to 20.')
  const values = [4, 6, 8, last]
  const total = values.reduce((sum, value) => sum + value, 0)
  const mean = total / values.length
  const below = values.reduce((sum, value) => sum + Math.max(mean - value, 0), 0)
  const above = values.reduce((sum, value) => sum + Math.max(value - mean, 0), 0)
  return { values, total, count: values.length, mean, below, above, change: (last - 10) / 4 }
}
