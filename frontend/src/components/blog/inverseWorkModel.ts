export function inverseWork(workers: number) {
  if (!Number.isInteger(workers) || workers < 1 || workers > 12) throw new RangeError('Use 1 to 12 equally productive workers.')
  const gcd = (a: number, b: number): number => b === 0 ? a : gcd(b, a % b)
  const g = gcd(24, workers)
  return { workers, workerDays: 24, days: 24/workers, numerator: 24/g, denominator: workers/g, workInSixDays: workers*6 }
}
