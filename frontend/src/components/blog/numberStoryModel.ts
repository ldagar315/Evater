export function positionalModel(amount: number, base: number) {
  if (!Number.isInteger(amount) || amount < 0 || amount > 255) throw new RangeError('Amount must be an integer from 0 to 255')
  if (!Number.isInteger(base) || base < 2 || base > 10) throw new RangeError('Base must be an integer from 2 to 10')
  const places: { digit: number; weight: number; contribution: number }[] = []
  let rest = amount, weight = 1
  do { const digit = rest % base; places.unshift({ digit, weight, contribution: digit * weight }); rest = Math.floor(rest / base); weight *= base } while (rest > 0)
  return { amount, base, places, numeral: places.map(p => p.digit).join('') }
}
