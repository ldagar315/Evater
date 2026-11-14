// A one-dimensional force diagram, not a time-stepped motion simulation.
export function forceModel(right: number, left: number) {
  if (![right, left].every(n => Number.isInteger(n) && n >= 0 && n <= 10)) {
    throw new RangeError('Each push must be a whole number from 0 to 10 N.')
  }
  const net = right - left
  return {
    right, left, net, magnitude: Math.abs(net),
    direction: net > 0 ? 'right' : net < 0 ? 'left' : 'balanced',
    rightLength: right * 8, leftLength: left * 8,
    result: net === 0 ? '0 N: the horizontal forces are balanced.' : `${Math.abs(net)} N to the ${net > 0 ? 'right' : 'left'}.`,
  }
}
