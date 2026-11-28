export type DensityState = 'floats' | 'suspended' | 'sinks'

// Grams and cubic centimetres throughout. The liquid density is supplied,
// not a claim about any real liquid's temperature or composition.
export function densityModel(mass: number, volume: number, liquidDensity: number) {
  if (![mass, volume, liquidDensity].every(Number.isFinite) || mass <= 0 || volume <= 0 || liquidDensity <= 0) {
    throw new RangeError('Mass, outer volume and liquid density must be positive finite numbers.')
  }
  const density = mass / volume
  const capacityMass = liquidDensity * volume
  const tolerance = Number.EPSILON * Math.max(mass, capacityMass) * 8
  const difference = mass - capacityMass
  const state: DensityState = Math.abs(difference) <= tolerance ? 'suspended' : difference < 0 ? 'floats' : 'sinks'
  return {
    density, state,
    // Only the floating equilibrium has a partially submerged fraction.
    submergedFraction: state === 'floats' ? mass / capacityMass : 1,
    maxDisplacedMass: capacityMass,
    displacedMass: state === 'floats' ? mass : capacityMass,
    result: state === 'floats' ? 'Floats partly submerged' : state === 'suspended' ? 'Can remain suspended when fully submerged' : 'Sinks when released in the liquid',
  }
}
