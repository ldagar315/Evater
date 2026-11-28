export const compressionVolumes = [100, 75, 50] as const
export type CompressionVolume = typeof compressionVolumes[number]
export const particleCount = 16
export const particleRadius = 6
/** A finite schematic, not a pressure, temperature or molecular dynamics model. */
export function compressionDrawing(volume: CompressionVolume) {
  if (!compressionVolumes.includes(volume)) throw new RangeError('Choose a supplied volume view')
  const left = 30, top = 40, height = 140, width = 3 * volume
  const particles = Array.from({length: particleCount}, (_, i) => ({
    x: left + (i % 4 + 0.5) * width / 4,
    y: top + (Math.floor(i / 4) + 0.5) * height / 4,
    radius: particleRadius,
  }))
  return {volume, left, top, width, height, particles, count: particleCount}
}
