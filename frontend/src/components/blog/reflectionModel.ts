export function reflectionModel(angle: number) {
  if (!Number.isInteger(angle) || angle < 0 || angle > 75) throw new RangeError('Angle must be an integer from 0 to 75 degrees')
  const radians = angle * Math.PI / 180
  const dx = 190 * Math.sin(radians)
  const dy = 190 * Math.cos(radians)
  const point = (x: number, y: number) => `${x.toFixed(3)},${y.toFixed(3)}`
  return {
    angle,
    surfaceAngle: 90 - angle,
    incoming: { x: 210 - dx, y: 250 - dy },
    outgoing: { x: 210 + dx, y: 250 - dy },
    incidentPath: `M${point(210 - dx, 250 - dy)} L210,250`,
    reflectedPath: `M210,250 L${point(210 + dx, 250 - dy)}`,
    incidentArrow: `M${point(210 - dx * .58, 250 - dy * .58)} L${point(210 - dx * .46, 250 - dy * .46)}`,
    reflectedArrow: `M${point(210 + dx * .66, 250 - dy * .66)} L${point(210 + dx * .78, 250 - dy * .78)}`,
    incidentArc: angle ? `M210,198 A52,52 0 0,0 ${point(210 - 52 * Math.sin(radians), 250 - 52 * Math.cos(radians))}` : '',
    reflectedArc: angle ? `M210,198 A52,52 0 0,1 ${point(210 + 52 * Math.sin(radians), 250 - 52 * Math.cos(radians))}` : '',
  }
}
