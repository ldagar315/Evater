export function cubeModel(side: number, layers: number, angle: number) {
  const n = Math.max(1, Math.min(6, Math.round(side)))
  const visibleLayers = Math.max(1, Math.min(n, Math.round(layers)))
  const radians = Math.max(20, Math.min(70, angle)) * Math.PI / 180
  const project = (x: number, y: number, z: number) => ({
    x: 170 + (x - n / 2) * Math.cos(radians) * 28 - (z - n / 2) * Math.sin(radians) * 28,
    y: 245 - y * 28 + ((x - n / 2) * Math.sin(radians) + (z - n / 2) * Math.cos(radians)) * 12,
  })
  const units = []
  // Draw farthest units first. All lengths are in unit-cube coordinates.
  for (let y = 0; y < visibleLayers; y++) {
    for (let z = 0; z < n; z++) {
      for (let x = 0; x < n; x++) {
        const face = (points: number[][]) => points.map(([a, b, c]) => {
          const p = project(a, b, c)
          return `${p.x},${p.y}`
        }).join(' ')
        units.push({
          key: `${x}-${y}-${z}`,
          depth: x + z + y * n * 2,
          top: face([[x, y + 1, z], [x + 1, y + 1, z], [x + 1, y + 1, z + 1], [x, y + 1, z + 1]]),
          front: face([[x, y, z + 1], [x + 1, y, z + 1], [x + 1, y + 1, z + 1], [x, y + 1, z + 1]]),
          right: face([[x + 1, y, z], [x + 1, y, z + 1], [x + 1, y + 1, z + 1], [x + 1, y + 1, z]]),
        })
      }
    }
  }
  return { side: n, layers: visibleLayers, baseArea: n * n, fullVolume: n * n * n, shownVolume: n * n * visibleLayers, units }
}
