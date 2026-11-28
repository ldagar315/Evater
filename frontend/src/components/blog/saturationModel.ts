export function saturationModel(solvent: number, added: number) {
  if (!Number.isInteger(solvent) || solvent < 1 || solvent > 4 || !Number.isInteger(added) || added < 0 || added > 20) throw new RangeError('Use 1–4 solvent units and 0–20 solute units.')
  const limit = solvent * 4
  const dissolved = Math.min(added, limit)
  const undissolved = added - dissolved
  return { solvent, added, limit, dissolved, undissolved, room: limit - dissolved, saturated: dissolved === limit }
}
