export function distributiveArea(x: number, a: number, b: number) {
  if (!Number.isInteger(x) || x < 2 || x > 10 || [a,b].some(n=>!Number.isInteger(n)||n<1||n>6)) throw new RangeError('Use x from 2 to 10 and a,b from 1 to 6.')
  const parts = [x*x,a*x,b*x,a*b]
  return { x,a,b,width:x+a,height:x+b,parts,total:(x+a)*(x+b),middle:a+b,missing:(a+b)*x,square:a===b }
}
