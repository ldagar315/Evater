export type Point = { x: number; y: number }
export function angleAt(a: Point, b: Point, c: Point) {
  const ux=a.x-b.x,uy=a.y-b.y,vx=c.x-b.x,vy=c.y-b.y
  return Math.atan2(Math.abs(ux*vy-uy*vx),ux*vx+uy*vy)*180/Math.PI
}
export function quadrilateralAngles(left: number, right: number) {
  if (![left,right].every(n=>Number.isInteger(n)&&n>=40&&n<=120&&n%10===0)) throw new RangeError('Use top insets 40 to 120 in steps of 10.')
  const A={x:25,y:220},B={x:325,y:220},C={x:350-right,y:40},D={x:left,y:40}
  const points=[A,B,C,D],angles=points.map((p,i)=>angleAt(points[(i+3)%4],p,points[(i+1)%4]))
  const first=[angleAt(B,A,C),angleAt(A,B,C),angleAt(A,C,B)]
  const second=[angleAt(C,A,D),angleAt(A,C,D),angleAt(A,D,C)]
  return {points,angles,first,second,total:angles.reduce((s,n)=>s+n,0)}
}
