export type Point = readonly [number, number]
export type Triangle = readonly [Point, Point, Point]
const middle = (a: Point, b: Point): Point => [(a[0]+b[0])/2,(a[1]+b[1])/2]
export function triangleArea(t: Triangle) {
 const [a,b,c]=t
 return Math.abs((b[0]-a[0])*(c[1]-a[1])-(c[0]-a[0])*(b[1]-a[1]))/2
}
export function sierpinskiStage(stage: number) {
 if(!Number.isInteger(stage)||stage<0||stage>5) throw new RangeError('Use a whole stage from 0 to 5.')
 let triangles: Triangle[]=[[[16,264],[304,264],[160,14.58468371008169]]]
 for(let n=0;n<stage;n++) triangles=triangles.flatMap(([a,b,c]):Triangle[]=>{const ab=middle(a,b),bc=middle(b,c),ca=middle(c,a);return [[a,ab,ca],[ab,b,bc],[ca,bc,c]]})
 const count=3**stage,denominator=4**stage
 return {stage,triangles,count,areaNumerator:count,areaDenominator:denominator,areaFraction:count/denominator,sideDenominator:2**stage}
}
