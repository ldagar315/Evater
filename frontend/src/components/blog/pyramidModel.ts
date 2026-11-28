export function numberPyramid(bottom: readonly number[]) {
 if(bottom.length!==4 || !bottom.every(n=>Number.isInteger(n)&&n>=0&&n<=10)) throw new RangeError('Use four whole numbers from 0 to 10.')
 const rows=[Array.from(bottom)]
 while(rows[rows.length-1].length>1){const row=rows[rows.length-1];rows.push(row.slice(1).map((n,i)=>row[i]+n))}
 const weights=[1,3,3,1],contributions=bottom.map((n,i)=>n*weights[i])
 return {rows,weights,contributions,top:rows[3][0],bottomTotal:bottom.reduce((s,n)=>s+n,0)}
}
