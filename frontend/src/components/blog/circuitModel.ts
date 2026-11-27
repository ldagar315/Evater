export type CircuitState={switchA:boolean,switchB:boolean,wireConnected:boolean}
export function seriesCircuit(s: CircuitState){
 if(!s||[s.switchA,s.switchB,s.wireConnected].some(v=>typeof v!=='boolean'))throw new TypeError('Circuit controls must be true or false.')
 const gaps=[...(!s.switchA?['Switch A']:[]),...(!s.switchB?['Switch B']:[]),...(!s.wireConnected?['Return wire']:[])]
 return {complete:gaps.length===0,lampOn:gaps.length===0,gaps}
}
