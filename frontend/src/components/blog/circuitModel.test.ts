import {describe,expect,it} from 'vitest'
import {seriesCircuit,type CircuitState} from './circuitModel'
describe('one loop with two series switches and a return wire',()=>{
 const states=[false,true].flatMap(switchA=>[false,true].flatMap(switchB=>[false,true].map(wireConnected=>({switchA,switchB,wireConnected}))))
 it.each(states)('checks every state %j',s=>{const m=seriesCircuit(s);expect(m.complete).toBe(s.switchA&&s.switchB&&s.wireConnected);expect(m.lampOn).toBe(m.complete);expect(m.gaps).toHaveLength(Object.values(s).filter(v=>!v).length);expect(m.gaps.includes('Switch A')).toBe(!s.switchA);expect(m.gaps.includes('Switch B')).toBe(!s.switchB);expect(m.gaps.includes('Return wire')).toBe(!s.wireConnected)})
 it('each break alone interrupts the entire loop',()=>{for(const key of ['switchA','switchB','wireConnected'])expect(seriesCircuit({switchA:true,switchB:true,wireConnected:true,[key]:false}).lampOn).toBe(false)})
 it('closing one switch cannot repair another gap',()=>{expect(seriesCircuit({switchA:true,switchB:false,wireConnected:true}).lampOn).toBe(false);expect(seriesCircuit({switchA:true,switchB:true,wireConnected:false}).lampOn).toBe(false)})
 it('does not mutate or retain input',()=>{const s={switchA:true,switchB:true,wireConnected:true};const m=seriesCircuit(s);m.gaps.push('x');expect(s).toEqual({switchA:true,switchB:true,wireConnected:true});expect(seriesCircuit(s).gaps).toEqual([])})
 it.each([null,{}, {switchA:1,switchB:true,wireConnected:true},{switchA:true,switchB:'false',wireConnected:true},{switchA:true,switchB:true,wireConnected:undefined}])('rejects malformed state %j',s=>expect(()=>seriesCircuit(s as unknown as CircuitState)).toThrow(TypeError))
})
