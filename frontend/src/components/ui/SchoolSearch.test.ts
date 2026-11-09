import { describe, it, expect } from 'vitest'
import { findSchools, normalizeSchoolSearch } from './SchoolSearch'
import schools from '../../data/gurugramSchools.json'
describe('sourced school search',()=>{
 it('keeps campuses separate and supports multiword sector queries',()=>{const result=findSchools('euro 45');expect(result.some(s=>s.id==='cbse-531398')).toBe(true);expect(findSchools('euro').length).toBeGreaterThan(1)})
 it('treats Gurgaon as Gurugram without changing stored names',()=>expect(normalizeSchoolSearch('Gurgaon')).toBe(normalizeSchoolSearch('Gurugram')))
 it('offers no invented matches and limits visible results',()=>{expect(findSchools('no-such-school-xyz')).toEqual([]);expect(findSchools('').length).toBe(20)})
 it('all entries have unique identifiers and first-party source URLs',()=>{expect(new Set(schools.map(s=>s.id)).size).toBe(schools.length);expect(schools.every(s=>/^https:\/\/(saras.cbse.gov.in|gurugram.gov.in)\//.test(s.source))).toBe(true)})
})
