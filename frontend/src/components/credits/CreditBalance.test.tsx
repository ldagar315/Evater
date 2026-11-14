import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { CreditBalance, creditLabel } from './CreditBalance'
describe('credit presentation only', () => {
  it('formats zero as a real balance and missing as unavailable', () => {
    expect(creditLabel(0)).toBe('0 credits')
    expect(creditLabel(null)).toBe('Balance unavailable')
    expect(creditLabel(undefined)).toBe('Balance unavailable')
    expect(creditLabel(123456789)).toBe('12,34,56,789 credits')
  })
  for(const variant of ['compact','summary','panel'] as const) it(`renders ${variant} without inventing transaction controls`,()=>{
    const html=renderToStaticMarkup(<MemoryRouter><CreditBalance balance={125} variant={variant} /></MemoryRouter>)
    expect(html).toContain('125')
    expect(html).toContain('Credits')
    expect(html).not.toContain('E-coins')
    expect(html).not.toMatch(/Buy credits|Top up|Redeem|Earn credits/)
    expect(html).toContain(variant==='panel'?'/practice':'/profile#credits')
  })
})
