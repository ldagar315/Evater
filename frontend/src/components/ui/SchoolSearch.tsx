import { useEffect, useId, useMemo, useState } from 'react'
import schools from '../../data/gurugramSchools.json'

export const normalizeSchoolSearch = (value: string) => value.toLowerCase().replace(/gurgaon/g, 'gurugram').replace(/[^a-z0-9]+/g, ' ').trim()
export function findSchools(query: string) {
  const words = normalizeSchoolSearch(query).split(' ').filter(Boolean)
  return schools.filter((school) => words.every((word) => normalizeSchoolSearch(`${school.name} ${school.address}`).includes(word))).slice(0, 20)
}

export function SchoolSearch({ value, onChange }: { value: string; onChange: (value: string) => void }) {
  const id = useId()
  const [query, setQuery] = useState(value)
  const [custom, setCustom] = useState(Boolean(value && !schools.some((school) => school.label === value)))
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(-1)
  const results = useMemo(() => findSchools(query), [query])
  useEffect(() => {
    if (!value) return
    setQuery(value)
    setCustom(!schools.some((school) => school.label === value))
  }, [value])
  useEffect(() => {
    if (open && active >= 0) document.getElementById(`${id}-option-${active}`)?.scrollIntoView({ block: 'nearest' })
  }, [active, open, id])
  const choose = (index: number) => {
    if (index === results.length) {
      setCustom(true)
      onChange(query.trim())
    } else {
      setQuery(results[index].label)
      onChange(results[index].label)
    }
    setOpen(false)
    setActive(-1)
  }
  const inputClass = 'mt-1 min-h-12 w-full rounded-lg border border-neutral-300 bg-white px-3 py-2 text-sm text-dark focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500/20'
  return <div className="relative">
    <label htmlFor={id} className="block text-sm font-medium text-dark">School</label>
    {custom ? <>
      <input id={id} className={inputClass} value={value} required maxLength={300} autoComplete="organization" onChange={(event) => onChange(event.target.value)} placeholder="Enter your school's name and campus" aria-describedby={`${id}-help`} />
      <button type="button" className="mt-2 min-h-11 text-sm font-semibold text-primary-700 underline" onClick={() => {setCustom(false);setQuery('');setOpen(true);onChange('')}}>Search the Gurugram directory instead</button>
    </> : <>
      <input id={id} role="combobox" aria-autocomplete="list" aria-expanded={open} aria-controls={`${id}-list`} aria-activedescendant={open && active >= 0 ? `${id}-option-${active}` : undefined} aria-describedby={`${id}-help`} className={inputClass} value={query} autoComplete="off" placeholder="Search school name, sector or village" onFocus={() => setOpen(true)} onBlur={() => {setOpen(false);setActive(-1)}} onChange={(event) => {setQuery(event.target.value);onChange('');setActive(-1);setOpen(true)}} onKeyDown={(event) => {
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {event.preventDefault();setOpen(true);setActive((current) => event.key === 'ArrowDown' ? Math.min(current + 1, results.length) : Math.max(current - 1, 0))}
        else if (event.key === 'Enter' && open && active >= 0) {event.preventDefault();choose(active)}
        else if (event.key === 'Escape') {setOpen(false);setActive(-1)}
      }} />
      {open && <ul id={`${id}-list`} role="listbox" aria-label="School suggestions" className="absolute z-30 mt-1 max-h-72 w-full overflow-y-auto overscroll-contain rounded-xl border border-neutral-200 bg-white shadow-xl">
        {results.map((school, index) => <li id={`${id}-option-${index}`} key={school.id} role="option" aria-selected={active === index} onMouseDown={(event) => event.preventDefault()} onClick={() => choose(index)} className={`cursor-pointer px-3 py-3 text-sm ${active === index ? 'bg-primary-50' : 'hover:bg-neutral-50'}`}><span className="block font-semibold">{school.name}</span><span className="block break-words text-xs text-neutral-500">{school.address}</span></li>)}
        <li id={`${id}-option-${results.length}`} role="option" aria-selected={active === results.length} onMouseDown={(event) => event.preventDefault()} onClick={() => choose(results.length)} className={`cursor-pointer border-t px-3 py-3 text-sm font-semibold text-primary-700 ${active === results.length ? 'bg-primary-50' : ''}`}>Other / school not listed - enter a name</li>
      </ul>}
    </>}
    <p id={`${id}-help`} className="mt-2 text-xs leading-5 text-neutral-500">Suggestions from CBSE and Gurugram district directories (9 Nov 2026). Not every school is listed. Choose a campus, or use Other to keep your own school name.</p>
  </div>
}
