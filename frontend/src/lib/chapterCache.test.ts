import { describe, it, expect, vi } from 'vitest'
import { createChapterCache } from './chapterCache'
describe('chapter cache', () => {
  it('deduplicates concurrent reads and caches successful results', async () => {
    const cache = createChapterCache();const load = vi.fn(async () => [1]);
    expect(await Promise.all([cache('a','Science',load),cache('a','Science',load)])).toEqual([[1],[1]])
    await cache('a','Science',load);expect(load).toHaveBeenCalledTimes(1)
  })
  it('separates subjects and accounts and expires after TTL', async () => {
    let now=0;const cache=createChapterCache(10,()=>now);const load=vi.fn(async()=>[1]);
    await cache('a','Science',load);await cache('a','Maths',load);await cache('b','Science',load);now=11;await cache('b','Science',load);expect(load).toHaveBeenCalledTimes(4)
  })
  it('never caches failures', async () => {
    const cache=createChapterCache();const load=vi.fn().mockRejectedValueOnce(Error('network')).mockResolvedValue([1]);
    await expect(cache('a','Science',load)).rejects.toThrow('network');expect(await cache('a','Science',load)).toEqual([1]);expect(load).toHaveBeenCalledTimes(2)
  })
})
