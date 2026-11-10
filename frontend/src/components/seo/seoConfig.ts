export const SITE_URL = (import.meta.env.VITE_SITE_URL || 'https://evater.xyz').replace(/\/$/, '')

// Prerendered public pages are directory indexes on the current host. Match its
// served URL without changing app routes, asset paths, query strings or hashes.
export function publicPagePath(path: string): string {
  const normalized = path.startsWith('/') ? path : `/${path}`
  const match = normalized.match(/^([^?#]*)(.*)$/)!
  const pathname = match[1]
  return (pathname === '/about' || pathname === '/about/' || pathname === '/blog' || (/^\/blog\/[^/.]+\/?$/.test(pathname)))
    ? `${pathname.replace(/\/+$/, '')}/${match[2]}`
    : normalized
}
