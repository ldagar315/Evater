import React, { useEffect, useRef, useState } from 'react'
import {
  ChevronDown,
  Coins,
  GraduationCap,
  LogOut,
  Menu,
  School,
  User,
  X,
} from 'lucide-react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { Button } from '../ui/Button'
import { useAuthContext } from '../../contexts/AuthContext'
import { useProfile } from '../../hooks/useProfile'

const memberLinks = [
  { label: 'Practice', to: '/practice' },
  { label: 'Leaderboard', to: '/leaderboard' },
  { label: 'Profile', to: '/profile' },
]

function isActivePath(pathname: string, path: string) {
  return pathname === path || pathname.startsWith(`${path}/`)
}

export function Header() {
  const navigate = useNavigate()
  const location = useLocation()
  const { user, signOut } = useAuthContext()
  const { profile } = useProfile(user?.id)
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)
  const isBlogPage = location.pathname.startsWith('/blog')
  const displayName = profile?.name || profile?.user_name || 'User'
  const initials = displayName.trim().charAt(0).toUpperCase() || 'U'

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setDropdownOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  useEffect(() => {
    setMobileMenuOpen(false)
    setDropdownOpen(false)
  }, [location.pathname])

  const handleSignOut = async () => {
    await signOut()
    navigate('/auth')
  }

  const goTo = (path: string) => {
    navigate(path)
    setMobileMenuOpen(false)
    setDropdownOpen(false)
  }

  const closeMenus = () => {
    setMobileMenuOpen(false)
    setDropdownOpen(false)
  }

  const renderMemberLinks = (mobile = false) => (
    <nav aria-label="Learning navigation" className={mobile ? 'space-y-1' : 'flex items-center gap-1'}>
      {memberLinks.map((link) => {
        const active = isActivePath(location.pathname, link.to)
        return (
          <Link
            key={link.to}
            to={link.to}
            onClick={closeMenus}
            className={mobile
              ? `flex min-h-11 w-full items-center rounded-xl px-3 text-sm font-semibold transition-colors ${active ? 'bg-primary-50 text-primary-700' : 'text-neutral-600 hover:bg-white hover:text-dark'}`
              : `min-h-10 rounded-xl px-3 text-sm font-bold transition-colors ${active ? 'bg-primary-50 text-primary-700' : 'text-neutral-600 hover:bg-neutral-50 hover:text-dark'}`}
          >
            {link.label}
          </Link>
        )
      })}
    </nav>
  )

  return (
    <header className="sticky top-0 z-50 w-full border-b border-neutral-200 bg-cream/95 backdrop-blur-sm">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex min-h-16 items-center justify-between gap-4 sm:min-h-[4.5rem]">
          <Link
            to={user ? '/home' : '/'}
            onClick={closeMenus}
            className="group flex shrink-0 items-center rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
            aria-label={user ? 'Go to home' : 'Go to landing page'}
          >
            <img
              src="/Evater_logo_2.png"
              alt="Evater"
              className="h-10 w-auto max-w-[9.5rem] object-contain transition-transform group-hover:scale-[1.02] sm:h-12 sm:max-w-[11rem]"
            />
          </Link>

          <div className="hidden items-center gap-5 md:flex">
            {user && renderMemberLinks()}
            <div className="hidden h-6 w-px bg-neutral-200 lg:block" aria-hidden="true" />
            <nav aria-label="Information navigation" className="hidden items-center gap-1 lg:flex">
              <Link
                to="/blog"
                onClick={closeMenus}
                className={`min-h-10 rounded-xl px-3 text-sm font-semibold transition-colors ${isBlogPage ? 'bg-primary-50 text-primary-700' : 'text-neutral-500 hover:bg-neutral-50 hover:text-dark'}`}
              >
                Blog
              </Link>
              <Link
                to="/about"
                onClick={closeMenus}
                className="min-h-10 rounded-xl px-3 text-sm font-semibold text-neutral-500 transition-colors hover:bg-neutral-50 hover:text-dark"
              >
                About
              </Link>
            </nav>
          </div>

          <div className="hidden items-center gap-3 md:flex">
            {user ? (
              <>
                <div className="hidden min-h-10 items-center gap-2 rounded-xl border border-secondary-200 bg-secondary-50 px-3 text-sm font-bold text-secondary-800 lg:inline-flex">
                  <Coins className="h-4 w-4" aria-hidden="true" />
                  <span>{profile?.credits ?? '—'}</span>
                  <span className="hidden text-xs font-semibold text-secondary-700 lg:inline">credits</span>
                </div>
                <div className="relative" ref={dropdownRef}>
                  <button
                    type="button"
                    onClick={() => setDropdownOpen((open) => !open)}
                    aria-expanded={dropdownOpen}
                    aria-haspopup="menu"
                    className={`flex min-h-11 items-center gap-2 rounded-xl border px-2.5 pr-3 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 ${dropdownOpen ? 'border-primary-200 bg-primary-50' : 'border-neutral-200 bg-white hover:border-primary-200 hover:bg-primary-50/50'}`}
                  >
                    <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary-100 text-sm font-bold text-primary-700">
                      {initials}
                    </span>
                    <span className="hidden max-w-28 truncate text-sm font-bold text-dark lg:inline">{displayName}</span>
                    <ChevronDown className={`h-4 w-4 text-neutral-400 transition-transform ${dropdownOpen ? 'rotate-180' : ''}`} aria-hidden="true" />
                  </button>

                  {dropdownOpen && (
                    <div className="absolute right-0 mt-3 w-80 overflow-hidden rounded-2xl border border-neutral-200 bg-white shadow-xl shadow-dark/10" role="menu">
                      <div className="border-b border-neutral-100 bg-neutral-50/70 p-4">
                        <div className="flex items-center gap-3">
                          <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-primary-100 text-base font-bold text-primary-700">{initials}</span>
                          <div className="min-w-0">
                            <p className="truncate text-sm font-bold text-dark">{displayName}</p>
                            <p className="truncate text-xs text-neutral-500">{user.email}</p>
                          </div>
                        </div>
                      </div>
                      <div className="space-y-1 p-3">
                        <div className="grid grid-cols-2 gap-2 px-1 pb-2">
                          <div className="rounded-xl border border-neutral-100 bg-neutral-50 p-3">
                            <p className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-[0.08em] text-neutral-500">
                              <GraduationCap className="h-3.5 w-3.5" aria-hidden="true" /> Grade
                            </p>
                            <p className="mt-1 text-sm font-bold text-dark">{profile?.grade ? `Class ${profile.grade}` : 'Not set'}</p>
                          </div>
                          <div className="rounded-xl border border-neutral-100 bg-neutral-50 p-3">
                            <p className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-[0.08em] text-neutral-500">
                              <School className="h-3.5 w-3.5" aria-hidden="true" /> School
                            </p>
                            <p className="mt-1 truncate text-sm font-bold text-dark">{profile?.school || 'Not set'}</p>
                          </div>
                        </div>
                        <button
                          type="button"
                          onClick={() => goTo('/profile')}
                          className="flex min-h-11 w-full items-center rounded-xl px-3 text-sm font-semibold text-neutral-600 transition-colors hover:bg-neutral-50 hover:text-dark"
                          role="menuitem"
                        >
                          <User className="mr-2 h-4 w-4" aria-hidden="true" /> Edit profile
                        </button>
                        <button
                          type="button"
                          onClick={handleSignOut}
                          className="flex min-h-11 w-full items-center rounded-xl px-3 text-sm font-semibold text-red-600 transition-colors hover:bg-red-50"
                          role="menuitem"
                        >
                          <LogOut className="mr-2 h-4 w-4" aria-hidden="true" /> Sign out
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="flex items-center gap-2">
                <Button onClick={() => goTo('/auth')} variant="ghost" size="sm" className="text-neutral-600 hover:text-dark">
                  Log in
                </Button>
                <Button onClick={() => goTo('/auth')} variant="primary" size="sm">
                  Sign up
                </Button>
              </div>
            )}
          </div>

          <button
            type="button"
            onClick={() => setMobileMenuOpen((open) => !open)}
            className="flex min-h-11 min-w-11 items-center justify-center rounded-xl border border-neutral-200 bg-white text-neutral-600 transition-colors hover:border-primary-200 hover:bg-primary-50 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 md:hidden"
            aria-expanded={mobileMenuOpen}
            aria-label={mobileMenuOpen ? 'Close navigation menu' : 'Open navigation menu'}
          >
            {mobileMenuOpen ? <X className="h-5 w-5" aria-hidden="true" /> : <Menu className="h-5 w-5" aria-hidden="true" />}
          </button>
        </div>
      </div>

      {mobileMenuOpen && (
        <div className="absolute right-4 top-full z-50 max-h-[calc(100vh-5rem)] w-80 max-w-[calc(100vw-2rem)] overflow-y-auto rounded-2xl border border-neutral-200 bg-cream shadow-xl shadow-dark/10 md:hidden">
          <div className="space-y-5 p-5">
            {user && (
              <div className="flex items-center justify-between rounded-2xl border border-neutral-200 bg-white p-4">
                <div className="flex min-w-0 items-center gap-3">
                  <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary-100 text-sm font-bold text-primary-700">{initials}</span>
                  <div className="min-w-0">
                    <p className="truncate text-sm font-bold text-dark">{displayName}</p>
                    <p className="truncate text-xs text-neutral-500">{profile?.school || 'Evater learner'}</p>
                  </div>
                </div>
                <div className="inline-flex items-center gap-1.5 rounded-lg bg-secondary-50 px-2.5 py-2 text-sm font-bold text-secondary-800">
                  <Coins className="h-4 w-4" aria-hidden="true" /> {profile?.credits ?? '—'}
                </div>
              </div>
            )}

            {user && renderMemberLinks(true)}

            <nav aria-label="Information navigation" className="space-y-1 border-t border-neutral-200 pt-4">
              <Link
                to="/blog"
                onClick={closeMenus}
                className={`flex min-h-11 w-full items-center rounded-xl px-3 text-sm font-semibold transition-colors ${isBlogPage ? 'bg-primary-50 text-primary-700' : 'text-neutral-600 hover:bg-white hover:text-dark'}`}
              >
                Blog
              </Link>
              <Link
                to="/about"
                onClick={closeMenus}
                className="flex min-h-11 w-full items-center rounded-xl px-3 text-sm font-semibold text-neutral-600 transition-colors hover:bg-white hover:text-dark"
              >
                About
              </Link>
            </nav>

            {user ? (
              <div className="grid grid-cols-2 gap-2 border-t border-neutral-200 pt-4">
                <Button onClick={() => goTo('/profile')} variant="outline" className="w-full">
                  Edit profile
                </Button>
                <Button onClick={handleSignOut} variant="ghost" className="w-full text-red-600 hover:bg-red-50 hover:text-red-700">
                  <LogOut className="mr-2 h-4 w-4" aria-hidden="true" /> Sign out
                </Button>
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-2 border-t border-neutral-200 pt-4">
                <Button onClick={() => goTo('/auth')} variant="outline" className="w-full">Log in</Button>
                <Button onClick={() => goTo('/auth')} variant="primary" className="w-full">Sign up</Button>
              </div>
            )}
          </div>
        </div>
      )}
    </header>
  )
}
