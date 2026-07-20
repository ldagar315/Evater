import React, { useId } from 'react'

type LeagueFamily = 'bronze' | 'silver' | 'gold' | 'platinum' | 'diamond'

type LeagueEmblemProps = {
  tier: string
  size?: 'sm' | 'md' | 'lg' | 'xl'
  className?: string
}

const FAMILY_THEME: Record<LeagueFamily, { dark: string; light: string; accent: string }> = {
  bronze: { dark: '#9a4d24', light: '#f2b66d', accent: '#ffd49c' },
  silver: { dark: '#53616c', light: '#c5d0d6', accent: '#f4f7f8' },
  gold: { dark: '#9b6a0b', light: '#f4c44f', accent: '#fff1a8' },
  platinum: { dark: '#14777d', light: '#54c6c0', accent: '#c7fff3' },
  diamond: { dark: '#354aa0', light: '#86a8ff', accent: '#dce6ff' },
}

const SIZE_MAP = {
  sm: { className: 'h-10 w-10', textSize: 9 },
  md: { className: 'h-14 w-14', textSize: 11 },
  lg: { className: 'h-24 w-24', textSize: 15 },
  xl: { className: 'h-32 w-32', textSize: 18 },
}

function familyForTier(tier: string): LeagueFamily {
  const family = tier.split('_')[0] as LeagueFamily
  return FAMILY_THEME[family] ? family : 'bronze'
}

function stageForTier(tier: string) {
  const stage = tier.split('_')[1]
  if (stage === '1') return 'I'
  if (stage === '2') return 'II'
  return 'III'
}

export function LeagueEmblem({ tier, size = 'md', className = '' }: LeagueEmblemProps) {
  const id = useId().replace(/:/g, '')
  const family = familyForTier(tier)
  const theme = FAMILY_THEME[family]
  const stage = stageForTier(tier)
  const sizeConfig = SIZE_MAP[size]
  const gradientId = `league-${id}-${family}-${stage.toLowerCase()}-gradient`
  const label = `${family[0].toUpperCase()}${family.slice(1)} ${stage}`

  return (
    <svg
      className={`${sizeConfig.className} shrink-0 ${className}`}
      viewBox="0 0 100 112"
      role="img"
      aria-label={`${label} league emblem`}
    >
      <defs>
        <linearGradient id={gradientId} x1="18" y1="10" x2="82" y2="100" gradientUnits="userSpaceOnUse">
          <stop stopColor={theme.light} />
          <stop offset="1" stopColor={theme.dark} />
        </linearGradient>
      </defs>
      <path d="M50 4 91 18v37c0 24-17 43-41 53C26 98 9 79 9 55V18L50 4Z" fill="#fff" opacity=".9" />
      <path d="M50 9 86 21v33c0 21-14 37-36 47C28 91 14 75 14 54V21L50 9Z" fill={`url(#${gradientId})`} />
      <path d="m50 20 8 16 18 3-13 13 3 18-16-8-16 8 3-18-13-13 18-3 8-16Z" fill={theme.accent} opacity=".96" />
      <path d="M22 27v27c0 16 10 29 28 38" fill="none" stroke="#fff" strokeLinecap="round" strokeWidth="3" opacity=".35" />
      <text x="50" y="62" fill={theme.dark} fontFamily="Plus Jakarta Sans, sans-serif" fontSize={sizeConfig.textSize} fontWeight="800" textAnchor="middle">
        {stage}
      </text>
      <circle cx="50" cy="90" r="5" fill={theme.accent} stroke="#fff" strokeWidth="2" />
    </svg>
  )
}
