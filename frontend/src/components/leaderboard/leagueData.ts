export type LeagueFamily = 'bronze' | 'silver' | 'gold' | 'platinum' | 'diamond'

export type LeagueStage = {
  tier: string
  label: string
  family: LeagueFamily
  stage: string
  threshold: number | null
}

export const LEAGUE_LADDER: LeagueStage[] = [
  { tier: 'bronze_3', label: 'Bronze III', family: 'bronze', stage: 'III', threshold: 100 },
  { tier: 'bronze_2', label: 'Bronze II', family: 'bronze', stage: 'II', threshold: 150 },
  { tier: 'bronze_1', label: 'Bronze I', family: 'bronze', stage: 'I', threshold: 200 },
  { tier: 'silver_3', label: 'Silver III', family: 'silver', stage: 'III', threshold: 275 },
  { tier: 'silver_2', label: 'Silver II', family: 'silver', stage: 'II', threshold: 350 },
  { tier: 'silver_1', label: 'Silver I', family: 'silver', stage: 'I', threshold: 450 },
  { tier: 'gold_3', label: 'Gold III', family: 'gold', stage: 'III', threshold: 575 },
  { tier: 'gold_2', label: 'Gold II', family: 'gold', stage: 'II', threshold: 725 },
  { tier: 'gold_1', label: 'Gold I', family: 'gold', stage: 'I', threshold: 900 },
  { tier: 'platinum_3', label: 'Platinum III', family: 'platinum', stage: 'III', threshold: 1100 },
  { tier: 'platinum_2', label: 'Platinum II', family: 'platinum', stage: 'II', threshold: 1350 },
  { tier: 'platinum_1', label: 'Platinum I', family: 'platinum', stage: 'I', threshold: 1650 },
  { tier: 'diamond_3', label: 'Diamond III', family: 'diamond', stage: 'III', threshold: 2000 },
  { tier: 'diamond_2', label: 'Diamond II', family: 'diamond', stage: 'II', threshold: 2400 },
  { tier: 'diamond_1', label: 'Diamond I', family: 'diamond', stage: 'I', threshold: null },
]

export const SCORING_ROWS = [
  { difficulty: 'Easy', value: '+4', wrong: '−1', skipped: '−2', dotColor: 'bg-orange-400' },
  { difficulty: 'Medium', value: '+8', wrong: '−2', skipped: '−4', dotColor: 'bg-yellow-400' },
  { difficulty: 'Hard', value: '+12', wrong: '−3', skipped: '−6', dotColor: 'bg-indigo-500' },
]

export function stageForTier(tier?: string | null) {
  return LEAGUE_LADDER.find((league) => league.tier === tier) || LEAGUE_LADDER[0]
}
