import { BookOpen, Shield, UserRound, type LucideIcon } from 'lucide-react'

export interface NavItem {
  to: string
  label: string
  icon: LucideIcon
  description: string
}

/** Sidebar navigation, in display order. */
export const NAV_ITEMS: NavItem[] = [
  { to: '/teams', label: 'Teams', icon: Shield, description: 'Schedule-adjusted team ratings and stats' },
  { to: '/qbs', label: 'Quarterbacks', icon: UserRound, description: 'Schedule-adjusted QB ratings and stats' },
  { to: '/glossary', label: 'Glossary', icon: BookOpen, description: 'What every rating and column means' },
]
