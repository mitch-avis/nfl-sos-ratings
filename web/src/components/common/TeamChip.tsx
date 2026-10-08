import { teamColors } from '@/domain/teamPalettes'
import { cn } from '@/utils/cn'

/**
 * A small two-color dot in a team's main colors, set beside its abbreviation or name. It is
 * decorative (the text beside it names the team), so screen readers skip it; the ring keeps a
 * black or white half visible on either theme.
 */
export function TeamChip({ team, className }: { team: string; className?: string }) {
  const colors = teamColors(team)
  if (!colors) return null
  return (
    <span
      aria-hidden="true"
      data-team-chip={team}
      className={cn('inline-block size-2.5 shrink-0 rounded-full ring-1 ring-foreground/20', className)}
      style={{ background: `linear-gradient(135deg, ${colors[0]} 50%, ${colors[1]} 50%)` }}
    />
  )
}
