import { useId, useMemo, useState } from 'react'

import { useRatingPairs } from '@/api/queries'
import type { EntityKind, RowValue } from '@/api/types'
import { ErrorState } from '@/components/common/ErrorState'
import { TeamChip } from '@/components/common/TeamChip'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { getEntityId, getEntityLabel } from '@/domain/entityConfig'
import { describeRatingPair, isMissingRatingPairs, neighborId, parseRatingPairs } from '@/domain/ratingPairs'

/** The published rating each kind is ranked by, best first. */
const RATING_COLUMN: Record<EntityKind, string> = { teams: 'team_rating', qbs: 'adj_qb_epa_per_dropback' }

/**
 * How often this team or QB was rated above another one across resampled seasons, with a picker
 * that starts on the neighbor in the published ranking. Hidden for seasons built without
 * head-to-head chances.
 */
export function HeadToHeadCard({
  kind,
  season,
  entityId,
  rows,
}: {
  kind: EntityKind
  season: number
  entityId: string
  rows: Record<string, RowValue>[]
}) {
  const query = useRatingPairs(kind, season, entityId)
  const [picked, setPicked] = useState<string | null>(null)
  const titleId = useId()
  const pairs = useMemo(() => (query.data ? parseRatingPairs(kind, query.data) : []), [kind, query.data])
  const labels = useMemo(
    () => new Map(rows.map((row) => [getEntityId(kind, row), getEntityLabel(kind, row)])),
    [kind, rows],
  )
  const ranked = useMemo(() => {
    const compared = new Set([entityId, ...pairs.map((pair) => pair.otherId)])
    const ratingColumn = RATING_COLUMN[kind]
    return rows
      .filter((row) => compared.has(getEntityId(kind, row)) && typeof row[ratingColumn] === 'number')
      .sort((left, right) => Number(right[ratingColumn]) - Number(left[ratingColumn]))
      .map((row) => getEntityId(kind, row))
  }, [entityId, kind, pairs, rows])

  if (query.isError) {
    return isMissingRatingPairs(query.error) ? null : (
      <ErrorState error={query.error} title="Could not load the head-to-head chances" />
    )
  }
  if (pairs.length === 0) return null

  const labelOf = (id: string) => labels.get(id) ?? id
  const selected = picked ?? neighborId(ranked, entityId) ?? pairs[0]?.otherId ?? null
  const pair = pairs.find((candidate) => candidate.otherId === selected)
  const options = [...pairs].sort((left, right) => labelOf(left.otherId).localeCompare(labelOf(right.otherId)))
  const unit = kind === 'teams' ? 'team' : 'QB'

  return (
    <Card role="region" aria-labelledby={titleId} className="gap-4">
      <CardHeader>
        <CardTitle id={titleId} className="text-base">
          Head to head
        </CardTitle>
        <CardDescription>
          How often this {unit} was rated above the one you pick across the same 1,000 redraws of
          the {season} season, counting only redraws with both. Both are re-rated in every redraw, so
          this answers the question more directly than two rank ranges.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <Select value={selected ?? undefined} onValueChange={setPicked}>
          <SelectTrigger size="sm" className="w-64 max-w-full" aria-label="Compare with">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {options.map((option) => (
              <SelectItem key={option.otherId} value={option.otherId}>
                {kind === 'teams' ? <TeamChip team={option.otherId} /> : null}
                {labelOf(option.otherId)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {pair ? <p className="text-sm">{describeRatingPair(kind, labelOf(entityId), labelOf(pair.otherId), pair)}</p> : null}
      </CardContent>
    </Card>
  )
}

/**
 * One sentence on how often `entityId` was rated above `otherId`, for the comparison panel when
 * exactly two rows are compared; nothing while loading or for seasons built without pairs.
 */
export function HeadToHeadSentence({
  kind,
  season,
  entityId,
  otherId,
  labels,
}: {
  kind: EntityKind
  season: number
  entityId: string
  otherId: string
  labels: { subject: string; other: string }
}) {
  const query = useRatingPairs(kind, season, entityId)
  const pair = query.data ? parseRatingPairs(kind, query.data).find((candidate) => candidate.otherId === otherId) : undefined
  if (!pair) return null
  return <p className="text-sm">{describeRatingPair(kind, labels.subject, labels.other, pair)}</p>
}
