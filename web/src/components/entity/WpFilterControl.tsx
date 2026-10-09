import { Hourglass } from 'lucide-react'
import { useEffect, useState } from 'react'

import { useWpRatings } from '@/api/queries'
import type { EntityKind } from '@/api/types'
import { useWpThreshold } from '@/app/useWpThreshold'
import { Button } from '@/components/ui/button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Slider } from '@/components/ui/slider'
import { describeWpThreshold, MAX_WP_THRESHOLD } from '@/domain/wpFilter'
import { useDebouncedValue } from '@/hooks/use-debounced-value'

// Wait this long after the slider stops before putting the threshold in the address.
const WP_DEBOUNCE_MS = 250

/**
 * The garbage-time filter as one compact button: it names the threshold (`?wp=`), and its popover
 * holds the 0-20% slider and what the filter does. The filtered values themselves show where the
 * page puts them (beside the rating in the index table, under the rating summary on a detail page).
 */
export function WpFilterControl({ kind, season, where }: { kind: EntityKind; season: number; where: string }) {
  const [threshold, setThreshold] = useWpThreshold()
  const [draft, setDraft] = useState(threshold)
  const [synced, setSynced] = useState(threshold)
  // Follow a threshold that changes from outside (navigation, a shared link, the off button).
  if (threshold !== synced) {
    setSynced(threshold)
    setDraft(threshold)
  }
  const settled = useDebouncedValue(draft, WP_DEBOUNCE_MS)
  useEffect(() => {
    // Commit only a value the user has stopped moving, never a stale one after a navigation.
    if (settled === draft && settled !== threshold) setThreshold(settled)
  }, [draft, settled, setThreshold, threshold])
  const query = useWpRatings(kind, season, threshold)

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button size="sm" variant={threshold > 0 ? 'secondary' : 'outline'}>
          <Hourglass aria-hidden />
          Garbage time: {threshold > 0 ? `${threshold}%` : 'off'}
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" className="flex w-80 flex-col gap-3 text-sm">
        <div className="flex items-baseline justify-between gap-2">
          <span className="font-semibold">Garbage-time filter</span>
          <span className="tabular font-medium" aria-hidden>
            {draft === 0 ? 'Off' : `${draft}%`}
          </span>
        </div>
        <Slider
          aria-label="Garbage-time filter"
          className="py-2"
          min={0}
          max={MAX_WP_THRESHOLD}
          step={1}
          value={[draft]}
          onValueChange={([value]) => setDraft(value ?? 0)}
        />
        <p>{describeWpThreshold(draft)}</p>
        <p className="text-muted-foreground">
          The filtered rating and rank show {where}.
          Exploration only: everything else uses every play, and when filters of 5%, 10%, and 20%
          were tested by predicting each game&apos;s margin from earlier games (1999-2025), none beat
          using every play.
          {kind === 'qbs'
            ? ' Filtered QB ratings use play-by-play EPA, a hair off the official EPA behind the published rating.'
            : null}
        </p>
        {query.isError ? <p className="text-destructive">Could not load the filtered ratings.</p> : null}
        {threshold > 0 ? (
          <Button size="sm" variant="ghost" className="self-start" onClick={() => setThreshold(0)}>
            Count every play
          </Button>
        ) : null}
      </PopoverContent>
    </Popover>
  )
}
