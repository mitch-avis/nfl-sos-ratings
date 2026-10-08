import { useQueryClient } from '@tanstack/react-query'
import { RefreshCw } from 'lucide-react'
import { useEffect, useRef } from 'react'

import { isDataQuery, useRefreshStatus, useStartRefresh } from '@/api/queries'
import type { RefreshState } from '@/api/types'
import { Button } from '@/components/ui/button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { FAILED_LOG_LINES, refreshFinished, refreshMessage } from '@/domain/refresh'
import { cn } from '@/utils/cn'

/** A local time such as "2:05 PM" for an ISO time from the server. */
function localClock(iso: string): string {
  return new Date(iso).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
}

/** Refetches every page's data when a refresh that was running ends, since `data/` changed. */
function useRefetchAfterRefresh(state: RefreshState | undefined) {
  const queryClient = useQueryClient()
  const previous = useRef(state)
  useEffect(() => {
    if (refreshFinished(previous.current, state)) {
      void queryClient.invalidateQueries({ predicate: (query) => isDataQuery(query.queryKey) })
    }
    previous.current = state
  }, [queryClient, state])
}

/** Opens the failure output at its end, where the step that failed says so. */
function scrollToEnd(element: HTMLPreElement | null) {
  if (element) element.scrollTop = element.scrollHeight
}

function buttonLabel(state: RefreshState): string {
  if (state === 'running') return 'Refreshing data'
  if (state === 'failed') return 'Refresh data (last run failed)'
  return 'Refresh data'
}

/**
 * The header's refresh button, shown only when the server runs with `--allow-refresh`. Its panel
 * says what a refresh does before starting one, then follows the run; the icon spins meanwhile.
 */
export function RefreshControl() {
  const status = useRefreshStatus()
  const start = useStartRefresh()
  useRefetchAfterRefresh(status.data?.state)
  if (!status.data?.allowed) return null
  const { state, log_tail: logTail } = status.data
  const running = state === 'running'
  const label = buttonLabel(state)
  const lastLine = logTail.at(-1)
  return (
    <Popover onOpenChange={(open) => open && void status.refetch()}>
      <Tooltip>
        <TooltipTrigger asChild>
          <PopoverTrigger asChild>
            <Button variant="ghost" size="icon" aria-label={label}>
              <RefreshCw
                className={cn(
                  'size-4',
                  running && 'animate-spin motion-reduce:animate-none',
                  state === 'failed' && 'text-destructive',
                )}
              />
            </Button>
          </PopoverTrigger>
        </TooltipTrigger>
        <TooltipContent>{label}</TooltipContent>
      </Tooltip>
      <PopoverContent align="end" className="w-80 space-y-3 text-sm">
        <div className="space-y-1">
          <p className="font-medium">Refresh the season in progress</p>
          <p className="text-muted-foreground">
            Downloads the latest nflverse data, rebuilds the season in progress, checks the result,
            and lists which data files changed. It takes a few minutes; every page updates when it
            finishes.
          </p>
        </div>
        <div role="status" className="space-y-1">
          <p>{refreshMessage(status.data, localClock)}</p>
          {running && lastLine && (
            <p className="line-clamp-2 font-mono text-xs wrap-anywhere text-muted-foreground">{lastLine}</p>
          )}
        </div>
        {state === 'failed' && (
          <pre
            ref={scrollToEnd}
            className="max-h-48 overflow-auto rounded-md bg-muted p-2 font-mono text-xs whitespace-pre-wrap wrap-anywhere"
          >
            {logTail.slice(-FAILED_LOG_LINES).join('\n')}
          </pre>
        )}
        {start.error && (
          <p role="alert" className="text-destructive">
            {start.error.message}
          </p>
        )}
        <Button size="sm" disabled={running || start.isPending} onClick={() => start.mutate()}>
          {running ? 'Refreshing…' : 'Start refresh'}
        </Button>
      </PopoverContent>
    </Popover>
  )
}
