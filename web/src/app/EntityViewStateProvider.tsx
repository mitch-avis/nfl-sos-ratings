import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react'

import type { EntityKind } from '@/api/types'
import {
  RESET_PATCH,
  resolvePageViewState,
  type EntityPageViewState,
  type ResolvedPageViewState,
} from '@/domain/pageViewState'

interface EntityViewStateContextValue {
  states: Record<EntityKind, EntityPageViewState>
  update: (kind: EntityKind, patch: EntityPageViewState) => void
}

const EntityViewStateContext = createContext<EntityViewStateContextValue | null>(null)

/**
 * Keeps each page's controls (view, category, subcategories, sort, search, compared rows) while
 * the user moves between the index, detail, and glossary routes.
 */
export function EntityViewStateProvider({ children }: { children: ReactNode }) {
  const [states, setStates] = useState<Record<EntityKind, EntityPageViewState>>({
    teams: {},
    qbs: {},
  })

  const update = useCallback((kind: EntityKind, patch: EntityPageViewState) => {
    setStates((current) => ({ ...current, [kind]: { ...current[kind], ...patch } }))
  }, [])

  const value = useMemo(() => ({ states, update }), [states, update])
  return <EntityViewStateContext.Provider value={value}>{children}</EntityViewStateContext.Provider>
}

export interface EntityPageState extends ResolvedPageViewState {
  update: (patch: EntityPageViewState) => void
  reset: () => void
}

/** The resolved controls for one entity kind, plus setters. */
export function useEntityPageState(kind: EntityKind): EntityPageState {
  const context = useContext(EntityViewStateContext)
  if (!context) throw new Error('useEntityPageState must be used inside EntityViewStateProvider')
  const { states, update } = context
  const resolved = useMemo(() => resolvePageViewState(kind, states[kind]), [kind, states])
  return useMemo(
    () => ({
      ...resolved,
      update: (patch: EntityPageViewState) => update(kind, patch),
      reset: () => update(kind, RESET_PATCH),
    }),
    [kind, resolved, update],
  )
}
