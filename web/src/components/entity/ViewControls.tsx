import { RotateCcw } from 'lucide-react'

import type { EntityKind, PrimaryView } from '@/api/types'
import { Button } from '@/components/ui/button'
import { humanizeGroup } from '@/domain/format'
import { PRIMARY_VIEWS, type ResolvedEntityViewState } from '@/domain/viewModel'

interface ViewControlsProps {
  canReset: boolean
  kind: EntityKind
  onReset: () => void
  onSelectTeamCategory: (category: string) => void
  onSelectView: (view: PrimaryView) => void
  onToggleSubcategory: (subcategory: string) => void
  state: ResolvedEntityViewState
}

function ToggleRow({
  label,
  options,
  isActive,
  onSelect,
}: {
  label: string
  options: string[]
  isActive: (option: string) => boolean
  onSelect: (option: string) => void
}) {
  return (
    // On phones the options stay on one row that scrolls sideways instead of wrapping into four.
    <div
      role="group"
      aria-label={label}
      className="flex w-full min-w-0 gap-1.5 overflow-x-auto pb-1 [scrollbar-width:thin] sm:w-auto sm:flex-wrap sm:overflow-visible sm:pb-0"
    >
      {options.map((option) => {
        const active = isActive(option)
        return (
          <Button
            key={option}
            type="button"
            size="sm"
            variant={active ? 'default' : 'outline'}
            className="shrink-0"
            aria-pressed={active}
            onClick={() => onSelect(option)}
          >
            {humanizeGroup(option)}
          </Button>
        )
      })}
    </div>
  )
}

/**
 * The page's stat controls: one of six views, then (outside `Ratings`) the team category and the
 * subcategories to include.
 */
export function ViewControls({
  canReset,
  kind,
  onReset,
  onSelectTeamCategory,
  onSelectView,
  onToggleSubcategory,
  state,
}: ViewControlsProps) {
  const showStatRows = state.primaryView !== 'ratings'
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <ToggleRow
          label="View"
          options={PRIMARY_VIEWS}
          isActive={(view) => state.primaryView === view}
          onSelect={(view) => onSelectView(view as PrimaryView)}
        />
        <Button type="button" size="sm" variant="ghost" disabled={!canReset} onClick={onReset}>
          <RotateCcw />
          Reset
        </Button>
      </div>
      {showStatRows && kind === 'teams' ? (
        <ToggleRow
          label="Category"
          options={state.teamCategories}
          isActive={(category) => state.teamCategory === category}
          onSelect={onSelectTeamCategory}
        />
      ) : null}
      {showStatRows && state.activeSubcategoryOptions.length > 0 ? (
        <ToggleRow
          label="Subcategories"
          options={state.activeSubcategoryOptions}
          isActive={(subcategory) => Boolean(state.activeSubcategories[subcategory])}
          onSelect={onToggleSubcategory}
        />
      ) : null}
    </div>
  )
}
