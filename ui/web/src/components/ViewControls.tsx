import type { ReactElement } from 'react';

import { humanizeGroup } from '../format';
import type { EntityKind, PrimaryView } from '../types';
import type { ResolvedEntityViewState } from '../viewModel';

interface ViewControlsProps {
  canReset: boolean;
  kind: EntityKind;
  onReset: () => void;
  onSelectTeamCategory: (category: string) => void;
  onSelectView: (view: PrimaryView) => void;
  onToggleSubcategory: (subcategory: string) => void;
  state: ResolvedEntityViewState;
}

export function ViewControls({
  canReset,
  kind,
  onReset,
  onSelectTeamCategory,
  onSelectView,
  onToggleSubcategory,
  state,
}: ViewControlsProps): ReactElement {
  const showStatRows = state.primaryView !== 'ratings';

  return (
    <div className="view-control-stack">
      <div className="control-row primary-control-row">
        {[
          'ratings',
          'raw_total_stats',
          'per_game_rates',
          'per_play_rates',
          'opponent_per_game_rates',
          'opponent_per_play_rates',
        ].map((view) => (
          <button
            key={view}
            type="button"
            className={state.primaryView === view ? 'group-toggle active' : 'group-toggle'}
            onClick={() => onSelectView(view as PrimaryView)}
          >
            <span>{humanizeGroup(view)}</span>
          </button>
        ))}
        <button
          type="button"
          className="group-toggle reset-toggle"
          disabled={!canReset}
          onClick={onReset}
        >
          <span>Reset</span>
        </button>
      </div>

      {kind === 'teams' ? (
        <>
          <div
            aria-hidden={!showStatRows}
            className={showStatRows ? 'control-row' : 'control-row reserved-control-row'}
          >
            {state.teamCategories.map((category) => (
              <button
                key={category}
                type="button"
                disabled={!showStatRows}
                className={
                  showStatRows && state.teamCategory === category
                    ? 'group-toggle active'
                    : 'group-toggle'
                }
                onClick={() => onSelectTeamCategory(category)}
                tabIndex={showStatRows ? 0 : -1}
              >
                <span>{category}</span>
              </button>
            ))}
          </div>
          <div
            aria-hidden={!showStatRows}
            className={showStatRows ? 'control-row' : 'control-row reserved-control-row'}
          >
            {state.activeSubcategoryOptions.map((subcategory) => (
              <button
                key={subcategory}
                type="button"
                disabled={!showStatRows}
                className={
                  showStatRows && state.activeSubcategories[subcategory]
                    ? 'group-toggle active'
                    : 'group-toggle'
                }
                onClick={() => onToggleSubcategory(subcategory)}
                tabIndex={showStatRows ? 0 : -1}
              >
                <span>{subcategory}</span>
              </button>
            ))}
          </div>
        </>
      ) : (
        <div
          aria-hidden={!showStatRows}
          className={showStatRows ? 'control-row' : 'control-row reserved-control-row'}
        >
          {state.activeSubcategoryOptions.map((subcategory) => (
            <button
              key={subcategory}
              type="button"
              disabled={!showStatRows}
              className={
                showStatRows && state.activeSubcategories[subcategory]
                  ? 'group-toggle active'
                  : 'group-toggle'
              }
              onClick={() => onToggleSubcategory(subcategory)}
              tabIndex={showStatRows ? 0 : -1}
            >
              <span>{subcategory}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}