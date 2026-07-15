import type { ReactElement } from 'react';

import type { EntityKind } from '../types';

interface OverviewCardsProps {
  displayCount: number;
  entityLabel: string;
  metricCount: number;
  kind: EntityKind;
  selectedSlice: string;
  selectedView: string;
  totalCount: number;
}

export function OverviewCards({
  displayCount,
  entityLabel,
  metricCount,
  kind,
  selectedSlice,
  selectedView,
  totalCount,
}: OverviewCardsProps): ReactElement {
  return (
    <section className="overview-grid">
      <article className="panel stat-card feature-card">
        <p className="eyebrow">Snapshot</p>
        <h2>{displayCount}</h2>
        <p>
          {displayCount === totalCount
            ? `${entityLabel} rows available for this season.`
            : `${displayCount} displayed out of ${totalCount} ${entityLabel.toLowerCase()} rows.`}
        </p>
      </article>
      <article className="panel stat-card">
        <p className="eyebrow">Current View</p>
        <h3>{selectedView}</h3>
        <p>
          {kind === 'teams'
            ? 'Single active stat view for the current team table.'
            : 'Single active stat view for the current QB table.'}
        </p>
      </article>
      <article className="panel stat-card">
        <p className="eyebrow">Current Slice</p>
        <h3>{metricCount}</h3>
        <p>{selectedSlice}</p>
      </article>
    </section>
  );
}