import { getGroupDescription } from '../metricMetadata';
import type { ReactElement } from 'react';

import { humanizeGroup } from '../format';
import type { EntityKind, TablePayload } from '../types';

interface OverviewCardsProps {
  displayCount: number;
  table: TablePayload;
  entityLabel: string;
  kind: EntityKind;
  totalCount: number;
}

export function OverviewCards({
  displayCount,
  table,
  entityLabel,
  kind,
  totalCount,
}: OverviewCardsProps): ReactElement {
  const groups = Object.entries(table.column_groups);

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
      {groups.map(([group, columns]) => (
        <article key={group} className="panel stat-card">
          <p className="eyebrow">{humanizeGroup(group)}</p>
          <h3>{columns.length}</h3>
          <p>{getGroupDescription(kind, group)}</p>
        </article>
      ))}
    </section>
  );
}