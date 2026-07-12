import type { ReactElement } from 'react';
import { Link } from 'react-router-dom';

import { getEntityLabel } from '../entityConfig';
import { formatValue } from '../format';
import { getMetricMetadata, getMetricTooltip } from '../metricMetadata';
import { TooltipLabel } from './TooltipLabel';
import { buildColumnStats, getHeatCellStyle } from '../tableState';
import type { EntityConfig, PaletteMode, RowValue, TablePayload, ThemeMode } from '../types';

interface ComparisonPanelProps {
  basePath: string;
  compareColumns: string[];
  compareIds: string[];
  config: EntityConfig;
  palette: PaletteMode;
  season: number;
  table: TablePayload;
  theme: ThemeMode;
  onRemove: (entityId: string) => void;
}

export function ComparisonPanel({
  basePath,
  compareColumns,
  compareIds,
  config,
  palette,
  season,
  table,
  theme,
  onRemove,
}: ComparisonPanelProps): ReactElement | null {
  const compareRows = compareIds
    .map((entityId) => table.rows.find((row) => String(row[config.identityKey] ?? '') === entityId))
    .filter((row): row is Record<string, RowValue> => row !== undefined);
  const compareStats = buildColumnStats(compareRows, compareColumns);

  if (compareRows.length === 0) {
    return null;
  }

  return (
    <section className="panel compare-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Compare Mode</p>
          <h2>{config.singularLabel} Comparison</h2>
        </div>
        <div className="panel-stats">
          <span>{compareRows.length} selected</span>
          <span>Deep-linkable via query string</span>
        </div>
      </div>
      <div className="compare-selected-wrap">
        <p className="eyebrow">Selected</p>
        <div className="compare-pill-row">
          {compareRows.map((row) => {
            const entityId = String(row[config.identityKey] ?? '');
            return (
              <div key={entityId} className="compare-pill">
                <strong>{getEntityLabel(config.kind, row)}</strong>
                <Link to={`${basePath}/${entityId}?season=${season}`}>Detail</Link>
                <button type="button" onClick={() => onRemove(entityId)}>
                  Remove
                </button>
              </div>
            );
          })}
        </div>
      </div>
      <div className="table-shell compare-table-shell">
        <table>
          <thead>
            <tr>
              <th>{config.singularLabel}</th>
              {compareColumns.map((column) => {
                const metadata = getMetricMetadata(column);
                return (
                  <th key={column}>
                    <TooltipLabel label={metadata.label} tooltip={getMetricTooltip(column)} />
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {compareRows.map((row) => {
              const entityId = String(row[config.identityKey] ?? '');
              return (
                <tr key={entityId}>
                  <td>{getEntityLabel(config.kind, row)}</td>
                  {compareColumns.map((column) => (
                    <td
                      key={column}
                      style={getHeatCellStyle(column, row[column] ?? null, compareStats, theme, palette)}
                    >
                      {formatValue(row[column] ?? null)}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}