import type { ReactElement } from 'react';

import { getEntityConfig } from '../entityConfig';
import { GLOSSARY_SECTIONS, getMetricDescription, getMetricMetadata } from '../metricMetadata';

export function GlossaryPage(): ReactElement {
  const teamConfig = getEntityConfig('teams');
  const qbConfig = getEntityConfig('qbs');

  return (
    <>
      <section className="hero panel detail-hero">
        <p className="eyebrow">Glossary</p>
        <h2>Ratings, Methodology, and Reading Guide</h2>
        <p>
          Use this page to decode the abbreviations in the tables and understand which ratings are
          best suited for overall ranking versus offense-only or schedule-only interpretation.
        </p>
      </section>

      <section className="panel guidance-panel">
        <div className="guidance-grid">
          <article>
            <p className="eyebrow">Teams</p>
            <strong>{teamConfig.primaryRankingLabel}</strong>
            <p>{teamConfig.primaryRankingDescription}</p>
          </article>
          <article>
            <p className="eyebrow">QBs</p>
            <strong>{qbConfig.primaryRankingLabel}</strong>
            <p>{qbConfig.primaryRankingDescription}</p>
          </article>
        </div>
      </section>

      {GLOSSARY_SECTIONS.map((section) => (
        <section key={section.title} className="panel detail-section glossary-section">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Glossary Section</p>
              <h2>{section.title}</h2>
            </div>
          </div>
          <p className="glossary-description">{section.description}</p>
          <dl className="metric-grid">
            {section.metrics.map((metric) => {
              const metadata = getMetricMetadata(metric);
              return (
                <div key={metric} className="metric-cell glossary-cell">
                  <dt>{metadata.fullName}</dt>
                  <dd>{metadata.shortDescription}</dd>
                  <p>
                    {metadata.label !== metadata.fullName ? `Shown in the UI as ${metadata.label}. ` : ''}
                    {getMetricDescription(metric)}
                  </p>
                </div>
              );
            })}
          </dl>
        </section>
      ))}
    </>
  );
}