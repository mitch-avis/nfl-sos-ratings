import { PageHeader } from '@/components/common/PageHeader'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { getEntityConfig } from '@/domain/entityConfig'
import { GLOSSARY_SECTIONS, getMetricDescription, getMetricMetadata } from '@/domain/metricMetadata'

/** Rating meanings, the headline ranking per entity, and every glossary metric. */
export function GlossaryPage() {
  const teams = getEntityConfig('teams')
  const qbs = getEntityConfig('qbs')
  return (
    <div className="flex flex-col gap-5">
      <PageHeader
        title="Glossary"
        description={
          <>
            What the ratings mean and which one to use for which question. The full methodology is in
            the repository at <code className="text-foreground">docs/methodology.md</code>.
          </>
        }
      />

      <div className="grid gap-3 md:grid-cols-2">
        {[
          { heading: 'Teams', config: teams },
          { heading: 'Quarterbacks', config: qbs },
        ].map(({ heading, config }) => (
          <Card key={heading} className="gap-2 px-4 py-3">
            <div className="text-xs font-medium tracking-wide text-muted-foreground uppercase">{heading}</div>
            <div className="font-semibold">{config.primaryRankingLabel}</div>
            <p className="text-sm text-muted-foreground">{config.primaryRankingDescription}</p>
          </Card>
        ))}
      </div>

      {GLOSSARY_SECTIONS.map((section) => (
        <Card key={section.title} className="gap-4">
          <CardHeader>
            <CardTitle className="text-base">{section.title}</CardTitle>
            <CardDescription>{section.description}</CardDescription>
          </CardHeader>
          <CardContent>
            <dl className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {section.metrics.map((metric) => {
                const metadata = getMetricMetadata(metric)
                return (
                  <div key={metric} className="rounded-md border bg-muted/30 px-3 py-2">
                    <dt className="font-medium">{metadata.fullName}</dt>
                    <dd className="text-sm">{metadata.shortDescription}</dd>
                    <p className="mt-1 text-sm text-muted-foreground">
                      {metadata.label !== metadata.fullName ? `Shown in the tables as ${metadata.label}. ` : ''}
                      {getMetricDescription(metric)}
                    </p>
                  </div>
                )
              })}
            </dl>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
