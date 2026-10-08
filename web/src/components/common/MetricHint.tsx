import { directionText, getMetricDescription, getMetricFormula, getMetricMetadata } from '@/domain/metricMetadata'

/**
 * What a stat's hint says: its full name, the plain sentence, which way is better, and how it is
 * computed when the registry gives a formula. `description` overrides the registry sentence.
 */
export function MetricHint({ column, description }: { column: string; description?: string }) {
  const sentence = description ?? getMetricDescription(column)
  const direction = directionText(column)
  const formula = getMetricFormula(column)
  return (
    <div className="flex flex-col gap-1">
      <div className="font-medium">{getMetricMetadata(column).fullName}</div>
      {sentence ? <p>{sentence}</p> : null}
      {direction ? <p className="text-muted-foreground">{direction}</p> : null}
      {formula ? (
        <p className="text-muted-foreground">
          <span className="font-medium text-foreground">How it&apos;s computed:</span> {formula}
        </p>
      ) : null}
    </div>
  )
}
