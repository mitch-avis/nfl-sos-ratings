import { Search } from 'lucide-react'
import { useId, useMemo, useState } from 'react'

import { useMetricRegistry } from '@/api/queries'
import { ErrorState } from '@/components/common/ErrorState'
import { LoadingState } from '@/components/common/LoadingState'
import { PageHeader } from '@/components/common/PageHeader'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { buildGlossary, filterGlossary, type GlossaryEntry, type GlossarySection } from '@/domain/glossary'

const METHODOLOGY_URL = 'https://github.com/mitch-avis/nfl-sos-ratings/blob/main/docs/methodology.md'

function GlossaryItem({ entry }: { entry: GlossaryEntry }) {
  return (
    <div className="rounded-md border bg-muted/30 px-3 py-2">
      <dt className="font-medium">{entry.title}</dt>
      <dd className="text-sm text-muted-foreground">
        {entry.shownAs ? `Shown in the tables as ${entry.shownAs}. ` : ''}
        {entry.description}
      </dd>
      {entry.direction ? <dd className="mt-1 text-xs text-muted-foreground">{entry.direction}</dd> : null}
      {entry.formula ? (
        <dd className="mt-1 text-xs text-muted-foreground">
          <span className="font-medium text-foreground">How it&apos;s computed:</span> {entry.formula}
        </dd>
      ) : null}
      {entry.sameAs ? <dd className="mt-1 text-xs text-muted-foreground">The same number as {entry.sameAs}.</dd> : null}
      {entry.since ? <dd className="mt-1 text-xs text-muted-foreground">Data from {entry.since} on.</dd> : null}
      {entry.note ? <dd className="mt-1 text-xs text-muted-foreground">{entry.note}</dd> : null}
    </div>
  )
}

function GlossaryCard({ section }: { section: GlossarySection }) {
  const titleId = useId()
  return (
    <section aria-labelledby={titleId}>
      <Card className="gap-4">
        <CardHeader>
          <CardTitle id={titleId} className="text-base">
            {section.title}
          </CardTitle>
          {section.description ? <CardDescription>{section.description}</CardDescription> : null}
        </CardHeader>
        <CardContent>
          <dl className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {section.entries.map((entry) => (
              <GlossaryItem key={entry.id} entry={entry} />
            ))}
          </dl>
        </CardContent>
      </Card>
    </section>
  )
}

/**
 * Every rating and stat in plain words, built from the metric registry: "Start here" with the ideas
 * the pages rely on and the headline ratings, then each category, with a search over all of it. It
 * loads the registry itself, so a direct visit works without season data.
 */
export function GlossaryPage() {
  const registry = useMetricRegistry()
  const [query, setQuery] = useState('')
  const sections = useMemo(() => (registry.data ? buildGlossary(registry.data) : []), [registry.data])
  const shown = useMemo(() => filterGlossary(sections, query), [query, sections])
  return (
    <div className="flex flex-col gap-5">
      <PageHeader
        title="Glossary"
        description={
          <>
            What every rating and stat means, in plain words, and which way is better. The full method,
            with its tests, is in the{' '}
            <a
              href={METHODOLOGY_URL}
              target="_blank"
              rel="noreferrer"
              className="font-medium text-primary underline-offset-4 hover:underline"
            >
              methodology
            </a>
            .
          </>
        }
      />
      <div className="relative max-w-md">
        <Search className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          type="search"
          aria-label="Search the glossary"
          placeholder="Search the glossary"
          className="pl-8"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
      </div>
      {registry.isPending ? <LoadingState label="Loading the metric definitions…" /> : null}
      {registry.isError ? <ErrorState error={registry.error} title="Could not load the metric definitions" /> : null}
      {shown.map((section) => (
        <GlossaryCard key={section.id} section={section} />
      ))}
      {registry.data && shown.length === 0 ? (
        <p className="text-sm text-muted-foreground">Nothing in the glossary matches &ldquo;{query.trim()}&rdquo;.</p>
      ) : null}
    </div>
  )
}
