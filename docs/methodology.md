# Ratings Methodology

This page explains what the published team and quarterback ratings mean,
what they deliberately avoid, and how they are checked.

The short version:

- The ratings are regular-season, schedule-adjusted quality estimates.
- They are intentionally outcome-free: wins, comeback totals, and similar result stats do not feed
  the published quality ratings.
- The published scale is within-season.
  A rating of `+1.0` means one standard deviation above that season's peers,
  not one standard deviation above some pooled all-time baseline.

See [validation-report.md](validation-report.md) for the current report output and
[README.md](../README.md) for the pipeline overview.

## What The Ratings Claim

For teams, the project publishes:

- `SaOR`: offense after adjusting for the defenses actually faced
- `SaDR`: defense after adjusting for the offenses actually faced
- `SaSTR`: special teams after adjusting for special-teams context
- `SaOvR`: overall team quality from those three adjusted pieces
- `SaCR`: the published weighted team composite

For quarterbacks, the project publishes:

- `QSaOR`: adjusted EPA per dropback from the simultaneous QB solve
- `QSoS`: the difficulty of the defenses faced
- `QRaw`: the raw-performance composite before schedule adjustment
- `QSaCR`: the published weighted QB composite
- `QOutcome`: descriptive outcome context only

These ratings try to answer:

- How good was this team relative to the opponents it actually played?
- How good was this quarterback relative to the defenses he actually faced?

They do not try to answer:

- Who deserved credit for the most wins?
- Who was most clutch?
- Who would necessarily be better in a different era with different rules?

## What They Refuse To Use

The published quality ratings do not consume win totals, win percentage,
fourth-quarter comebacks, game-winning drives, or other result-only outcome stats.

Those fields still exist in the data and in the UI because they are useful context.
They are surfaced separately so users can compare performance and outcomes without mixing them.

Turnover margin is also kept out of the published quality ratings.
It remains a descriptive field, not a hidden rating ingredient.

## Scale And Era Meaning

Every published rating is standardized within its own season.

- `0.0` means that season's average qualifying team or quarterback.
- `+1.0` means one standard deviation better than that season's average.
- `-1.0` means one standard deviation worse than that season's average.

That scale makes cross-season comparisons readable in an era-honest way.
If one QB is `+2.0` in 2007 and another is `+2.0` in 2024,
the claim is not that their raw stat lines were directly equal.
The claim is that each was two standard deviations better than his own contemporaries.

The important assumption is explicit:
cross-season comparison here means within-era dominance,
not a claim that the environment was identical across rules, strategy, and data eras.

## Team Adjustment

The team system is built from play-level EPA on offense and defense,
plus a special-teams component.

At a high level, it solves three things at once across the full schedule graph:

- offense strength
- defense strength
- home-field advantage

That is done with ridge regression.
The ridge penalty shrinks noisy early-season estimates toward the mean,
which is the point: extreme values from small samples should move less than extreme values from
large samples.

The published team composite, `SaCR`, is a weighted blend of five standardized components:

- adjusted offensive passing EPA per offensive snap: `0.3829`
- adjusted offensive rushing EPA per offensive snap: `0.1906`
- adjusted defensive passing EPA per offensive snap: `0.2716`
- adjusted defensive rushing EPA per offensive snap: `0.0974`
- special-teams rating: `0.0575`

The special-teams surface is published separately as `SaSTR` because it is a real part of team
quality, but a smaller one than offense and defense.

## Quarterback Adjustment

The quarterback system is built around QB-controlled passing outcomes,
not team record.

The adjustment backbone is adjusted EPA per dropback from a simultaneous ridge solve.
Each QB row is weighted by dropbacks,
so a 40-dropback game carries more evidence than a tiny relief sample.

The published QB composite, `QSaCR`, is a weighted blend of four standardized components:

- adjusted EPA per dropback: `0.6688`
- adjusted completion percentage above expectation: `0.2146`
- adjusted sack rate: `0.0673`
- adjusted TD-INT margin rate: `0.0493`

`QRaw` uses the same design philosophy before schedule adjustment.

Because CPOE is one of the headline composite ingredients,
the CPOE-bearing QB composites are published for `2006+` only.
For `1999-2005`, `QRaw` and `QSaCR` are intentionally null.
The project does not ship reduced-input versions of those headline metrics,
because two different formulas under the same name would be easy to misread.

Non-CPOE QB ratings still publish for every season:

- `QSaOR`
- `QSoS`
- `QOutcome`

## How The Weights Are Chosen

The composite weights are not chosen by win correlation.
They are fit to predict next-season opponent-adjusted performance.

That choice matters.
If a metric predicts future adjusted performance better than another one,
it deserves more weight.
If it mainly tracks wins or good fortune, it does not.

The current frozen fit windows are:

- teams: `1999-2025`
- quarterbacks: `2006-2025`

The frozen-weight snapshots are recorded in the metrics registry,
and the fitting workflow is reproducible with:

```bash
uv run python -m nfl_sos_ratings.composite_weights
```

Refitting is deliberate.
Weights do not change as a side effect of a normal pipeline run.

## How The Ratings Are Validated

The main team check is walk-forward margin prediction.
At each week cutoff,
the system rebuilds ratings using only information available before the next games,
then predicts next-week home margin from the rating gap plus home field.

The most important comparisons are information-matched baselines:

- raw EPA differential
- SRS
- Elo as an external reference baseline

The current team result is best described as parity with SRS,
not a clean victory over it.

The current report records that the published play-level team backbone plus special teams:

- beats raw EPA on held-out MAE
- improves year-over-year stability over the earlier team path
- reaches practical parity with SRS, with `P(backbone <= SRS) = 0.965`

That is why the methodology page uses the word parity.
The result is strong enough to support the construct,
but not strong enough to justify a superiority claim.

On the QB side, the important checks are different.
The current report shows that `QSaCR`:

- beats passer rating and ANY/A on year-over-year stability
- tracks ESPN QBR closely, at roughly `0.89 / 0.87` mean Pearson/Spearman correlation in the
  current report

Those are secondary checks, not the target used to fit the metric.

## The 2025 QB Worked Example

The motivating concern was simple:
could a QB who feasted on weak defenses look better than he really was,
even after the linear schedule adjustment?

The 2025 Drake Maye versus Matthew Stafford comparison was used as the named example.
In the current report's case-study table:

- Drake Maye: raw EPA/dropback `0.306`, adjusted EPA/dropback `0.244`
- Matthew Stafford: raw EPA/dropback `0.244`, adjusted EPA/dropback `0.226`

Maye's schedule was softer.
The faced-defense coefficient was `-0.029` for Maye versus `0.007` for Stafford,
so the model already penalized him more.

The follow-up program then asked whether some missing channel still favored soft-schedule QBs.
The checks were:

- strong-defense split-half performance
- placebo split against weaker defenses
- opponent-offense/game-script spillover
- leverage filtering
- playoff out-of-sample prediction

The answers were conservative.
The strong-defense split produced a signal,
but the placebo side moved the same way,
so the interpretation was not defense-specific.
The opponent-offense and leverage channels came back null in pooled tests.
The leverage-filtered companion hurt both stability and playoff correlation,
so it was not adopted.

The conclusion was not that Maye had no schedule help.
It was that the current published composite withstood every pre-registered challenge strongly
enough that no new QB-path change was justified.
His edge also survived restriction to top-half defenses,
so the final verdict did not depend on all-opponent averaging alone.

## Subjective Choices That Remain

Not every choice is purely mechanical.
The project is explicit about the remaining judgment calls.

- Component menu: the chosen components are meant to balance predictive value, interpretability,
  and overlap control.
- Predictive target: weights target next-season adjusted performance, not wins.
- Leverage filter default: the flag exists, but the default remains off because the leverage-only
  companion underperformed the published QB path on both stability and playoff checks.

Transparency matters more than pretending those choices do not exist.

## How To Challenge These Ratings

The right way to challenge the methodology is not to argue from one anecdote.
The right way is to propose a falsifiable alternative and test it against a fixed gate.

That means:

- define the new hypothesis clearly
- define the information set it is allowed to use
- define the decision rule before looking at the result
- compare it against the current published path and the relevant baselines

That is the standard the current validation report follows,
and it is the standard future revisions should keep.
