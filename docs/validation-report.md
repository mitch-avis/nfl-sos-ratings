# Validation Report

Evaluation seasons: 1999-2025. Prediction weeks start at 5.

## Command

```bash
nfl-sos-ratings validate --data-dir data --start-season 1999 --end-season 2025 --start-week 5 --report-path docs/validation-report.md
```

## Decision Rule

The published team rating (TeamRating) is rebuilt each week from that season's earlier games, with
its preseason prior from earlier seasons until a team has played 9 games, alongside SRS and raw EPA
margin built from the same games alone. A margin model fit on earlier predictions turns each rating
gap into a predicted home margin. TeamRating stays the published headline unless its overall mean
absolute error is significantly worse than RawEPA's or SRS's: the 95% paired-bootstrap interval of
the difference lies entirely above zero. Elo carries every rating across seasons and is shown as a
reference only. This rule was written before the first run, and the prior joined TeamRating after
its own pre-registered test.

Decision: adopt. TeamRating is not significantly worse than either comparator.

- TeamRating vs RawEPA, overall MAE difference -0.121 (95% CI -0.183 to -0.060).
- TeamRating vs SRS, overall MAE difference -0.083 (95% CI -0.155 to -0.015).

## Intervals That Exclude Zero

Every pair and split whose 95% interval excludes zero, in either direction. A negative difference
favors the first baseline.

- early: TeamRating vs RawEPA, -0.223 (95% CI -0.344 to -0.107)
- late: RawEPA vs Elo, +0.119 (95% CI +0.015 to +0.217)
- late: SRS vs Elo, +0.100 (95% CI +0.007 to +0.196)
- late: TeamRating vs RawEPA, -0.092 (95% CI -0.166 to -0.022)
- overall: RawEPA vs Elo, +0.116 (95% CI +0.034 to +0.199)
- overall: SRS vs Elo, +0.078 (95% CI +0.000 to +0.159)
- overall: TeamRating vs RawEPA, -0.121 (95% CI -0.183 to -0.060)
- overall: TeamRating vs SRS, -0.083 (95% CI -0.155 to -0.015)

## Walk-Forward Summary

| Baseline | Split | Games | MAE | RMSE |
| --- | --- | --- | --- | --- |
| TeamRating | early | 1141 | 10.567 | 13.691 |
| SRS | early | 1141 | 10.684 | 13.887 |
| Elo | early | 1141 | 10.686 | 13.880 |
| RawEPA | early | 1141 | 10.790 | 13.919 |
| Elo | late | 4156 | 10.550 | 13.511 |
| TeamRating | late | 4156 | 10.577 | 13.526 |
| SRS | late | 4156 | 10.651 | 13.707 |
| RawEPA | late | 4156 | 10.670 | 13.730 |
| TeamRating | overall | 5297 | 10.575 | 13.562 |
| Elo | overall | 5297 | 10.580 | 13.591 |
| SRS | overall | 5297 | 10.658 | 13.746 |
| RawEPA | overall | 5297 | 10.696 | 13.771 |

## Paired Bootstrap MAE Differences

A negative difference favors Baseline A.

| Baseline A | Baseline B | Split | Games | MAE Diff | CI Lower | CI Upper |
| --- | --- | --- | --- | --- | --- | --- |
| RawEPA | Elo | early | 1141 | 0.104 | -0.036 | 0.243 |
| RawEPA | SRS | early | 1141 | 0.106 | -0.041 | 0.256 |
| SRS | Elo | early | 1141 | -0.002 | -0.134 | 0.133 |
| TeamRating | Elo | early | 1141 | -0.119 | -0.262 | 0.028 |
| TeamRating | RawEPA | early | 1141 | -0.223 | -0.344 | -0.107 |
| TeamRating | SRS | early | 1141 | -0.117 | -0.273 | 0.035 |
| RawEPA | Elo | late | 4156 | 0.119 | 0.015 | 0.217 |
| RawEPA | SRS | late | 4156 | 0.019 | -0.027 | 0.064 |
| SRS | Elo | late | 4156 | 0.100 | 0.007 | 0.196 |
| TeamRating | Elo | late | 4156 | 0.027 | -0.051 | 0.104 |
| TeamRating | RawEPA | late | 4156 | -0.092 | -0.166 | -0.022 |
| TeamRating | SRS | late | 4156 | -0.074 | -0.155 | 0.005 |
| RawEPA | Elo | overall | 5297 | 0.116 | 0.034 | 0.199 |
| RawEPA | SRS | overall | 5297 | 0.038 | -0.008 | 0.085 |
| SRS | Elo | overall | 5297 | 0.078 | 0.000 | 0.159 |
| TeamRating | Elo | overall | 5297 | -0.005 | -0.074 | 0.063 |
| TeamRating | RawEPA | overall | 5297 | -0.121 | -0.183 | -0.060 |
| TeamRating | SRS | overall | 5297 | -0.083 | -0.155 | -0.015 |

## Weekly MAE

| Week | Baseline | Games | MAE | RMSE |
| --- | --- | --- | --- | --- |
| 5 | Elo | 384 | 10.223 | 13.427 |
| 5 | RawEPA | 384 | 10.255 | 13.443 |
| 5 | SRS | 384 | 10.167 | 13.421 |
| 5 | TeamRating | 384 | 10.051 | 13.285 |
| 6 | Elo | 379 | 10.511 | 13.496 |
| 6 | RawEPA | 379 | 10.575 | 13.609 |
| 6 | SRS | 379 | 10.352 | 13.433 |
| 6 | TeamRating | 379 | 10.396 | 13.365 |
| 7 | Elo | 378 | 11.332 | 14.689 |
| 7 | RawEPA | 378 | 11.550 | 14.681 |
| 7 | SRS | 378 | 11.543 | 14.773 |
| 7 | TeamRating | 378 | 11.264 | 14.401 |
| 8 | Elo | 378 | 10.412 | 13.243 |
| 8 | RawEPA | 378 | 10.550 | 13.419 |
| 8 | SRS | 378 | 10.577 | 13.470 |
| 8 | TeamRating | 378 | 10.466 | 13.214 |
| 9 | Elo | 371 | 10.112 | 13.102 |
| 9 | RawEPA | 371 | 10.308 | 13.202 |
| 9 | SRS | 371 | 10.156 | 13.089 |
| 9 | TeamRating | 371 | 10.300 | 13.150 |
| 10 | Elo | 384 | 11.199 | 14.258 |
| 10 | RawEPA | 384 | 10.934 | 14.043 |
| 10 | SRS | 384 | 10.994 | 14.170 |
| 10 | TeamRating | 384 | 11.207 | 14.229 |
| 11 | Elo | 401 | 9.606 | 12.779 |
| 11 | RawEPA | 401 | 9.772 | 12.973 |
| 11 | SRS | 401 | 9.752 | 12.934 |
| 11 | TeamRating | 401 | 9.650 | 12.716 |
| 12 | Elo | 417 | 9.730 | 12.585 |
| 12 | RawEPA | 417 | 9.909 | 12.807 |
| 12 | SRS | 417 | 9.896 | 12.785 |
| 12 | TeamRating | 417 | 9.790 | 12.590 |
| 13 | Elo | 421 | 10.288 | 13.246 |
| 13 | RawEPA | 421 | 10.581 | 13.726 |
| 13 | SRS | 421 | 10.531 | 13.645 |
| 13 | TeamRating | 421 | 10.487 | 13.515 |
| 14 | Elo | 418 | 11.129 | 14.002 |
| 14 | RawEPA | 418 | 11.248 | 14.343 |
| 14 | SRS | 418 | 11.172 | 14.242 |
| 14 | TeamRating | 418 | 10.874 | 13.860 |
| 15 | Elo | 429 | 10.613 | 13.504 |
| 15 | RawEPA | 429 | 10.794 | 13.886 |
| 15 | SRS | 429 | 10.722 | 13.752 |
| 15 | TeamRating | 429 | 10.484 | 13.547 |
| 16 | Elo | 429 | 11.166 | 14.019 |
| 16 | RawEPA | 429 | 11.139 | 14.225 |
| 16 | SRS | 429 | 11.215 | 14.264 |
| 16 | TeamRating | 429 | 11.192 | 14.068 |
| 17 | Elo | 428 | 11.227 | 14.265 |
| 17 | RawEPA | 428 | 11.469 | 14.643 |
| 17 | SRS | 428 | 11.503 | 14.679 |
| 17 | TeamRating | 428 | 11.369 | 14.366 |
| 18 | Elo | 80 | 10.239 | 13.045 |
| 18 | RawEPA | 80 | 10.090 | 12.511 |
| 18 | SRS | 80 | 10.033 | 12.515 |
| 18 | TeamRating | 80 | 10.012 | 12.433 |

## Year-Over-Year Stability

Correlation of each metric with the same team's or qualified passer's value the next season.
Informative only; it does not gate anything.

| Entity | Metric | Pairs | Pearson | Spearman |
| --- | --- | --- | --- | --- |
| qb | adj_qb_epa_per_dropback | 601 | 0.455 | 0.442 |
| qb | qb_any_a | 601 | 0.398 | 0.385 |
| qb | qb_passer_rating | 601 | 0.460 | 0.460 |
| team | SRS | 829 | 0.437 | 0.425 |
| team | team_rating | 829 | 0.432 | 0.425 |

## ESPN QBR Reference

Per-season correlation between adjusted EPA per dropback and ESPN QBR for qualified passers. Mean
Pearson 0.892, mean Spearman 0.874. QBR is a reference, not a fitting target.

| Season | QBs | Pearson | Spearman |
| --- | --- | --- | --- |
| 2006 | 31 | 0.872 | 0.796 |
| 2007 | 28 | 0.915 | 0.905 |
| 2008 | 31 | 0.847 | 0.839 |
| 2009 | 28 | 0.960 | 0.957 |
| 2010 | 31 | 0.931 | 0.917 |
| 2011 | 32 | 0.945 | 0.956 |
| 2012 | 32 | 0.931 | 0.916 |
| 2013 | 34 | 0.871 | 0.868 |
| 2014 | 32 | 0.888 | 0.865 |
| 2015 | 33 | 0.926 | 0.932 |
| 2016 | 30 | 0.928 | 0.910 |
| 2017 | 30 | 0.783 | 0.795 |
| 2018 | 32 | 0.931 | 0.897 |
| 2019 | 30 | 0.854 | 0.841 |
| 2020 | 32 | 0.897 | 0.896 |
| 2021 | 31 | 0.922 | 0.869 |
| 2022 | 30 | 0.801 | 0.741 |
| 2023 | 30 | 0.919 | 0.882 |
| 2024 | 31 | 0.864 | 0.875 |
| 2025 | 30 | 0.861 | 0.821 |
