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

- TeamRating vs RawEPA, overall MAE difference -0.128 (95% CI -0.190 to -0.069).
- TeamRating vs SRS, overall MAE difference -0.091 (95% CI -0.159 to -0.024).

## Intervals That Exclude Zero

Every pair and split whose 95% interval excludes zero, in either direction. A negative difference
favors the first baseline.

- early: TeamRating vs Elo, -0.148 (95% CI -0.291 to -0.004)
- early: TeamRating vs RawEPA, -0.245 (95% CI -0.365 to -0.130)
- late: RawEPA vs Elo, +0.121 (95% CI +0.017 to +0.219)
- late: SRS vs Elo, +0.100 (95% CI +0.007 to +0.196)
- late: TeamRating vs RawEPA, -0.096 (95% CI -0.168 to -0.025)
- overall: RawEPA vs Elo, +0.116 (95% CI +0.033 to +0.199)
- overall: SRS vs Elo, +0.078 (95% CI +0.000 to +0.159)
- overall: TeamRating vs RawEPA, -0.128 (95% CI -0.190 to -0.069)
- overall: TeamRating vs SRS, -0.091 (95% CI -0.159 to -0.024)

## Walk-Forward Summary

| Baseline | Split | Games | MAE | RMSE |
| --- | --- | --- | --- | --- |
| TeamRating | early | 1141 | 10.538 | 13.673 |
| SRS | early | 1141 | 10.684 | 13.887 |
| Elo | early | 1141 | 10.686 | 13.880 |
| RawEPA | early | 1141 | 10.783 | 13.912 |
| Elo | late | 4156 | 10.550 | 13.511 |
| TeamRating | late | 4156 | 10.576 | 13.525 |
| SRS | late | 4156 | 10.651 | 13.707 |
| RawEPA | late | 4156 | 10.671 | 13.733 |
| TeamRating | overall | 5297 | 10.567 | 13.557 |
| Elo | overall | 5297 | 10.580 | 13.591 |
| SRS | overall | 5297 | 10.658 | 13.746 |
| RawEPA | overall | 5297 | 10.695 | 13.772 |

## Paired Bootstrap MAE Differences

A negative difference favors Baseline A.

| Baseline A | Baseline B | Split | Games | MAE Diff | CI Lower | CI Upper |
| --- | --- | --- | --- | --- | --- | --- |
| RawEPA | Elo | early | 1141 | 0.097 | -0.045 | 0.237 |
| RawEPA | SRS | early | 1141 | 0.099 | -0.048 | 0.247 |
| SRS | Elo | early | 1141 | -0.002 | -0.134 | 0.133 |
| TeamRating | Elo | early | 1141 | -0.148 | -0.291 | -0.004 |
| TeamRating | RawEPA | early | 1141 | -0.245 | -0.365 | -0.130 |
| TeamRating | SRS | early | 1141 | -0.146 | -0.304 | 0.002 |
| RawEPA | Elo | late | 4156 | 0.121 | 0.017 | 0.219 |
| RawEPA | SRS | late | 4156 | 0.021 | -0.027 | 0.067 |
| SRS | Elo | late | 4156 | 0.100 | 0.007 | 0.196 |
| TeamRating | Elo | late | 4156 | 0.025 | -0.052 | 0.103 |
| TeamRating | RawEPA | late | 4156 | -0.096 | -0.168 | -0.025 |
| TeamRating | SRS | late | 4156 | -0.075 | -0.153 | 0.002 |
| RawEPA | Elo | overall | 5297 | 0.116 | 0.033 | 0.199 |
| RawEPA | SRS | overall | 5297 | 0.038 | -0.008 | 0.083 |
| SRS | Elo | overall | 5297 | 0.078 | 0.000 | 0.159 |
| TeamRating | Elo | overall | 5297 | -0.012 | -0.081 | 0.057 |
| TeamRating | RawEPA | overall | 5297 | -0.128 | -0.190 | -0.069 |
| TeamRating | SRS | overall | 5297 | -0.091 | -0.159 | -0.024 |

## Weekly MAE

| Week | Baseline | Games | MAE | RMSE |
| --- | --- | --- | --- | --- |
| 5 | Elo | 384 | 10.223 | 13.427 |
| 5 | RawEPA | 384 | 10.243 | 13.431 |
| 5 | SRS | 384 | 10.167 | 13.421 |
| 5 | TeamRating | 384 | 9.996 | 13.255 |
| 6 | Elo | 379 | 10.511 | 13.496 |
| 6 | RawEPA | 379 | 10.564 | 13.599 |
| 6 | SRS | 379 | 10.352 | 13.433 |
| 6 | TeamRating | 379 | 10.383 | 13.364 |
| 7 | Elo | 378 | 11.332 | 14.689 |
| 7 | RawEPA | 378 | 11.552 | 14.682 |
| 7 | SRS | 378 | 11.543 | 14.773 |
| 7 | TeamRating | 378 | 11.244 | 14.380 |
| 8 | Elo | 378 | 10.412 | 13.243 |
| 8 | RawEPA | 378 | 10.558 | 13.429 |
| 8 | SRS | 378 | 10.577 | 13.470 |
| 8 | TeamRating | 378 | 10.460 | 13.221 |
| 9 | Elo | 371 | 10.112 | 13.102 |
| 9 | RawEPA | 371 | 10.304 | 13.189 |
| 9 | SRS | 371 | 10.156 | 13.089 |
| 9 | TeamRating | 371 | 10.286 | 13.132 |
| 10 | Elo | 384 | 11.199 | 14.258 |
| 10 | RawEPA | 384 | 10.932 | 14.049 |
| 10 | SRS | 384 | 10.994 | 14.170 |
| 10 | TeamRating | 384 | 11.244 | 14.263 |
| 11 | Elo | 401 | 9.606 | 12.779 |
| 11 | RawEPA | 401 | 9.775 | 12.976 |
| 11 | SRS | 401 | 9.752 | 12.934 |
| 11 | TeamRating | 401 | 9.652 | 12.716 |
| 12 | Elo | 417 | 9.730 | 12.585 |
| 12 | RawEPA | 417 | 9.916 | 12.817 |
| 12 | SRS | 417 | 9.896 | 12.785 |
| 12 | TeamRating | 417 | 9.795 | 12.602 |
| 13 | Elo | 421 | 10.288 | 13.246 |
| 13 | RawEPA | 421 | 10.587 | 13.726 |
| 13 | SRS | 421 | 10.531 | 13.645 |
| 13 | TeamRating | 421 | 10.493 | 13.503 |
| 14 | Elo | 418 | 11.129 | 14.002 |
| 14 | RawEPA | 418 | 11.256 | 14.356 |
| 14 | SRS | 418 | 11.172 | 14.242 |
| 14 | TeamRating | 418 | 10.871 | 13.871 |
| 15 | Elo | 429 | 10.613 | 13.504 |
| 15 | RawEPA | 429 | 10.796 | 13.883 |
| 15 | SRS | 429 | 10.722 | 13.752 |
| 15 | TeamRating | 429 | 10.470 | 13.530 |
| 16 | Elo | 429 | 11.166 | 14.019 |
| 16 | RawEPA | 429 | 11.128 | 14.222 |
| 16 | SRS | 429 | 11.215 | 14.264 |
| 16 | TeamRating | 429 | 11.171 | 14.054 |
| 17 | Elo | 428 | 11.227 | 14.265 |
| 17 | RawEPA | 428 | 11.469 | 14.645 |
| 17 | SRS | 428 | 11.503 | 14.679 |
| 17 | TeamRating | 428 | 11.362 | 14.359 |
| 18 | Elo | 80 | 10.239 | 13.045 |
| 18 | RawEPA | 80 | 10.089 | 12.512 |
| 18 | SRS | 80 | 10.033 | 12.515 |
| 18 | TeamRating | 80 | 10.007 | 12.428 |

## Year-Over-Year Stability

Correlation of each metric with the same team's or qualified passer's value the next season.
Informative only; it does not gate anything.

| Entity | Metric | Pairs | Pearson | Spearman |
| --- | --- | --- | --- | --- |
| qb | adj_qb_epa_per_dropback | 601 | 0.455 | 0.441 |
| qb | qb_any_a | 601 | 0.398 | 0.385 |
| qb | qb_passer_rating | 601 | 0.460 | 0.460 |
| team | SRS | 829 | 0.437 | 0.425 |
| team | team_rating | 829 | 0.434 | 0.427 |

## ESPN QBR Reference

Per-season correlation between adjusted EPA per dropback and ESPN QBR for qualified passers. Mean
Pearson 0.892, mean Spearman 0.874. QBR is a reference, not a fitting target.

| Season | QBs | Pearson | Spearman |
| --- | --- | --- | --- |
| 2006 | 31 | 0.872 | 0.796 |
| 2007 | 28 | 0.914 | 0.906 |
| 2008 | 31 | 0.846 | 0.837 |
| 2009 | 28 | 0.961 | 0.956 |
| 2010 | 31 | 0.930 | 0.918 |
| 2011 | 32 | 0.944 | 0.954 |
| 2012 | 32 | 0.932 | 0.919 |
| 2013 | 34 | 0.868 | 0.870 |
| 2014 | 32 | 0.889 | 0.858 |
| 2015 | 33 | 0.927 | 0.932 |
| 2016 | 30 | 0.929 | 0.909 |
| 2017 | 30 | 0.784 | 0.795 |
| 2018 | 32 | 0.931 | 0.897 |
| 2019 | 30 | 0.855 | 0.841 |
| 2020 | 32 | 0.897 | 0.895 |
| 2021 | 31 | 0.923 | 0.877 |
| 2022 | 30 | 0.801 | 0.744 |
| 2023 | 30 | 0.919 | 0.878 |
| 2024 | 31 | 0.864 | 0.875 |
| 2025 | 30 | 0.860 | 0.826 |
