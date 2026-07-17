# Validation Report

Evaluation seasons: 1999-2025.

Prediction weeks start at 5.

## Command

```bash
uv run python -m nfl_sos_ratings.validation.walk_forward --data-dir data --start-season 1999 --end-season 2025 --start-week 5 --report-path docs/validation-report.md
```

## Block R Regression Note

- A Stage 3c regression combined pooled offense/defense reference arrays with current-season-only
  special-teams reference values, causing the team ratings path to raise a NumPy broadcast error
  before `*_combined.parquet` and `*_ratings.parquet` wrote.
- The fix backfills historical `st_rating` values from `*_simultaneous_team_adjustments.parquet`
  when rebuilding pooled team references and makes the multi-season pipeline exit non-zero with a
  failure summary if any season data step fails.

## Stage 3 History

The original Stage 3 headline compared prior-carrying Elo against within-season-only backbones. That
result is preserved here as history rather than deleted or rewritten.

## Stage 3b Criterion

Stage 3b re-registers the validation target into information-matched leagues.

- League 1 is binding: within-season-only team backbones must beat SRS and RawEPA on held-out MAE,
  with paired-bootstrap support.
- League 2 is informative: prior-carrying forecast-only variants can be compared against Elo, but
  that is not the binding published-rating gate.

## Stage 3b Acceptance Check

- League 1 team headline: Fail. Rolling EPA Weights overall MAE 10.782; Rolling EPA Weights + ST
  overall MAE 10.759; SRS 10.658; RawEPA 10.695.
- League 1 bootstrap vs SRS: MAE delta 0.101 with 95% CI [0.037, 0.167].
- League 1 bootstrap vs RawEPA: MAE delta 0.063 with 95% CI [-0.010, 0.139].
- QB revision sweep: not adopted. Current eligible-QB slope 0.505; fixed-defense slope 0.386; best
  tested lighter-defense-penalty slope 0.557 (lighter_defense_penalty_x0).
- League 2 forecast-only prior experiment: not evaluated in this worktree.

## Stage 3c Decision Rule

> A candidate team backbone is promoted to the published ratings if, on the full held-out
> walk-forward window: (1) it is significantly better than RawEPA and than the Stage 1 SaOvR (95%
> paired-bootstrap CI excluding zero); (2) it is numerically better than SRS on both overall MAE and
> overall RMSE, and not significantly worse than SRS; and (3) adopting it does not degrade team
> year-over-year stability below the Stage 3 recorded value. Statistical parity with SRS plus the
> construct advantages (schedule-adjusted, outcome-free components, unit-level decomposition) is
> sufficient and will be stated plainly, as parity, in the methodology documentation — never
> overclaimed as superiority.

- Rationale: the stricter "beat SRS with CI clearing zero" bar is statistically unattainable on this
  sample, and the current report already shows SRS itself does not separate from RawEPA at 95%.

## Stage 3c Team Outcome

- Candidate selected for the final Stage 3c gate: Play-Level EPA Weights + ST.
- Play-level displacement check: Play-Level EPA Weights + ST overall MAE 10.736 and RMSE 13.779
  versus Rolling EPA Weights + ST MAE 10.759 and RMSE 13.818. Bootstrap delta -0.023 with 95% CI
  [-0.053, 0.006] and P(A<=B) 0.942.
- Candidate vs RawEPA: MAE delta 0.041 with 95% CI [-0.032, 0.117] and P(A<=B) 0.137.
- Candidate vs Stage 1 SaOvR: MAE delta 0.035 with 95% CI [-0.039, 0.108] and P(A<=B) 0.173.
- Candidate vs SRS: overall MAE/RMSE 10.736/13.779 versus 10.658/13.746. Bootstrap delta 0.078 with
  95% CI [0.016, 0.137] and P(A<=B) 0.006.
- Stability guard: Play-Level EPA Weights + ST Pearson/Spearman 0.399/0.377 versus Stage 3 SaOvR
  0.380/0.364.
- Promotion decision under the fixed Stage 3c rule: Fail.

## Acceptance Check

- Leakage discipline: the snapshot perturbation test and prior-only fit test pass.
- Team headline: Fail. SaOvR overall MAE 10.701; Elo 10.580; SRS 10.658; RawEPA 10.695.
- Team late-season context: SaOvR late-week MAE 10.649; Elo 10.550; SRS 10.651; RawEPA 10.671.
- QB stability: Pass. QSaCR Pearson/Spearman 0.498/0.484; passer rating 0.473/0.475; ANY/A
  0.403/0.388.
- External reference: mean QBR Pearson/Spearman correlation 0.889/0.868 across 20 seasons.

## Original Walk-Forward Summary

| Baseline | Split | Games | MAE | RMSE |
| --- | --- | --- | --- | --- |
| Elo | early | 1141 | 10.686 | 13.880 |
| Elo | late | 4156 | 10.550 | 13.511 |
| Elo | overall | 5297 | 10.580 | 13.591 |
| RawEPA | early | 1141 | 10.783 | 13.912 |
| RawEPA | late | 4156 | 10.671 | 13.733 |
| RawEPA | overall | 5297 | 10.695 | 13.771 |
| SRS | early | 1141 | 10.684 | 13.887 |
| SRS | late | 4156 | 10.651 | 13.707 |
| SRS | overall | 5297 | 10.658 | 13.746 |
| SaOvR | early | 1141 | 10.892 | 14.083 |
| SaOvR | late | 4156 | 10.649 | 13.649 |
| SaOvR | overall | 5297 | 10.701 | 13.744 |

## League 1 Team Experiments

| Baseline | Split | Games | MAE | RMSE |
| --- | --- | --- | --- | --- |
| Elo | early | 1141 | 10.686 | 13.880 |
| Elo | late | 4156 | 10.550 | 13.511 |
| Elo | overall | 5297 | 10.580 | 13.591 |
| RawEPA | early | 1141 | 10.783 | 13.912 |
| RawEPA | late | 4156 | 10.671 | 13.733 |
| RawEPA | overall | 5297 | 10.695 | 13.771 |
| SRS | early | 1141 | 10.684 | 13.887 |
| SRS | late | 4156 | 10.651 | 13.707 |
| SRS | overall | 5297 | 10.658 | 13.746 |
| SaOvR | early | 1141 | 10.892 | 14.083 |
| SaOvR | late | 4156 | 10.649 | 13.649 |
| SaOvR | overall | 5297 | 10.701 | 13.744 |
| Rolling EPA Weights | early | 1141 | 10.860 | 14.028 |
| Rolling EPA Weights | late | 4156 | 10.761 | 13.799 |
| Rolling EPA Weights | overall | 5297 | 10.782 | 13.848 |
| Rolling EPA Weights + ST | early | 1141 | 10.835 | 14.002 |
| Rolling EPA Weights + ST | late | 4156 | 10.738 | 13.767 |
| Rolling EPA Weights + ST | overall | 5297 | 10.759 | 13.818 |
| Play-Level EPA Weights + ST | early | 1141 | 10.807 | 13.910 |
| Play-Level EPA Weights + ST | late | 4156 | 10.716 | 13.743 |
| Play-Level EPA Weights + ST | overall | 5297 | 10.736 | 13.779 |

## Paired Bootstrap MAE Deltas

| Baseline A | Baseline B | Split | Games | MAE Delta | CI Lower | CI Upper | P(A<=B) | Distinguishable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Play-Level EPA Weights + ST | RawEPA | early | 1141 | 0.024 | -0.149 | 0.199 | 0.392 | False |
| Play-Level EPA Weights + ST | Rolling EPA Weights | early | 1141 | -0.052 | -0.208 | 0.094 | 0.765 | False |
| Play-Level EPA Weights + ST | Rolling EPA Weights + ST | early | 1141 | -0.027 | -0.098 | 0.045 | 0.774 | False |
| Play-Level EPA Weights + ST | SRS | early | 1141 | 0.123 | -0.019 | 0.274 | 0.045 | False |
| Play-Level EPA Weights + ST | SaOvR | early | 1141 | -0.085 | -0.273 | 0.096 | 0.829 | False |
| RawEPA | SaOvR | early | 1141 | -0.109 | -0.225 | 0.007 | 0.968 | False |
| Rolling EPA Weights | RawEPA | early | 1141 | 0.077 | -0.031 | 0.187 | 0.064 | False |
| Rolling EPA Weights | SRS | early | 1141 | 0.175 | 0.019 | 0.336 | 0.015 | True |
| Rolling EPA Weights | SaOvR | early | 1141 | -0.033 | -0.140 | 0.069 | 0.730 | False |
| Rolling EPA Weights + ST | RawEPA | early | 1141 | 0.051 | -0.109 | 0.220 | 0.270 | False |
| Rolling EPA Weights + ST | Rolling EPA Weights | early | 1141 | -0.025 | -0.165 | 0.110 | 0.665 | False |
| Rolling EPA Weights + ST | SRS | early | 1141 | 0.150 | -0.016 | 0.320 | 0.036 | False |
| Rolling EPA Weights + ST | SaOvR | early | 1141 | -0.058 | -0.229 | 0.108 | 0.753 | False |
| SRS | RawEPA | early | 1141 | -0.099 | -0.245 | 0.053 | 0.911 | False |
| SRS | SaOvR | early | 1141 | -0.208 | -0.390 | -0.042 | 0.995 | True |
| Play-Level EPA Weights + ST | RawEPA | late | 4156 | 0.045 | -0.036 | 0.129 | 0.137 | False |
| Play-Level EPA Weights + ST | Rolling EPA Weights | late | 4156 | -0.044 | -0.123 | 0.034 | 0.875 | False |
| Play-Level EPA Weights + ST | Rolling EPA Weights + ST | late | 4156 | -0.022 | -0.052 | 0.009 | 0.920 | False |
| Play-Level EPA Weights + ST | SRS | late | 4156 | 0.066 | -0.003 | 0.132 | 0.031 | False |
| Play-Level EPA Weights + ST | SaOvR | late | 4156 | 0.067 | -0.017 | 0.149 | 0.059 | False |
| RawEPA | SaOvR | late | 4156 | 0.022 | -0.027 | 0.071 | 0.188 | False |
| Rolling EPA Weights | RawEPA | late | 4156 | 0.089 | 0.036 | 0.145 | 0.001 | True |
| Rolling EPA Weights | SRS | late | 4156 | 0.110 | 0.040 | 0.177 | 0.001 | True |
| Rolling EPA Weights | SaOvR | late | 4156 | 0.112 | 0.070 | 0.153 | 0.000 | True |
| Rolling EPA Weights + ST | RawEPA | late | 4156 | 0.067 | -0.017 | 0.144 | 0.052 | False |
| Rolling EPA Weights + ST | Rolling EPA Weights | late | 4156 | -0.023 | -0.091 | 0.048 | 0.747 | False |
| Rolling EPA Weights + ST | SRS | late | 4156 | 0.087 | 0.015 | 0.158 | 0.006 | True |
| Rolling EPA Weights + ST | SaOvR | late | 4156 | 0.089 | 0.006 | 0.171 | 0.019 | True |
| SRS | RawEPA | late | 4156 | -0.021 | -0.066 | 0.025 | 0.813 | False |
| SRS | SaOvR | late | 4156 | 0.002 | -0.064 | 0.068 | 0.470 | False |
| Play-Level EPA Weights + ST | RawEPA | overall | 5297 | 0.041 | -0.032 | 0.117 | 0.137 | False |
| Play-Level EPA Weights + ST | Rolling EPA Weights | overall | 5297 | -0.046 | -0.115 | 0.020 | 0.908 | False |
| Play-Level EPA Weights + ST | Rolling EPA Weights + ST | overall | 5297 | -0.023 | -0.053 | 0.006 | 0.942 | False |
| Play-Level EPA Weights + ST | SRS | overall | 5297 | 0.078 | 0.016 | 0.137 | 0.006 | True |
| Play-Level EPA Weights + ST | SaOvR | overall | 5297 | 0.035 | -0.039 | 0.108 | 0.173 | False |
| RawEPA | SaOvR | overall | 5297 | -0.006 | -0.052 | 0.040 | 0.605 | False |
| Rolling EPA Weights | RawEPA | overall | 5297 | 0.087 | 0.037 | 0.136 | 0.000 | True |
| Rolling EPA Weights | SRS | overall | 5297 | 0.124 | 0.056 | 0.182 | 0.000 | True |
| Rolling EPA Weights | SaOvR | overall | 5297 | 0.081 | 0.040 | 0.124 | 0.000 | True |
| Rolling EPA Weights + ST | RawEPA | overall | 5297 | 0.063 | -0.010 | 0.139 | 0.045 | False |
| Rolling EPA Weights + ST | Rolling EPA Weights | overall | 5297 | -0.023 | -0.087 | 0.039 | 0.771 | False |
| Rolling EPA Weights + ST | SRS | overall | 5297 | 0.101 | 0.037 | 0.167 | 0.000 | True |
| Rolling EPA Weights + ST | SaOvR | overall | 5297 | 0.057 | -0.015 | 0.130 | 0.061 | False |
| SRS | RawEPA | overall | 5297 | -0.037 | -0.086 | 0.008 | 0.940 | False |
| SRS | SaOvR | overall | 5297 | -0.043 | -0.106 | 0.019 | 0.913 | False |

## Weekly MAE Curves

| Week | Baseline | Games | MAE | RMSE |
| --- | --- | --- | --- | --- |
| 5 | Elo | 384 | 10.223 | 13.427 |
| 6 | Elo | 379 | 10.511 | 13.496 |
| 7 | Elo | 378 | 11.332 | 14.689 |
| 8 | Elo | 378 | 10.412 | 13.243 |
| 9 | Elo | 371 | 10.112 | 13.102 |
| 10 | Elo | 384 | 11.199 | 14.258 |
| 11 | Elo | 401 | 9.606 | 12.779 |
| 12 | Elo | 417 | 9.730 | 12.585 |
| 13 | Elo | 421 | 10.288 | 13.246 |
| 14 | Elo | 418 | 11.129 | 14.002 |
| 15 | Elo | 429 | 10.613 | 13.504 |
| 16 | Elo | 429 | 11.166 | 14.019 |
| 17 | Elo | 428 | 11.227 | 14.265 |
| 18 | Elo | 80 | 10.239 | 13.045 |
| 5 | Play-Level EPA Weights + ST | 384 | 10.253 | 13.316 |
| 6 | Play-Level EPA Weights + ST | 379 | 10.572 | 13.586 |
| 7 | Play-Level EPA Weights + ST | 378 | 11.607 | 14.793 |
| 8 | Play-Level EPA Weights + ST | 378 | 10.725 | 13.570 |
| 9 | Play-Level EPA Weights + ST | 371 | 10.332 | 13.134 |
| 10 | Play-Level EPA Weights + ST | 384 | 11.282 | 14.467 |
| 11 | Play-Level EPA Weights + ST | 401 | 9.665 | 12.863 |
| 12 | Play-Level EPA Weights + ST | 417 | 10.009 | 12.929 |
| 13 | Play-Level EPA Weights + ST | 421 | 10.340 | 13.399 |
| 14 | Play-Level EPA Weights + ST | 418 | 11.202 | 14.341 |
| 15 | Play-Level EPA Weights + ST | 429 | 10.670 | 13.724 |
| 16 | Play-Level EPA Weights + ST | 429 | 11.383 | 14.355 |
| 17 | Play-Level EPA Weights + ST | 428 | 11.590 | 14.622 |
| 18 | Play-Level EPA Weights + ST | 80 | 10.135 | 12.604 |
| 5 | Rolling EPA Weights + ST | 384 | 10.318 | 13.461 |
| 6 | Rolling EPA Weights + ST | 379 | 10.628 | 13.698 |
| 7 | Rolling EPA Weights + ST | 378 | 11.567 | 14.818 |
| 8 | Rolling EPA Weights + ST | 378 | 10.734 | 13.658 |
| 9 | Rolling EPA Weights + ST | 371 | 10.393 | 13.174 |
| 10 | Rolling EPA Weights + ST | 384 | 11.198 | 14.343 |
| 11 | Rolling EPA Weights + ST | 401 | 9.692 | 12.903 |
| 12 | Rolling EPA Weights + ST | 417 | 9.956 | 12.900 |
| 13 | Rolling EPA Weights + ST | 421 | 10.331 | 13.430 |
| 14 | Rolling EPA Weights + ST | 418 | 11.333 | 14.393 |
| 15 | Rolling EPA Weights + ST | 429 | 10.700 | 13.757 |
| 16 | Rolling EPA Weights + ST | 429 | 11.410 | 14.407 |
| 17 | Rolling EPA Weights + ST | 428 | 11.628 | 14.642 |
| 18 | Rolling EPA Weights + ST | 80 | 10.348 | 12.794 |
| 5 | SRS | 384 | 10.167 | 13.421 |
| 6 | SRS | 379 | 10.352 | 13.433 |
| 7 | SRS | 378 | 11.543 | 14.773 |
| 8 | SRS | 378 | 10.577 | 13.470 |
| 9 | SRS | 371 | 10.156 | 13.089 |
| 10 | SRS | 384 | 10.994 | 14.170 |
| 11 | SRS | 401 | 9.752 | 12.934 |
| 12 | SRS | 417 | 9.896 | 12.785 |
| 13 | SRS | 421 | 10.531 | 13.645 |
| 14 | SRS | 418 | 11.172 | 14.242 |
| 15 | SRS | 429 | 10.722 | 13.752 |
| 16 | SRS | 429 | 11.215 | 14.264 |
| 17 | SRS | 428 | 11.503 | 14.679 |
| 18 | SRS | 80 | 10.033 | 12.515 |
| 5 | SaOvR | 384 | 10.464 | 13.755 |
| 6 | SaOvR | 379 | 10.641 | 13.722 |
| 7 | SaOvR | 378 | 11.579 | 14.752 |
| 8 | SaOvR | 378 | 10.578 | 13.431 |
| 9 | SaOvR | 371 | 10.314 | 13.175 |
| 10 | SaOvR | 384 | 11.141 | 14.281 |
| 11 | SaOvR | 401 | 9.663 | 12.789 |
| 12 | SaOvR | 417 | 9.886 | 12.762 |
| 13 | SaOvR | 421 | 10.481 | 13.575 |
| 14 | SaOvR | 418 | 11.178 | 14.176 |
| 15 | SaOvR | 429 | 10.752 | 13.800 |
| 16 | SaOvR | 429 | 11.108 | 14.069 |
| 17 | SaOvR | 428 | 11.385 | 14.424 |
| 18 | SaOvR | 80 | 10.262 | 12.570 |

## Original Per-Season Walk-Forward

| Season | Baseline | Games | MAE | RMSE |
| --- | --- | --- | --- | --- |
| 1999 | Elo | 190 | 10.602 | 13.433 |
| 1999 | RawEPA | 190 | 10.478 | 13.087 |
| 1999 | SRS | 190 | 10.313 | 12.953 |
| 1999 | SaOvR | 190 | 10.385 | 12.987 |
| 2000 | Elo | 189 | 10.654 | 13.095 |
| 2000 | RawEPA | 189 | 10.541 | 13.255 |
| 2000 | SRS | 189 | 10.534 | 13.235 |
| 2000 | SaOvR | 189 | 10.365 | 12.893 |
| 2001 | Elo | 190 | 9.705 | 12.270 |
| 2001 | RawEPA | 190 | 9.934 | 12.837 |
| 2001 | SRS | 190 | 9.949 | 12.781 |
| 2001 | SaOvR | 190 | 10.176 | 13.016 |
| 2002 | Elo | 196 | 9.916 | 13.310 |
| 2002 | RawEPA | 196 | 9.963 | 13.320 |
| 2002 | SRS | 196 | 9.951 | 13.232 |
| 2002 | SaOvR | 196 | 9.924 | 13.323 |
| 2003 | Elo | 196 | 10.374 | 13.297 |
| 2003 | RawEPA | 196 | 10.359 | 13.537 |
| 2003 | SRS | 196 | 10.376 | 13.664 |
| 2003 | SaOvR | 196 | 10.666 | 13.748 |
| 2004 | Elo | 196 | 10.970 | 13.932 |
| 2004 | RawEPA | 196 | 10.995 | 14.022 |
| 2004 | SRS | 196 | 10.858 | 13.938 |
| 2004 | SaOvR | 196 | 10.955 | 13.849 |
| 2005 | Elo | 196 | 9.965 | 13.300 |
| 2005 | RawEPA | 196 | 10.522 | 13.761 |
| 2005 | SRS | 196 | 10.316 | 13.556 |
| 2005 | SaOvR | 196 | 10.378 | 13.589 |
| 2006 | Elo | 196 | 11.167 | 13.919 |
| 2006 | RawEPA | 196 | 10.973 | 13.802 |
| 2006 | SRS | 196 | 10.965 | 13.820 |
| 2006 | SaOvR | 196 | 11.023 | 13.794 |
| 2007 | Elo | 194 | 11.132 | 14.030 |
| 2007 | RawEPA | 194 | 11.482 | 14.504 |
| 2007 | SRS | 194 | 11.017 | 14.003 |
| 2007 | SaOvR | 194 | 11.397 | 14.462 |
| 2008 | Elo | 196 | 11.348 | 14.316 |
| 2008 | RawEPA | 196 | 11.363 | 14.532 |
| 2008 | SRS | 196 | 11.426 | 14.545 |
| 2008 | SaOvR | 196 | 11.343 | 14.331 |
| 2009 | Elo | 194 | 12.268 | 15.755 |
| 2009 | RawEPA | 194 | 12.054 | 15.655 |
| 2009 | SRS | 194 | 12.052 | 15.680 |
| 2009 | SaOvR | 194 | 12.100 | 15.725 |
| 2010 | Elo | 194 | 11.027 | 14.428 |
| 2010 | RawEPA | 194 | 11.193 | 14.566 |
| 2010 | SRS | 194 | 11.236 | 14.640 |
| 2010 | SaOvR | 194 | 11.079 | 14.450 |
| 2011 | Elo | 192 | 11.250 | 14.638 |
| 2011 | RawEPA | 192 | 11.300 | 14.716 |
| 2011 | SRS | 192 | 11.268 | 14.655 |
| 2011 | SaOvR | 192 | 11.328 | 14.728 |
| 2012 | Elo | 193 | 11.192 | 14.633 |
| 2012 | RawEPA | 193 | 11.359 | 14.946 |
| 2012 | SRS | 193 | 11.233 | 14.767 |
| 2012 | SaOvR | 193 | 11.151 | 14.718 |
| 2013 | Elo | 193 | 10.196 | 12.908 |
| 2013 | RawEPA | 193 | 10.390 | 13.298 |
| 2013 | SRS | 193 | 10.273 | 13.139 |
| 2013 | SaOvR | 193 | 10.357 | 13.219 |
| 2014 | Elo | 195 | 11.197 | 14.413 |
| 2014 | RawEPA | 195 | 11.448 | 14.746 |
| 2014 | SRS | 195 | 11.276 | 14.622 |
| 2014 | SaOvR | 195 | 11.865 | 15.074 |
| 2015 | Elo | 193 | 10.271 | 13.177 |
| 2015 | RawEPA | 193 | 10.349 | 13.388 |
| 2015 | SRS | 193 | 10.384 | 13.354 |
| 2015 | SaOvR | 193 | 10.398 | 13.229 |
| 2016 | Elo | 193 | 9.298 | 11.860 |
| 2016 | RawEPA | 193 | 9.770 | 12.390 |
| 2016 | SRS | 193 | 9.529 | 12.139 |
| 2016 | SaOvR | 193 | 9.679 | 12.318 |
| 2017 | Elo | 193 | 10.334 | 13.086 |
| 2017 | RawEPA | 193 | 10.630 | 13.384 |
| 2017 | SRS | 193 | 10.477 | 13.288 |
| 2017 | SaOvR | 193 | 10.746 | 13.716 |
| 2018 | Elo | 193 | 10.529 | 13.697 |
| 2018 | RawEPA | 193 | 10.529 | 13.770 |
| 2018 | SRS | 193 | 10.606 | 13.901 |
| 2018 | SaOvR | 193 | 10.552 | 13.829 |
| 2019 | Elo | 193 | 10.603 | 13.590 |
| 2019 | RawEPA | 193 | 10.735 | 13.556 |
| 2019 | SRS | 193 | 10.821 | 13.822 |
| 2019 | SaOvR | 193 | 10.688 | 13.622 |
| 2020 | Elo | 193 | 10.663 | 13.836 |
| 2020 | RawEPA | 193 | 10.927 | 14.246 |
| 2020 | SRS | 193 | 10.912 | 14.277 |
| 2020 | SaOvR | 193 | 10.777 | 14.082 |
| 2021 | Elo | 208 | 11.704 | 14.649 |
| 2021 | RawEPA | 208 | 11.700 | 14.699 |
| 2021 | SRS | 208 | 11.814 | 14.762 |
| 2021 | SaOvR | 208 | 11.706 | 14.666 |
| 2022 | Elo | 207 | 9.136 | 11.927 |
| 2022 | RawEPA | 207 | 9.245 | 12.038 |
| 2022 | SRS | 207 | 9.300 | 12.080 |
| 2022 | SaOvR | 207 | 9.415 | 12.184 |
| 2023 | Elo | 208 | 10.091 | 13.081 |
| 2023 | RawEPA | 208 | 10.068 | 12.999 |
| 2023 | SRS | 208 | 10.010 | 13.087 |
| 2023 | SaOvR | 208 | 10.021 | 12.891 |
| 2024 | Elo | 208 | 9.855 | 12.821 |
| 2024 | RawEPA | 208 | 10.046 | 12.973 |
| 2024 | SRS | 208 | 10.260 | 13.250 |
| 2024 | SaOvR | 208 | 10.091 | 12.952 |
| 2025 | Elo | 208 | 10.304 | 12.913 |
| 2025 | RawEPA | 208 | 10.528 | 13.268 |
| 2025 | SRS | 208 | 10.664 | 13.371 |
| 2025 | SaOvR | 208 | 10.462 | 13.139 |

## Per-Season SaOvR vs SRS

| Season | SaOvR MAE | SRS MAE | MAE Delta | RMSE Delta |
| --- | --- | --- | --- | --- |
| 1999 | 10.385 | 10.313 | 0.072 | 0.034 |
| 2000 | 10.365 | 10.534 | -0.169 | -0.341 |
| 2001 | 10.176 | 9.949 | 0.227 | 0.235 |
| 2002 | 9.924 | 9.951 | -0.027 | 0.091 |
| 2003 | 10.666 | 10.376 | 0.290 | 0.084 |
| 2004 | 10.955 | 10.858 | 0.098 | -0.089 |
| 2005 | 10.378 | 10.316 | 0.062 | 0.033 |
| 2006 | 11.023 | 10.965 | 0.058 | -0.025 |
| 2007 | 11.397 | 11.017 | 0.380 | 0.459 |
| 2008 | 11.343 | 11.426 | -0.083 | -0.214 |
| 2009 | 12.100 | 12.052 | 0.048 | 0.045 |
| 2010 | 11.079 | 11.236 | -0.157 | -0.189 |
| 2011 | 11.328 | 11.268 | 0.060 | 0.072 |
| 2012 | 11.151 | 11.233 | -0.082 | -0.049 |
| 2013 | 10.357 | 10.273 | 0.083 | 0.081 |
| 2014 | 11.865 | 11.276 | 0.589 | 0.452 |
| 2015 | 10.398 | 10.384 | 0.013 | -0.124 |
| 2016 | 9.679 | 9.529 | 0.150 | 0.179 |
| 2017 | 10.746 | 10.477 | 0.269 | 0.428 |
| 2018 | 10.552 | 10.606 | -0.054 | -0.072 |
| 2019 | 10.688 | 10.821 | -0.133 | -0.200 |
| 2020 | 10.777 | 10.912 | -0.135 | -0.196 |
| 2021 | 11.706 | 11.814 | -0.108 | -0.096 |
| 2022 | 9.415 | 9.300 | 0.115 | 0.104 |
| 2023 | 10.021 | 10.010 | 0.011 | -0.196 |
| 2024 | 10.091 | 10.260 | -0.169 | -0.298 |
| 2025 | 10.462 | 10.664 | -0.201 | -0.231 |

## Per-Season Rolling EPA Weights + ST vs SRS

| Season | Rolling EPA Weights + ST MAE | SRS MAE | MAE Delta | RMSE Delta |
| --- | --- | --- | --- | --- |
| 1999 | 10.368 | 10.313 | 0.054 | 0.134 |
| 2000 | 10.081 | 10.534 | -0.453 | -0.522 |
| 2001 | 10.012 | 9.949 | 0.063 | 0.200 |
| 2002 | 10.207 | 9.951 | 0.255 | 0.206 |
| 2003 | 10.644 | 10.376 | 0.269 | 0.101 |
| 2004 | 11.168 | 10.858 | 0.311 | 0.356 |
| 2005 | 10.328 | 10.316 | 0.012 | 0.225 |
| 2006 | 10.999 | 10.965 | 0.034 | 0.045 |
| 2007 | 11.278 | 11.017 | 0.261 | 0.396 |
| 2008 | 11.740 | 11.426 | 0.314 | 0.214 |
| 2009 | 12.284 | 12.052 | 0.232 | 0.238 |
| 2010 | 11.273 | 11.236 | 0.037 | 0.026 |
| 2011 | 11.473 | 11.268 | 0.205 | 0.211 |
| 2012 | 11.466 | 11.233 | 0.233 | 0.128 |
| 2013 | 10.391 | 10.273 | 0.118 | 0.002 |
| 2014 | 11.227 | 11.276 | -0.049 | -0.070 |
| 2015 | 10.477 | 10.384 | 0.093 | 0.100 |
| 2016 | 9.497 | 9.529 | -0.032 | -0.051 |
| 2017 | 10.609 | 10.477 | 0.131 | 0.272 |
| 2018 | 10.953 | 10.606 | 0.347 | 0.365 |
| 2019 | 10.725 | 10.821 | -0.097 | -0.185 |
| 2020 | 10.797 | 10.912 | -0.115 | -0.288 |
| 2021 | 11.728 | 11.814 | -0.086 | -0.180 |
| 2022 | 9.839 | 9.300 | 0.539 | 0.258 |
| 2023 | 10.088 | 10.010 | 0.078 | -0.049 |
| 2024 | 10.336 | 10.260 | 0.075 | 0.054 |
| 2025 | 10.541 | 10.664 | -0.122 | -0.283 |

## Stability

| Metric | Entity | Paired Rows | Pearson | Spearman |
| --- | --- | --- | --- | --- |
| QSaCR | qb | 605 | 0.498 | 0.484 |
| qb_any_a | qb | 605 | 0.403 | 0.388 |
| qb_passer_rating | qb | 605 | 0.473 | 0.475 |
| SaOvR | team | 829 | 0.380 | 0.364 |

## QBR Correlations

| Season | Joined Rows | Pearson | Spearman |
| --- | --- | --- | --- |
| 2006 | 31 | 0.852 | 0.768 |
| 2007 | 28 | 0.909 | 0.907 |
| 2008 | 31 | 0.861 | 0.848 |
| 2009 | 28 | 0.963 | 0.961 |
| 2010 | 31 | 0.942 | 0.930 |
| 2011 | 32 | 0.950 | 0.949 |
| 2012 | 32 | 0.930 | 0.905 |
| 2013 | 34 | 0.876 | 0.879 |
| 2014 | 32 | 0.909 | 0.907 |
| 2015 | 33 | 0.903 | 0.885 |
| 2016 | 30 | 0.914 | 0.887 |
| 2017 | 30 | 0.807 | 0.788 |
| 2018 | 32 | 0.920 | 0.881 |
| 2019 | 30 | 0.833 | 0.830 |
| 2020 | 32 | 0.887 | 0.891 |
| 2021 | 31 | 0.911 | 0.861 |
| 2022 | 30 | 0.800 | 0.747 |
| 2023 | 30 | 0.918 | 0.879 |
| 2024 | 31 | 0.875 | 0.885 |
| 2025 | 30 | 0.824 | 0.776 |

## QB Adjustment Audit

| Season | Eligible QBs | Slope | Correlation | Mean Abs Residual |
| --- | --- | --- | --- | --- |
| 1999 | 38 | 0.945 | 0.732 | 0.037 |
| 2000 | 36 | 0.809 | 0.561 | 0.030 |
| 2001 | 30 | 0.858 | 0.613 | 0.017 |
| 2002 | 32 | 0.627 | 0.383 | 0.019 |
| 2003 | 32 | 1.084 | 0.631 | 0.041 |
| 2004 | 33 | 1.014 | 0.927 | 0.030 |
| 2005 | 34 | 1.095 | 0.571 | 0.023 |
| 2006 | 32 | 1.124 | 0.685 | 0.027 |
| 2007 | 33 | 0.735 | 0.513 | 0.026 |
| 2008 | 32 | 0.974 | 0.846 | 0.016 |
| 2009 | 32 | 0.860 | 0.541 | 0.034 |
| 2010 | 31 | 0.925 | 0.677 | 0.017 |
| 2011 | 35 | 0.865 | 0.414 | 0.027 |
| 2012 | 32 | 0.645 | 0.534 | 0.022 |
| 2013 | 36 | 1.075 | 0.728 | 0.022 |
| 2014 | 33 | 1.029 | 0.760 | 0.018 |
| 2015 | 36 | 0.937 | 0.720 | 0.028 |
| 2016 | 30 | 0.656 | 0.527 | 0.027 |
| 2017 | 32 | 0.932 | 0.694 | 0.031 |
| 2018 | 33 | 0.917 | 0.481 | 0.020 |
| 2019 | 32 | 0.818 | 0.564 | 0.020 |
| 2020 | 35 | 1.105 | 0.814 | 0.032 |
| 2021 | 31 | 1.426 | 0.774 | 0.016 |
| 2022 | 34 | 1.641 | 0.593 | 0.024 |
| 2023 | 33 | 0.443 | 0.334 | 0.034 |
| 2024 | 36 | 1.037 | 0.667 | 0.020 |
| 2025 | 34 | 0.140 | 0.059 | 0.029 |

## QB Defense Spread Audit

| Season | Team Defense SD | QB Defense SD | QB/Team Ratio |
| --- | --- | --- | --- |
| 1999 | 0.037 | 0.090 | 2.397 |
| 2000 | 0.035 | 0.087 | 2.504 |
| 2001 | 0.034 | 0.079 | 2.312 |
| 2002 | 0.038 | 0.093 | 2.415 |
| 2003 | 0.036 | 0.088 | 2.414 |
| 2004 | 0.049 | 0.099 | 2.028 |
| 2005 | 0.026 | 0.066 | 2.544 |
| 2006 | 0.033 | 0.081 | 2.436 |
| 2007 | 0.027 | 0.071 | 2.585 |
| 2008 | 0.040 | 0.100 | 2.527 |
| 2009 | 0.054 | 0.097 | 1.818 |
| 2010 | 0.030 | 0.075 | 2.474 |
| 2011 | 0.029 | 0.073 | 2.495 |
| 2012 | 0.031 | 0.078 | 2.500 |
| 2013 | 0.034 | 0.079 | 2.336 |
| 2014 | 0.032 | 0.075 | 2.376 |
| 2015 | 0.038 | 0.092 | 2.399 |
| 2016 | 0.047 | 0.085 | 1.785 |
| 2017 | 0.037 | 0.092 | 2.506 |
| 2018 | 0.029 | 0.069 | 2.415 |
| 2019 | 0.042 | 0.099 | 2.344 |
| 2020 | 0.038 | 0.091 | 2.367 |
| 2021 | 0.033 | 0.076 | 2.271 |
| 2022 | 0.025 | 0.060 | 2.376 |
| 2023 | 0.034 | 0.078 | 2.302 |
| 2024 | 0.026 | 0.062 | 2.391 |
| 2025 | 0.042 | 0.101 | 2.419 |

## QB Revision Sweep

| Variant | Eligible QBs | Slope | Correlation | Defense Penalty Multiplier |
| --- | --- | --- | --- | --- |
| current | 33 | 0.505 | 0.444 | - |
| fixed_team_defense | 33 | 0.386 | 0.158 | - |
| lighter_defense_penalty_x0 | 33 | 0.557 | 0.545 | 0.000 |
| lighter_defense_penalty_x0.05 | 33 | 0.554 | 0.540 | 0.050 |
| lighter_defense_penalty_x0.1 | 33 | 0.552 | 0.534 | 0.100 |
| lighter_defense_penalty_x0.25 | 33 | 0.544 | 0.518 | 0.250 |
| lighter_defense_penalty_x0.5 | 33 | 0.531 | 0.493 | 0.500 |
| lighter_defense_penalty_x1 | 33 | 0.505 | 0.444 | 1.000 |

## Maye/Stafford Case Study

| Variant | QB | Raw EPA/DB | Adjusted EPA/DB | Faced Difficulty | Adjustment Delta |
| --- | --- | --- | --- | --- | --- |
| current | Matthew Stafford | 0.244 | 0.226 | 0.007 | -0.018 |
| current | Drake Maye | 0.306 | 0.244 | -0.029 | -0.063 |
| fixed_team_defense | Matthew Stafford | 0.244 | 0.222 | 0.005 | -0.022 |
| fixed_team_defense | Drake Maye | 0.306 | 0.253 | -0.016 | -0.054 |
| lighter_defense_penalty_x0 | Matthew Stafford | 0.244 | 0.229 | 0.008 | -0.015 |
| lighter_defense_penalty_x0 | Drake Maye | 0.306 | 0.240 | -0.035 | -0.066 |

## QB Open Status

- The published QB composite target and weights remain unchanged, and the split-half companion
  metric is not promoted to a published surface.
- The earlier QB audit continues to stand as a positive linear-adjustment result: the additive
  adjustment operated at full strength in EPA units, the identity checks held, and the fixed-defense
  and lighter-defense-penalty variants were correctly not adopted.
- The only remaining QB follow-up is the opponent-context batch below. If those checks also come
  back null, the current published composite stands as the system's answer.

## D5 Opponent-Offense Effect

- Gate reading: not_supported.
- Pooled weighted slope -0.041 with 95% CI [-0.134, 0.052] and 10 / 27 positive seasons (p = 0.939).

| Scope | Season | QB Seasons | Dropbacks | Slope | Correlation | CI Lower | CI Upper | Positive Seasons | Season Count | Binomial P |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | - | 12415 | 425780.000 | -0.041 | -0.008 | -0.134 | 0.052 | 10 | 27 | 0.939 |
| season | 1999 | 535 | 16820.000 | 0.022 | 0.003 | - | - | - | - | - |
| season | 2000 | 536 | 16882.000 | -0.062 | -0.011 | - | - | - | - | - |
| season | 2001 | 448 | 15027.000 | -0.058 | -0.009 | - | - | - | - | - |
| season | 2002 | 440 | 15155.000 | -0.079 | -0.013 | - | - | - | - | - |
| season | 2003 | 434 | 14208.000 | -0.056 | -0.009 | - | - | - | - | - |
| season | 2004 | 451 | 14756.000 | -0.058 | -0.018 | - | - | - | - | - |
| season | 2005 | 447 | 14402.000 | -0.056 | -0.009 | - | - | - | - | - |
| season | 2006 | 450 | 14475.000 | 0.219 | 0.031 | - | - | - | - | - |
| season | 2007 | 429 | 13955.000 | -0.185 | -0.035 | - | - | - | - | - |
| season | 2008 | 458 | 14996.000 | 0.030 | 0.005 | - | - | - | - | - |
| season | 2009 | 450 | 15090.000 | -0.085 | -0.025 | - | - | - | - | - |
| season | 2010 | 437 | 15113.000 | -0.107 | -0.019 | - | - | - | - | - |
| season | 2011 | 455 | 16007.000 | -0.112 | -0.025 | - | - | - | - | - |
| season | 2012 | 465 | 16741.000 | 0.107 | 0.020 | - | - | - | - | - |
| season | 2013 | 475 | 17144.000 | -0.120 | -0.021 | - | - | - | - | - |
| season | 2014 | 454 | 16366.000 | 0.136 | 0.022 | - | - | - | - | - |
| season | 2015 | 462 | 17049.000 | 0.096 | 0.014 | - | - | - | - | - |
| season | 2016 | 447 | 16684.000 | 0.055 | 0.014 | - | - | - | - | - |
| season | 2017 | 450 | 15770.000 | -0.087 | -0.014 | - | - | - | - | - |
| season | 2018 | 458 | 16551.000 | -0.082 | -0.015 | - | - | - | - | - |
| season | 2019 | 450 | 16347.000 | -0.330 | -0.049 | - | - | - | - | - |
| season | 2020 | 468 | 16736.000 | -0.006 | -0.001 | - | - | - | - | - |
| season | 2021 | 469 | 16694.000 | -0.388 | -0.065 | - | - | - | - | - |
| season | 2022 | 450 | 15717.000 | 0.038 | 0.007 | - | - | - | - | - |
| season | 2023 | 448 | 15575.000 | -0.037 | -0.007 | - | - | - | - | - |
| season | 2024 | 494 | 16389.000 | 0.035 | 0.006 | - | - | - | - | - |
| season | 2025 | 455 | 15131.000 | 0.185 | 0.033 | - | - | - | - | - |

| Season | QB | Faced Opponent Offense | Mean Adjusted Residual | Dropbacks |
| --- | --- | --- | --- | --- |
| 2025 | Drake Maye | -0.026 | -0.000 | 539.000 |
| 2025 | Matthew Stafford | 0.011 | 0.000 | 617.000 |

## D6 Leverage Profile and Filtered Variant

- Moderate-leverage win-probability band: 0.05-0.95.
- Gate reading: not_supported.
- Companion gate: stability fail, playoff correlation fail.
- Pooled weighted slope -0.026 with 95% CI [-0.202, 0.153] and 14 / 27 positive seasons (p = 0.500).

| Scope | Season | QB Seasons | Dropbacks | Slope | Correlation | CI Lower | CI Upper | Positive Seasons | Season Count | Binomial P |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | - | 892 | 421518.000 | -0.026 | -0.010 | -0.202 | 0.153 | 14 | 27 | 0.500 |
| season | 1999 | 38 | 14816.000 | 0.536 | 0.208 | - | - | - | - | - |
| season | 2000 | 36 | 14724.000 | 0.286 | 0.132 | - | - | - | - | - |
| season | 2001 | 30 | 15027.000 | -1.006 | -0.333 | - | - | - | - | - |
| season | 2002 | 32 | 15155.000 | -0.555 | -0.187 | - | - | - | - | - |
| season | 2003 | 32 | 14208.000 | 0.101 | 0.032 | - | - | - | - | - |
| season | 2004 | 33 | 14756.000 | 0.103 | 0.064 | - | - | - | - | - |
| season | 2005 | 34 | 14402.000 | -0.020 | -0.007 | - | - | - | - | - |
| season | 2006 | 32 | 14475.000 | -0.061 | -0.030 | - | - | - | - | - |
| season | 2007 | 33 | 13955.000 | -0.819 | -0.255 | - | - | - | - | - |
| season | 2008 | 32 | 14996.000 | 0.434 | 0.233 | - | - | - | - | - |
| season | 2009 | 32 | 15090.000 | 0.012 | 0.006 | - | - | - | - | - |
| season | 2010 | 31 | 15048.000 | 0.327 | 0.111 | - | - | - | - | - |
| season | 2011 | 34 | 15972.000 | 0.474 | 0.149 | - | - | - | - | - |
| season | 2012 | 32 | 16741.000 | 0.027 | 0.011 | - | - | - | - | - |
| season | 2013 | 36 | 17144.000 | -0.493 | -0.194 | - | - | - | - | - |
| season | 2014 | 33 | 16366.000 | -0.785 | -0.316 | - | - | - | - | - |
| season | 2015 | 35 | 17049.000 | 0.148 | 0.059 | - | - | - | - | - |
| season | 2016 | 30 | 16684.000 | 0.781 | 0.259 | - | - | - | - | - |
| season | 2017 | 32 | 15770.000 | -0.263 | -0.092 | - | - | - | - | - |
| season | 2018 | 33 | 16551.000 | 0.310 | 0.089 | - | - | - | - | - |
| season | 2019 | 32 | 16347.000 | -0.650 | -0.220 | - | - | - | - | - |
| season | 2020 | 35 | 16736.000 | 0.341 | 0.170 | - | - | - | - | - |
| season | 2021 | 31 | 16694.000 | -0.644 | -0.179 | - | - | - | - | - |
| season | 2022 | 33 | 15717.000 | -1.037 | -0.175 | - | - | - | - | - |
| season | 2023 | 32 | 15575.000 | 0.194 | 0.074 | - | - | - | - | - |
| season | 2024 | 36 | 16389.000 | -0.781 | -0.223 | - | - | - | - | - |
| season | 2025 | 33 | 15131.000 | -0.342 | -0.118 | - | - | - | - | - |

| Season | QB | Schedule Softness | Low-Leverage Share | Moderate-Leverage Share |
| --- | --- | --- | --- | --- |
| 2025 | Matthew Stafford | -0.007 | 0.091 | 0.909 |
| 2025 | Drake Maye | 0.029 | 0.121 | 0.879 |

## Stage 3d D1 Split-Half Diagnostics

- Decision gate reading: not_supported.
- Primary top-half gate: passed.
- Placebo check: bottom-half residuals showed a same-direction signal, so the
  strong-defense-specific interpretation is not supported.

Top-half residual regression summary:

| Scope | Season | QB Seasons | Dropbacks | Slope | CI Lower | CI Upper | Positive Seasons | Season Count | Binomial P |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | - | 892 | 215070.000 | 0.488 | 0.154 | 0.818 | 19 | 27 | 0.026 |
| season | 1999 | 38 | 8715.000 | 1.510 | - | - | - | - | - |
| season | 2000 | 36 | 8914.000 | 0.839 | - | - | - | - | - |
| season | 2001 | 30 | 7708.000 | 1.247 | - | - | - | - | - |
| season | 2002 | 32 | 7447.000 | 0.068 | - | - | - | - | - |
| season | 2003 | 32 | 7282.000 | 0.030 | - | - | - | - | - |
| season | 2004 | 33 | 7289.000 | 0.085 | - | - | - | - | - |
| season | 2005 | 34 | 7057.000 | -0.175 | - | - | - | - | - |
| season | 2006 | 32 | 7520.000 | 0.501 | - | - | - | - | - |
| season | 2007 | 33 | 7177.000 | 0.868 | - | - | - | - | - |
| season | 2008 | 32 | 7670.000 | -0.697 | - | - | - | - | - |
| season | 2009 | 32 | 7602.000 | 0.111 | - | - | - | - | - |
| season | 2010 | 31 | 7674.000 | 1.988 | - | - | - | - | - |
| season | 2011 | 34 | 8128.000 | 2.154 | - | - | - | - | - |
| season | 2012 | 32 | 8192.000 | 0.433 | - | - | - | - | - |
| season | 2013 | 36 | 8739.000 | -0.265 | - | - | - | - | - |
| season | 2014 | 33 | 8443.000 | 0.813 | - | - | - | - | - |
| season | 2015 | 35 | 8795.000 | 1.254 | - | - | - | - | - |
| season | 2016 | 30 | 8416.000 | -2.121 | - | - | - | - | - |
| season | 2017 | 32 | 8019.000 | 0.283 | - | - | - | - | - |
| season | 2018 | 33 | 8268.000 | -0.195 | - | - | - | - | - |
| season | 2019 | 32 | 8195.000 | -0.427 | - | - | - | - | - |
| season | 2020 | 35 | 8362.000 | 0.647 | - | - | - | - | - |
| season | 2021 | 31 | 8338.000 | -0.395 | - | - | - | - | - |
| season | 2022 | 33 | 7332.000 | -2.324 | - | - | - | - | - |
| season | 2023 | 32 | 7726.000 | 1.028 | - | - | - | - | - |
| season | 2024 | 36 | 8627.000 | 1.256 | - | - | - | - | - |
| season | 2025 | 33 | 7435.000 | 1.514 | - | - | - | - | - |

Bottom-half placebo summary:

| Scope | Season | QB Seasons | Dropbacks | Slope | CI Lower | CI Upper |
| --- | --- | --- | --- | --- | --- | --- |
| pooled | - | 892 | 210710.000 | 0.473 | 0.118 | 0.836 |
| season | 1999 | 38 | 8105.000 | -0.505 | - | - |
| season | 2000 | 36 | 7968.000 | 0.258 | - | - |
| season | 2001 | 30 | 7319.000 | -0.110 | - | - |
| season | 2002 | 32 | 7708.000 | 0.911 | - | - |
| season | 2003 | 32 | 6926.000 | 0.964 | - | - |
| season | 2004 | 33 | 7467.000 | 1.417 | - | - |
| season | 2005 | 34 | 7345.000 | 0.572 | - | - |
| season | 2006 | 32 | 6955.000 | 0.946 | - | - |
| season | 2007 | 33 | 6778.000 | 0.827 | - | - |
| season | 2008 | 32 | 7326.000 | 1.633 | - | - |
| season | 2009 | 32 | 7488.000 | 0.641 | - | - |
| season | 2010 | 31 | 7439.000 | -0.816 | - | - |
| season | 2011 | 34 | 7879.000 | -0.857 | - | - |
| season | 2012 | 32 | 8549.000 | 0.599 | - | - |
| season | 2013 | 36 | 8405.000 | 1.509 | - | - |
| season | 2014 | 33 | 7923.000 | 0.410 | - | - |
| season | 2015 | 35 | 8254.000 | 0.058 | - | - |
| season | 2016 | 30 | 8268.000 | 3.277 | - | - |
| season | 2017 | 32 | 7751.000 | 0.939 | - | - |
| season | 2018 | 33 | 8283.000 | 1.493 | - | - |
| season | 2019 | 32 | 8152.000 | 1.178 | - | - |
| season | 2020 | 35 | 8374.000 | 0.153 | - | - |
| season | 2021 | 31 | 8356.000 | 0.785 | - | - |
| season | 2022 | 33 | 8385.000 | 3.275 | - | - |
| season | 2023 | 32 | 7849.000 | -0.734 | - | - |
| season | 2024 | 36 | 7762.000 | -0.657 | - | - |
| season | 2025 | 33 | 7696.000 | -0.360 | - | - |

2025 named case rows:

| Season | QB | Faced Difficulty | Additive Prediction | Top-Half Adj EPA/DB | Top-Half Residual | Top-Half DB | Bottom-Half Adj EPA/DB | Bottom-Half Residual | Bottom-Half DB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | Matthew Stafford | 0.007 | 0.251 | 0.237 | -0.014 | 291.000 | 0.263 | 0.012 | 326.000 |
| 2025 | Drake Maye | -0.025 | 0.282 | 0.259 | -0.022 | 186.000 | 0.293 | 0.012 | 353.000 |

## Stage 3d D3 Playoff Validation

- Interpretation rule: whichever metric best predicts playoff performance is evidence about that
  metric, not about 2025 specifically. If QSaCR wins, that is vindicating evidence for the current
  composite and must be recorded as such.

| Season | Metric | QB Seasons | Playoff Dropbacks | Spearman | Spearman CI Lower | Spearman CI Upper | Pearson | Pearson CI Lower | Pearson CI Upper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1999 | QRaw | 11 | 759.000 | 0.827 | 0.331 | 0.964 | 0.847 | 0.279 | 0.958 |
| 1999 | QSaCR | 11 | 759.000 | 0.780 | 0.189 | 0.965 | 0.769 | 0.321 | 0.944 |
| 1999 | QSaOR | 11 | 759.000 | 0.769 | 0.178 | 0.953 | 0.761 | 0.279 | 0.934 |
| 1999 | moderate_leverage_adjusted_epa_per_dropback | 11 | 759.000 | 0.846 | 0.386 | 0.975 | 0.817 | 0.425 | 0.964 |
| 1999 | qb_any_a | 11 | 759.000 | 0.723 | -0.019 | 0.959 | 0.857 | -0.047 | 0.946 |
| 1999 | qb_passer_rating | 11 | 759.000 | 0.595 | -0.389 | 0.887 | 0.686 | -0.326 | 0.902 |
| 1999 | vs_top_half_adjusted_epa_per_dropback | 11 | 759.000 | -0.095 | -0.809 | 0.679 | 0.224 | -0.496 | 0.757 |
| 2000 | QRaw | 12 | 636.000 | 0.303 | -0.438 | 0.934 | 0.418 | -0.242 | 0.859 |
| 2000 | QSaCR | 12 | 636.000 | 0.144 | -0.604 | 0.729 | 0.358 | -0.210 | 0.797 |
| 2000 | QSaOR | 12 | 636.000 | 0.139 | -0.593 | 0.740 | 0.371 | -0.237 | 0.800 |
| 2000 | moderate_leverage_adjusted_epa_per_dropback | 12 | 636.000 | 0.138 | -0.582 | 0.665 | 0.377 | -0.268 | 0.788 |
| 2000 | qb_any_a | 12 | 636.000 | 0.323 | -0.395 | 0.858 | 0.512 | -0.112 | 0.878 |
| 2000 | qb_passer_rating | 12 | 636.000 | 0.062 | -0.604 | 0.642 | 0.305 | -0.334 | 0.758 |
| 2000 | vs_top_half_adjusted_epa_per_dropback | 12 | 636.000 | 0.172 | -0.527 | 0.888 | 0.363 | -0.295 | 0.872 |
| 2001 | QRaw | 12 | 735.000 | 0.367 | -0.237 | 0.807 | 0.255 | -0.226 | 0.721 |
| 2001 | QSaCR | 12 | 735.000 | 0.317 | -0.248 | 0.801 | 0.217 | -0.222 | 0.659 |
| 2001 | QSaOR | 12 | 735.000 | 0.247 | -0.376 | 0.758 | 0.218 | -0.227 | 0.624 |
| 2001 | moderate_leverage_adjusted_epa_per_dropback | 12 | 735.000 | 0.308 | -0.308 | 0.762 | 0.261 | -0.130 | 0.646 |
| 2001 | qb_any_a | 12 | 735.000 | 0.398 | -0.227 | 0.880 | 0.236 | -0.254 | 0.757 |
| 2001 | qb_passer_rating | 12 | 735.000 | 0.443 | -0.167 | 0.828 | 0.339 | -0.147 | 0.726 |
| 2001 | vs_top_half_adjusted_epa_per_dropback | 12 | 735.000 | 0.292 | -0.343 | 0.728 | 0.154 | -0.280 | 0.605 |
| 2002 | QRaw | 11 | 837.000 | 0.308 | -0.356 | 0.807 | 0.279 | -0.151 | 0.730 |
| 2002 | QSaCR | 11 | 837.000 | 0.237 | -0.449 | 0.819 | 0.329 | -0.189 | 0.762 |
| 2002 | QSaOR | 11 | 837.000 | 0.294 | -0.346 | 0.845 | 0.313 | -0.242 | 0.751 |
| 2002 | moderate_leverage_adjusted_epa_per_dropback | 11 | 837.000 | 0.415 | -0.274 | 0.865 | 0.449 | -0.171 | 0.821 |
| 2002 | qb_any_a | 11 | 837.000 | 0.377 | -0.330 | 0.764 | 0.208 | -0.183 | 0.676 |
| 2002 | qb_passer_rating | 11 | 837.000 | 0.217 | -0.392 | 0.658 | 0.102 | -0.327 | 0.594 |
| 2002 | vs_top_half_adjusted_epa_per_dropback | 11 | 837.000 | 0.398 | -0.386 | 0.923 | 0.446 | -0.206 | 0.860 |
| 2003 | QRaw | 11 | 732.000 | 0.451 | -0.368 | 0.934 | 0.543 | -0.235 | 0.909 |
| 2003 | QSaCR | 11 | 732.000 | 0.471 | -0.363 | 0.927 | 0.553 | -0.194 | 0.917 |
| 2003 | QSaOR | 11 | 732.000 | 0.378 | -0.459 | 0.923 | 0.529 | -0.230 | 0.902 |
| 2003 | moderate_leverage_adjusted_epa_per_dropback | 11 | 732.000 | 0.393 | -0.380 | 0.872 | 0.485 | -0.208 | 0.856 |
| 2003 | qb_any_a | 11 | 732.000 | 0.433 | -0.378 | 0.886 | 0.470 | -0.169 | 0.896 |
| 2003 | qb_passer_rating | 11 | 732.000 | 0.239 | -0.554 | 0.853 | 0.337 | -0.382 | 0.857 |
| 2003 | vs_top_half_adjusted_epa_per_dropback | 11 | 732.000 | 0.511 | -0.255 | 0.943 | 0.575 | -0.120 | 0.878 |
| 2004 | QRaw | 12 | 771.000 | 0.340 | -0.371 | 0.807 | 0.498 | -0.156 | 0.854 |
| 2004 | QSaCR | 12 | 771.000 | 0.370 | -0.295 | 0.832 | 0.458 | -0.283 | 0.845 |
| 2004 | QSaOR | 12 | 771.000 | 0.322 | -0.409 | 0.779 | 0.432 | -0.247 | 0.832 |
| 2004 | moderate_leverage_adjusted_epa_per_dropback | 12 | 771.000 | 0.329 | -0.327 | 0.758 | 0.433 | -0.269 | 0.834 |
| 2004 | qb_any_a | 12 | 771.000 | 0.378 | -0.299 | 0.840 | 0.440 | -0.159 | 0.787 |
| 2004 | qb_passer_rating | 12 | 771.000 | 0.307 | -0.375 | 0.817 | 0.359 | -0.209 | 0.719 |
| 2004 | vs_top_half_adjusted_epa_per_dropback | 12 | 771.000 | 0.172 | -0.577 | 0.770 | 0.274 | -0.493 | 0.790 |
| 2005 | QRaw | 11 | 629.000 | 0.774 | 0.226 | 0.985 | 0.608 | 0.303 | 0.890 |
| 2005 | QSaCR | 11 | 629.000 | 0.755 | 0.175 | 1.000 | 0.540 | 0.223 | 0.846 |
| 2005 | QSaOR | 11 | 629.000 | 0.764 | 0.214 | 0.992 | 0.566 | 0.253 | 0.878 |
| 2005 | moderate_leverage_adjusted_epa_per_dropback | 11 | 629.000 | 0.753 | 0.201 | 0.973 | 0.554 | 0.230 | 0.862 |
| 2005 | qb_any_a | 11 | 629.000 | 0.840 | 0.362 | 0.976 | 0.693 | 0.327 | 0.949 |
| 2005 | qb_passer_rating | 11 | 629.000 | 0.720 | 0.132 | 1.000 | 0.669 | 0.328 | 0.914 |
| 2005 | vs_top_half_adjusted_epa_per_dropback | 11 | 629.000 | 0.830 | 0.416 | 1.000 | 0.621 | 0.371 | 0.883 |
| 2006 | QRaw | 10 | 703.000 | 0.537 | -0.180 | 0.849 | 0.599 | 0.180 | 0.885 |
| 2006 | QSaCR | 10 | 703.000 | 0.629 | -0.191 | 0.930 | 0.651 | 0.218 | 0.897 |
| 2006 | QSaOR | 10 | 703.000 | 0.628 | -0.249 | 0.913 | 0.677 | 0.229 | 0.889 |
| 2006 | moderate_leverage_adjusted_epa_per_dropback | 10 | 703.000 | 0.697 | -0.103 | 0.929 | 0.677 | 0.187 | 0.913 |
| 2006 | qb_any_a | 10 | 703.000 | 0.583 | -0.226 | 0.876 | 0.624 | 0.192 | 0.904 |
| 2006 | qb_passer_rating | 10 | 703.000 | 0.619 | -0.149 | 0.913 | 0.683 | 0.254 | 0.901 |
| 2006 | vs_top_half_adjusted_epa_per_dropback | 10 | 703.000 | 0.736 | -0.002 | 0.929 | 0.730 | 0.230 | 0.927 |
| 2007 | QRaw | 11 | 725.000 | -0.030 | -0.531 | 0.909 | -0.061 | -0.473 | 0.717 |
| 2007 | QSaCR | 11 | 725.000 | 0.052 | -0.549 | 0.878 | 0.018 | -0.414 | 0.848 |
| 2007 | QSaOR | 11 | 725.000 | 0.052 | -0.536 | 0.859 | 0.046 | -0.411 | 0.855 |
| 2007 | moderate_leverage_adjusted_epa_per_dropback | 11 | 725.000 | -0.041 | -0.565 | 0.865 | 0.005 | -0.459 | 0.870 |
| 2007 | qb_any_a | 11 | 725.000 | -0.172 | -0.672 | 0.668 | -0.152 | -0.617 | 0.589 |
| 2007 | qb_passer_rating | 11 | 725.000 | -0.135 | -0.673 | 0.721 | -0.112 | -0.561 | 0.621 |
| 2007 | vs_top_half_adjusted_epa_per_dropback | 11 | 725.000 | -0.247 | -0.765 | 0.652 | -0.092 | -0.583 | 0.627 |
| 2008 | QRaw | 11 | 754.000 | 0.079 | -0.696 | 0.691 | 0.203 | -0.404 | 0.654 |
| 2008 | QSaCR | 11 | 754.000 | 0.235 | -0.611 | 0.776 | 0.331 | -0.272 | 0.708 |
| 2008 | QSaOR | 11 | 754.000 | 0.155 | -0.642 | 0.713 | 0.257 | -0.249 | 0.608 |
| 2008 | moderate_leverage_adjusted_epa_per_dropback | 11 | 754.000 | 0.164 | -0.573 | 0.762 | 0.243 | -0.355 | 0.704 |
| 2008 | qb_any_a | 11 | 754.000 | 0.075 | -0.736 | 0.666 | 0.106 | -0.613 | 0.543 |
| 2008 | qb_passer_rating | 11 | 754.000 | 0.012 | -0.765 | 0.613 | 0.207 | -0.391 | 0.619 |
| 2008 | vs_top_half_adjusted_epa_per_dropback | 11 | 754.000 | 0.454 | -0.363 | 0.911 | 0.481 | -0.162 | 0.867 |
| 2009 | QRaw | 12 | 774.000 | 0.467 | -0.164 | 0.836 | 0.285 | -0.134 | 0.814 |
| 2009 | QSaCR | 12 | 774.000 | 0.190 | -0.487 | 0.704 | 0.117 | -0.401 | 0.710 |
| 2009 | QSaOR | 12 | 774.000 | 0.078 | -0.603 | 0.717 | 0.073 | -0.419 | 0.694 |
| 2009 | moderate_leverage_adjusted_epa_per_dropback | 12 | 774.000 | 0.341 | -0.326 | 0.744 | 0.154 | -0.276 | 0.729 |
| 2009 | qb_any_a | 12 | 774.000 | 0.297 | -0.305 | 0.772 | 0.162 | -0.201 | 0.650 |
| 2009 | qb_passer_rating | 12 | 774.000 | 0.581 | 0.046 | 0.862 | 0.322 | 0.003 | 0.813 |
| 2009 | vs_top_half_adjusted_epa_per_dropback | 12 | 774.000 | 0.259 | -0.391 | 0.799 | 0.066 | -0.362 | 0.723 |
| 2010 | QRaw | 12 | 767.000 | 0.059 | -0.687 | 0.680 | 0.022 | -0.632 | 0.611 |
| 2010 | QSaCR | 12 | 767.000 | -0.001 | -0.747 | 0.631 | 0.078 | -0.640 | 0.618 |
| 2010 | QSaOR | 12 | 767.000 | -0.020 | -0.738 | 0.625 | 0.044 | -0.685 | 0.619 |
| 2010 | moderate_leverage_adjusted_epa_per_dropback | 12 | 767.000 | 0.009 | -0.698 | 0.641 | 0.022 | -0.676 | 0.598 |
| 2010 | qb_any_a | 12 | 767.000 | -0.302 | -0.795 | 0.346 | -0.089 | -0.726 | 0.427 |
| 2010 | qb_passer_rating | 12 | 767.000 | -0.160 | -0.706 | 0.469 | -0.153 | -0.697 | 0.392 |
| 2010 | vs_top_half_adjusted_epa_per_dropback | 12 | 767.000 | 0.176 | -0.653 | 0.787 | 0.377 | -0.641 | 0.854 |
| 2011 | QRaw | 11 | 821.000 | 0.528 | -0.249 | 0.918 | 0.452 | -0.220 | 0.903 |
| 2011 | QSaCR | 11 | 821.000 | 0.457 | -0.242 | 0.908 | 0.380 | -0.250 | 0.853 |
| 2011 | QSaOR | 11 | 821.000 | 0.457 | -0.327 | 0.894 | 0.383 | -0.348 | 0.864 |
| 2011 | moderate_leverage_adjusted_epa_per_dropback | 11 | 821.000 | 0.442 | -0.344 | 0.870 | 0.418 | -0.353 | 0.860 |
| 2011 | qb_any_a | 11 | 821.000 | 0.555 | -0.197 | 0.903 | 0.453 | -0.165 | 0.918 |
| 2011 | qb_passer_rating | 11 | 821.000 | 0.458 | -0.302 | 0.921 | 0.406 | -0.264 | 0.878 |
| 2011 | vs_top_half_adjusted_epa_per_dropback | 11 | 821.000 | 0.359 | -0.407 | 0.864 | 0.320 | -0.374 | 0.871 |
| 2012 | QRaw | 10 | 694.000 | 0.231 | -0.588 | 0.903 | 0.345 | -0.554 | 0.900 |
| 2012 | QSaCR | 10 | 694.000 | 0.219 | -0.613 | 0.928 | 0.315 | -0.591 | 0.909 |
| 2012 | QSaOR | 10 | 694.000 | 0.235 | -0.607 | 0.982 | 0.268 | -0.540 | 0.931 |
| 2012 | moderate_leverage_adjusted_epa_per_dropback | 10 | 694.000 | 0.008 | -0.724 | 0.939 | 0.209 | -0.669 | 0.941 |
| 2012 | qb_any_a | 10 | 694.000 | 0.102 | -1.000 | 0.801 | 0.402 | -0.902 | 0.872 |
| 2012 | qb_passer_rating | 10 | 694.000 | 0.075 | -0.770 | 0.832 | 0.330 | -0.631 | 0.871 |
| 2012 | vs_top_half_adjusted_epa_per_dropback | 10 | 694.000 | -0.193 | -0.757 | 0.908 | 0.000 | -0.668 | 0.899 |
| 2013 | QRaw | 12 | 768.000 | 0.251 | -0.526 | 0.718 | 0.255 | -0.192 | 0.605 |
| 2013 | QSaCR | 12 | 768.000 | 0.347 | -0.380 | 0.773 | 0.282 | -0.136 | 0.640 |
| 2013 | QSaOR | 12 | 768.000 | 0.425 | -0.293 | 0.829 | 0.332 | -0.082 | 0.682 |
| 2013 | moderate_leverage_adjusted_epa_per_dropback | 12 | 768.000 | 0.421 | -0.288 | 0.816 | 0.364 | -0.102 | 0.696 |
| 2013 | qb_any_a | 12 | 768.000 | 0.322 | -0.394 | 0.789 | 0.307 | -0.297 | 0.654 |
| 2013 | qb_passer_rating | 12 | 768.000 | 0.203 | -0.480 | 0.695 | 0.293 | -0.216 | 0.615 |
| 2013 | vs_top_half_adjusted_epa_per_dropback | 12 | 768.000 | 0.362 | -0.390 | 0.868 | 0.433 | -0.232 | 0.731 |
| 2014 | QRaw | 11 | 803.000 | 0.121 | -0.622 | 0.697 | 0.188 | -0.439 | 0.622 |
| 2014 | QSaCR | 11 | 803.000 | 0.193 | -0.497 | 0.722 | 0.242 | -0.390 | 0.715 |
| 2014 | QSaOR | 11 | 803.000 | 0.181 | -0.524 | 0.739 | 0.292 | -0.342 | 0.760 |
| 2014 | moderate_leverage_adjusted_epa_per_dropback | 11 | 803.000 | 0.222 | -0.492 | 0.739 | 0.306 | -0.286 | 0.719 |
| 2014 | qb_any_a | 11 | 803.000 | 0.034 | -0.695 | 0.730 | 0.136 | -0.503 | 0.674 |
| 2014 | qb_passer_rating | 11 | 803.000 | 0.117 | -0.648 | 0.727 | 0.162 | -0.449 | 0.643 |
| 2014 | vs_top_half_adjusted_epa_per_dropback | 11 | 803.000 | 0.197 | -0.588 | 0.741 | 0.167 | -0.500 | 0.652 |
| 2015 | QRaw | 11 | 820.000 | -0.056 | -0.743 | 0.596 | 0.215 | -0.653 | 0.631 |
| 2015 | QSaCR | 11 | 820.000 | 0.075 | -0.728 | 0.684 | 0.213 | -0.704 | 0.632 |
| 2015 | QSaOR | 11 | 820.000 | 0.109 | -0.684 | 0.680 | 0.257 | -0.629 | 0.645 |
| 2015 | moderate_leverage_adjusted_epa_per_dropback | 11 | 820.000 | 0.048 | -0.687 | 0.686 | 0.232 | -0.616 | 0.658 |
| 2015 | qb_any_a | 11 | 820.000 | 0.166 | -0.627 | 0.716 | 0.285 | -0.561 | 0.733 |
| 2015 | qb_passer_rating | 11 | 820.000 | 0.084 | -0.721 | 0.693 | 0.295 | -0.699 | 0.691 |
| 2015 | vs_top_half_adjusted_epa_per_dropback | 11 | 820.000 | 0.014 | -0.759 | 0.712 | 0.347 | -0.784 | 0.777 |
| 2016 | QRaw | 10 | 782.000 | 0.910 | 0.525 | 1.000 | 0.871 | 0.625 | 0.977 |
| 2016 | QSaCR | 10 | 782.000 | 0.872 | 0.495 | 1.000 | 0.891 | 0.677 | 0.973 |
| 2016 | QSaOR | 10 | 782.000 | 0.861 | 0.419 | 1.000 | 0.862 | 0.613 | 0.970 |
| 2016 | moderate_leverage_adjusted_epa_per_dropback | 10 | 782.000 | 0.861 | 0.481 | 1.000 | 0.827 | 0.500 | 0.966 |
| 2016 | qb_any_a | 10 | 782.000 | 0.924 | 0.576 | 1.000 | 0.884 | 0.618 | 0.981 |
| 2016 | qb_passer_rating | 10 | 782.000 | 0.960 | 0.696 | 1.000 | 0.890 | 0.708 | 0.981 |
| 2016 | vs_top_half_adjusted_epa_per_dropback | 10 | 782.000 | 0.892 | 0.461 | 1.000 | 0.784 | 0.424 | 0.973 |
| 2017 | QRaw | 11 | 778.000 | 0.507 | -0.367 | 0.916 | 0.536 | -0.121 | 0.878 |
| 2017 | QSaCR | 11 | 778.000 | 0.625 | -0.134 | 0.949 | 0.584 | -0.078 | 0.880 |
| 2017 | QSaOR | 11 | 778.000 | 0.498 | -0.243 | 0.883 | 0.587 | -0.074 | 0.914 |
| 2017 | moderate_leverage_adjusted_epa_per_dropback | 11 | 778.000 | 0.399 | -0.424 | 0.802 | 0.478 | -0.205 | 0.836 |
| 2017 | qb_any_a | 11 | 778.000 | 0.228 | -0.531 | 0.745 | 0.428 | -0.260 | 0.811 |
| 2017 | qb_passer_rating | 11 | 778.000 | 0.427 | -0.350 | 0.821 | 0.438 | -0.240 | 0.802 |
| 2017 | vs_top_half_adjusted_epa_per_dropback | 11 | 778.000 | 0.530 | -0.242 | 0.901 | 0.717 | -0.033 | 0.917 |
| 2018 | QRaw | 10 | 748.000 | 0.235 | -0.414 | 0.875 | 0.241 | -0.209 | 0.761 |
| 2018 | QSaCR | 10 | 748.000 | 0.219 | -0.421 | 0.882 | 0.254 | -0.146 | 0.775 |
| 2018 | QSaOR | 10 | 748.000 | 0.255 | -0.396 | 0.854 | 0.237 | -0.236 | 0.716 |
| 2018 | moderate_leverage_adjusted_epa_per_dropback | 10 | 748.000 | 0.520 | -0.116 | 0.865 | 0.339 | -0.088 | 0.778 |
| 2018 | qb_any_a | 10 | 748.000 | 0.224 | -0.399 | 0.848 | 0.154 | -0.385 | 0.693 |
| 2018 | qb_passer_rating | 10 | 748.000 | 0.078 | -0.577 | 0.797 | 0.074 | -0.470 | 0.706 |
| 2018 | vs_top_half_adjusted_epa_per_dropback | 10 | 748.000 | 0.141 | -0.502 | 0.704 | 0.061 | -0.440 | 0.551 |
| 2019 | QRaw | 12 | 729.000 | -0.037 | -0.640 | 0.602 | 0.025 | -0.578 | 0.580 |
| 2019 | QSaCR | 12 | 729.000 | -0.012 | -0.629 | 0.746 | 0.094 | -0.526 | 0.762 |
| 2019 | QSaOR | 12 | 729.000 | 0.076 | -0.568 | 0.821 | 0.091 | -0.520 | 0.734 |
| 2019 | moderate_leverage_adjusted_epa_per_dropback | 12 | 729.000 | 0.116 | -0.588 | 0.761 | 0.259 | -0.471 | 0.749 |
| 2019 | qb_any_a | 12 | 729.000 | 0.108 | -0.542 | 0.666 | 0.160 | -0.363 | 0.558 |
| 2019 | qb_passer_rating | 12 | 729.000 | -0.111 | -0.674 | 0.621 | 0.041 | -0.444 | 0.469 |
| 2019 | vs_top_half_adjusted_epa_per_dropback | 12 | 729.000 | 0.172 | -0.598 | 0.787 | 0.179 | -0.627 | 0.715 |
| 2020 | QRaw | 13 | 944.000 | 0.446 | -0.113 | 0.801 | 0.429 | -0.022 | 0.747 |
| 2020 | QSaCR | 13 | 944.000 | 0.452 | -0.177 | 0.816 | 0.337 | -0.167 | 0.711 |
| 2020 | QSaOR | 13 | 944.000 | 0.513 | -0.000 | 0.812 | 0.366 | -0.052 | 0.701 |
| 2020 | moderate_leverage_adjusted_epa_per_dropback | 13 | 944.000 | 0.579 | 0.073 | 0.876 | 0.468 | 0.098 | 0.737 |
| 2020 | qb_any_a | 13 | 944.000 | 0.352 | -0.218 | 0.753 | 0.357 | -0.103 | 0.689 |
| 2020 | qb_passer_rating | 13 | 944.000 | 0.227 | -0.331 | 0.735 | 0.255 | -0.330 | 0.666 |
| 2020 | vs_top_half_adjusted_epa_per_dropback | 13 | 944.000 | -0.081 | -0.686 | 0.589 | -0.017 | -0.525 | 0.583 |
| 2021 | QRaw | 14 | 1007.000 | 0.132 | -0.412 | 0.631 | 0.044 | -0.379 | 0.442 |
| 2021 | QSaCR | 14 | 1007.000 | 0.092 | -0.570 | 0.610 | 0.108 | -0.295 | 0.499 |
| 2021 | QSaOR | 14 | 1007.000 | 0.264 | -0.324 | 0.707 | 0.187 | -0.222 | 0.571 |
| 2021 | moderate_leverage_adjusted_epa_per_dropback | 14 | 1007.000 | 0.034 | -0.557 | 0.559 | -0.081 | -0.519 | 0.329 |
| 2021 | qb_any_a | 14 | 1007.000 | 0.069 | -0.544 | 0.588 | -0.132 | -0.534 | 0.382 |
| 2021 | qb_passer_rating | 14 | 1007.000 | 0.064 | -0.597 | 0.567 | -0.117 | -0.579 | 0.407 |
| 2021 | vs_top_half_adjusted_epa_per_dropback | 14 | 1007.000 | -0.091 | -0.662 | 0.574 | 0.014 | -0.467 | 0.503 |
| 2022 | QRaw | 11 | 827.000 | 0.302 | -0.522 | 0.813 | 0.386 | -0.473 | 0.810 |
| 2022 | QSaCR | 11 | 827.000 | -0.006 | -0.850 | 0.630 | 0.244 | -0.723 | 0.720 |
| 2022 | QSaOR | 11 | 827.000 | -0.034 | -0.870 | 0.636 | 0.203 | -0.725 | 0.734 |
| 2022 | moderate_leverage_adjusted_epa_per_dropback | 11 | 827.000 | 0.181 | -0.757 | 0.743 | 0.314 | -0.559 | 0.760 |
| 2022 | qb_any_a | 11 | 827.000 | 0.341 | -0.562 | 0.842 | 0.368 | -0.487 | 0.826 |
| 2022 | qb_passer_rating | 11 | 827.000 | 0.365 | -0.490 | 0.863 | 0.499 | -0.183 | 0.894 |
| 2022 | vs_top_half_adjusted_epa_per_dropback | 11 | 827.000 | 0.249 | -0.624 | 0.885 | 0.224 | -0.774 | 0.868 |
| 2023 | QRaw | 12 | 897.000 | -0.234 | -0.841 | 0.404 | -0.215 | -0.796 | 0.174 |
| 2023 | QSaCR | 12 | 897.000 | -0.252 | -0.770 | 0.352 | -0.350 | -0.824 | -0.004 |
| 2023 | QSaOR | 12 | 897.000 | -0.252 | -0.725 | 0.391 | -0.264 | -0.751 | 0.172 |
| 2023 | moderate_leverage_adjusted_epa_per_dropback | 12 | 897.000 | -0.116 | -0.649 | 0.540 | -0.217 | -0.705 | 0.247 |
| 2023 | qb_any_a | 12 | 897.000 | -0.115 | -0.642 | 0.483 | -0.131 | -0.631 | 0.299 |
| 2023 | qb_passer_rating | 12 | 897.000 | -0.223 | -0.713 | 0.386 | -0.189 | -0.708 | 0.257 |
| 2023 | vs_top_half_adjusted_epa_per_dropback | 12 | 897.000 | 0.198 | -0.425 | 0.647 | 0.147 | -0.276 | 0.550 |
| 2024 | QRaw | 14 | 846.000 | 0.182 | -0.376 | 0.649 | 0.099 | -0.276 | 0.439 |
| 2024 | QSaCR | 14 | 846.000 | 0.215 | -0.325 | 0.672 | 0.168 | -0.192 | 0.452 |
| 2024 | QSaOR | 14 | 846.000 | 0.194 | -0.346 | 0.654 | 0.159 | -0.208 | 0.479 |
| 2024 | moderate_leverage_adjusted_epa_per_dropback | 14 | 846.000 | -0.027 | -0.623 | 0.546 | -0.062 | -0.503 | 0.330 |
| 2024 | qb_any_a | 14 | 846.000 | 0.142 | -0.422 | 0.670 | 0.073 | -0.363 | 0.492 |
| 2024 | qb_passer_rating | 14 | 846.000 | 0.214 | -0.349 | 0.716 | 0.069 | -0.319 | 0.528 |
| 2024 | vs_top_half_adjusted_epa_per_dropback | 14 | 846.000 | -0.076 | -0.590 | 0.507 | -0.010 | -0.366 | 0.383 |
| 2025 | QRaw | 15 | 959.000 | 0.220 | -0.357 | 0.731 | 0.171 | -0.347 | 0.717 |
| 2025 | QSaCR | 15 | 959.000 | 0.162 | -0.403 | 0.739 | 0.136 | -0.376 | 0.696 |
| 2025 | QSaOR | 15 | 959.000 | 0.240 | -0.368 | 0.768 | 0.190 | -0.330 | 0.685 |
| 2025 | moderate_leverage_adjusted_epa_per_dropback | 15 | 959.000 | 0.357 | -0.160 | 0.791 | 0.234 | -0.262 | 0.669 |
| 2025 | qb_any_a | 15 | 959.000 | 0.336 | -0.278 | 0.806 | 0.217 | -0.298 | 0.756 |
| 2025 | qb_passer_rating | 15 | 959.000 | 0.214 | -0.402 | 0.703 | -0.009 | -0.480 | 0.587 |
| 2025 | vs_top_half_adjusted_epa_per_dropback | 15 | 959.000 | 0.349 | -0.327 | 0.847 | 0.252 | -0.348 | 0.791 |
| pooled | QRaw | 313 | 21245.000 | 0.343 | 0.227 | 0.444 | 0.326 | 0.224 | 0.425 |
| pooled | QSaCR | 313 | 21245.000 | 0.326 | 0.209 | 0.440 | 0.308 | 0.205 | 0.412 |
| pooled | QSaOR | 313 | 21245.000 | 0.323 | 0.204 | 0.435 | 0.312 | 0.203 | 0.410 |
| pooled | moderate_leverage_adjusted_epa_per_dropback | 313 | 21245.000 | 0.315 | 0.204 | 0.425 | 0.312 | 0.212 | 0.408 |
| pooled | qb_any_a | 313 | 21245.000 | 0.325 | 0.212 | 0.427 | 0.314 | 0.207 | 0.420 |
| pooled | qb_passer_rating | 313 | 21245.000 | 0.296 | 0.181 | 0.404 | 0.284 | 0.173 | 0.386 |
| pooled | vs_top_half_adjusted_epa_per_dropback | 313 | 21245.000 | 0.257 | 0.143 | 0.369 | 0.267 | 0.153 | 0.372 |

## SaCR Caveat

- SaCR may be evaluated as a secondary line with a caveat: its frozen Stage 2 weights were fit on
  the full 1999-2025 history.
- A walk-forward SaCR line over that same window has look-ahead in the weights.
- SaOvR is the headline walk-forward metric because it does not depend on a fitted Stage 2 weight
  snapshot.
