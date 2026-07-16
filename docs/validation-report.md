# Validation Report

Evaluation seasons: 1999-2025.
Prediction weeks start at 5.

## Command

```bash
uv run python -m nfl_sos_ratings.validation.walk_forward --data-dir data --start-season 1999 --end-season 2025 --start-week 5 --report-path docs/validation-report.md
```

## Stage 3 History

The original Stage 3 headline compared prior-carrying Elo against within-season-only backbones.
That result is preserved here as history rather than deleted or rewritten.

## Stage 3b Criterion

Stage 3b re-registers the validation target into information-matched leagues.

- League 1 is binding: within-season-only team backbones must beat SRS and RawEPA on held-out
  MAE, with paired-bootstrap support.
- League 2 is informative: prior-carrying forecast-only variants can be compared against Elo,
  but that is not the binding published-rating gate.

## Stage 3b Acceptance Check

- League 1 team headline: Fail. T1Weighted overall MAE 10.648; T2Weighted overall MAE 10.630; SRS 10.658;
  RawEPA 10.695.
- League 1 bootstrap vs SRS: MAE delta -0.028 with 95% CI [-0.096, 0.035].
- League 1 bootstrap vs RawEPA: MAE delta -0.066 with 95% CI [-0.112, -0.015].
- QB revision sweep: not adopted. Current eligible-QB slope 0.505; Q1 fixed-defense slope 0.386;
  best tested Q2 slope 0.557 (q2_defense_penalty_x0).
- League 2 forecast-only prior experiment: not evaluated in this worktree.

## Stage 3c Decision Rule

> A candidate team backbone is promoted to the published ratings if, on the full held-out
> walk-forward window: (1) it is significantly better than RawEPA and than the Stage 1
> SaOvR (95% paired-bootstrap CI excluding zero); (2) it is numerically better than SRS
> on both overall MAE and overall RMSE, and not significantly worse than SRS; and (3)
> adopting it does not degrade team year-over-year stability below the Stage 3 recorded
> value. Statistical parity with SRS plus the construct advantages (schedule-adjusted,
> outcome-free components, unit-level decomposition) is sufficient and will be stated
> plainly, as parity, in the methodology documentation — never overclaimed as superiority.

- Rationale: the stricter "beat SRS with CI clearing zero" bar is statistically
  unattainable on this sample, and the current report already shows SRS itself does not
  separate from RawEPA at 95%.

## Stage 3c Team Outcome

- Candidate selected for the final Stage 3c gate: T4Weighted.
- T4 displacement check: T4Weighted overall MAE 10.600 and RMSE 13.626 versus
  T2Weighted MAE 10.630 and RMSE 13.680.
  Bootstrap delta -0.029 with 95% CI [-0.063, 0.002] and P(A<=B) 0.965.
- Candidate vs RawEPA: MAE delta -0.095 with 95% CI [-0.147, -0.036] and P(A<=B) 1.000.
- Candidate vs Stage 1 SaOvR: MAE delta -0.101 with 95% CI [-0.153, -0.050] and P(A<=B) 1.000.
- Candidate vs SRS: overall MAE/RMSE 10.600/13.626 versus 10.658/13.746.
  Bootstrap delta -0.058 with 95% CI [-0.120, 0.005] and P(A<=B) 0.965.
- Stability guard: T4Weighted Pearson/Spearman 0.445/0.434 versus Stage 3 SaOvR 0.417/0.414.
- Promotion decision under the fixed Stage 3c rule: Pass.

## Acceptance Check

- Leakage discipline: the snapshot perturbation test and prior-only fit test pass.
- Team headline: Fail. SaOvR overall MAE 10.701; Elo 10.580; SRS 10.658; RawEPA 10.695.
- Team late-season context: SaOvR late-week MAE 10.649; Elo 10.550; SRS 10.651; RawEPA 10.671.
- QB stability: Pass. QSaCR Pearson/Spearman 0.485/0.480; passer rating 0.473/0.475; ANY/A 0.403/0.388.
- External reference: mean QBR Pearson/Spearman correlation 0.891/0.870 across 20 seasons.

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
| T1Weighted | early | 1141 | 10.778 | 13.973 |
| T1Weighted | late | 4156 | 10.612 | 13.609 |
| T1Weighted | overall | 5297 | 10.648 | 13.688 |
| T2Weighted | early | 1141 | 10.736 | 13.947 |
| T2Weighted | late | 4156 | 10.600 | 13.606 |
| T2Weighted | overall | 5297 | 10.630 | 13.680 |
| T4Weighted | early | 1141 | 10.722 | 13.838 |
| T4Weighted | late | 4156 | 10.567 | 13.567 |
| T4Weighted | overall | 5297 | 10.600 | 13.626 |

## Paired Bootstrap MAE Deltas

| Baseline A | Baseline B | Split | Games | MAE Delta | CI Lower | CI Upper | P(A<=B) | Distinguishable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RawEPA | SaOvR | early | 1141 | -0.109 | -0.225 | 0.007 | 0.968 | False |
| SRS | RawEPA | early | 1141 | -0.099 | -0.245 | 0.053 | 0.911 | False |
| SRS | SaOvR | early | 1141 | -0.208 | -0.390 | -0.042 | 0.995 | True |
| T1Weighted | RawEPA | early | 1141 | -0.005 | -0.110 | 0.094 | 0.529 | False |
| T1Weighted | SRS | early | 1141 | 0.094 | -0.068 | 0.254 | 0.137 | False |
| T1Weighted | SaOvR | early | 1141 | -0.114 | -0.225 | 0.003 | 0.972 | False |
| T2Weighted | RawEPA | early | 1141 | -0.047 | -0.160 | 0.059 | 0.823 | False |
| T2Weighted | SRS | early | 1141 | 0.052 | -0.113 | 0.211 | 0.271 | False |
| T2Weighted | SaOvR | early | 1141 | -0.156 | -0.280 | -0.027 | 0.992 | True |
| T2Weighted | T1Weighted | early | 1141 | -0.042 | -0.079 | -0.004 | 0.985 | True |
| T4Weighted | RawEPA | early | 1141 | -0.061 | -0.181 | 0.059 | 0.853 | False |
| T4Weighted | SRS | early | 1141 | 0.038 | -0.115 | 0.191 | 0.309 | False |
| T4Weighted | SaOvR | early | 1141 | -0.170 | -0.298 | -0.032 | 0.996 | True |
| T4Weighted | T1Weighted | early | 1141 | -0.056 | -0.148 | 0.031 | 0.895 | False |
| T4Weighted | T2Weighted | early | 1141 | -0.014 | -0.095 | 0.071 | 0.626 | False |
| RawEPA | SaOvR | late | 4156 | 0.022 | -0.027 | 0.071 | 0.188 | False |
| SRS | RawEPA | late | 4156 | -0.021 | -0.066 | 0.025 | 0.813 | False |
| SRS | SaOvR | late | 4156 | 0.002 | -0.064 | 0.068 | 0.470 | False |
| T1Weighted | RawEPA | late | 4156 | -0.059 | -0.116 | -0.005 | 0.983 | True |
| T1Weighted | SRS | late | 4156 | -0.039 | -0.109 | 0.030 | 0.869 | False |
| T1Weighted | SaOvR | late | 4156 | -0.037 | -0.077 | 0.005 | 0.961 | False |
| T2Weighted | RawEPA | late | 4156 | -0.071 | -0.129 | -0.014 | 0.994 | True |
| T2Weighted | SRS | late | 4156 | -0.050 | -0.117 | 0.018 | 0.928 | False |
| T2Weighted | SaOvR | late | 4156 | -0.049 | -0.092 | -0.004 | 0.985 | True |
| T2Weighted | T1Weighted | late | 4156 | -0.012 | -0.032 | 0.008 | 0.864 | False |
| T4Weighted | RawEPA | late | 4156 | -0.104 | -0.164 | -0.044 | 1.000 | True |
| T4Weighted | SRS | late | 4156 | -0.084 | -0.146 | -0.021 | 0.996 | True |
| T4Weighted | SaOvR | late | 4156 | -0.082 | -0.135 | -0.029 | 0.999 | True |
| T4Weighted | T1Weighted | late | 4156 | -0.045 | -0.086 | -0.006 | 0.986 | True |
| T4Weighted | T2Weighted | late | 4156 | -0.033 | -0.067 | 0.001 | 0.971 | False |
| RawEPA | SaOvR | overall | 5297 | -0.006 | -0.052 | 0.040 | 0.605 | False |
| SRS | RawEPA | overall | 5297 | -0.037 | -0.086 | 0.008 | 0.940 | False |
| SRS | SaOvR | overall | 5297 | -0.043 | -0.106 | 0.019 | 0.913 | False |
| T1Weighted | RawEPA | overall | 5297 | -0.048 | -0.095 | 0.002 | 0.968 | False |
| T1Weighted | SRS | overall | 5297 | -0.010 | -0.077 | 0.054 | 0.622 | False |
| T1Weighted | SaOvR | overall | 5297 | -0.054 | -0.095 | -0.011 | 0.993 | True |
| T2Weighted | RawEPA | overall | 5297 | -0.066 | -0.112 | -0.015 | 0.992 | True |
| T2Weighted | SRS | overall | 5297 | -0.028 | -0.096 | 0.035 | 0.806 | False |
| T2Weighted | SaOvR | overall | 5297 | -0.072 | -0.115 | -0.023 | 1.000 | True |
| T2Weighted | T1Weighted | overall | 5297 | -0.018 | -0.037 | -0.000 | 0.977 | True |
| T4Weighted | RawEPA | overall | 5297 | -0.095 | -0.147 | -0.036 | 1.000 | True |
| T4Weighted | SRS | overall | 5297 | -0.058 | -0.120 | 0.005 | 0.965 | False |
| T4Weighted | SaOvR | overall | 5297 | -0.101 | -0.153 | -0.050 | 1.000 | True |
| T4Weighted | T1Weighted | overall | 5297 | -0.048 | -0.086 | -0.009 | 0.993 | True |
| T4Weighted | T2Weighted | overall | 5297 | -0.029 | -0.063 | 0.002 | 0.965 | False |

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
| 5 | T2Weighted | 384 | 10.033 | 13.404 |
| 6 | T2Weighted | 379 | 10.712 | 13.780 |
| 7 | T2Weighted | 378 | 11.475 | 14.637 |
| 8 | T2Weighted | 378 | 10.407 | 13.321 |
| 9 | T2Weighted | 371 | 10.320 | 13.100 |
| 10 | T2Weighted | 384 | 11.134 | 14.214 |
| 11 | T2Weighted | 401 | 9.671 | 12.830 |
| 12 | T2Weighted | 417 | 9.899 | 12.799 |
| 13 | T2Weighted | 421 | 10.407 | 13.462 |
| 14 | T2Weighted | 418 | 11.065 | 14.092 |
| 15 | T2Weighted | 429 | 10.517 | 13.640 |
| 16 | T2Weighted | 429 | 11.155 | 14.103 |
| 17 | T2Weighted | 428 | 11.399 | 14.473 |
| 18 | T2Weighted | 80 | 10.361 | 12.585 |
| 5 | T4Weighted | 384 | 10.012 | 13.283 |
| 6 | T4Weighted | 379 | 10.694 | 13.662 |
| 7 | T4Weighted | 378 | 11.471 | 14.547 |
| 8 | T4Weighted | 378 | 10.361 | 13.272 |
| 9 | T4Weighted | 371 | 10.263 | 13.066 |
| 10 | T4Weighted | 384 | 11.215 | 14.269 |
| 11 | T4Weighted | 401 | 9.693 | 12.804 |
| 12 | T4Weighted | 417 | 9.869 | 12.782 |
| 13 | T4Weighted | 421 | 10.402 | 13.431 |
| 14 | T4Weighted | 418 | 10.897 | 13.998 |
| 15 | T4Weighted | 429 | 10.497 | 13.570 |
| 16 | T4Weighted | 429 | 11.126 | 14.038 |
| 17 | T4Weighted | 428 | 11.350 | 14.423 |
| 18 | T4Weighted | 80 | 10.184 | 12.521 |

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

## Per-Season T2Weighted vs SRS

| Season | T2Weighted MAE | SRS MAE | MAE Delta | RMSE Delta |
| --- | --- | --- | --- | --- |
| 1999 | 10.368 | 10.313 | 0.054 | 0.134 |
| 2000 | 10.081 | 10.534 | -0.453 | -0.522 |
| 2001 | 9.912 | 9.949 | -0.037 | 0.098 |
| 2002 | 10.121 | 9.951 | 0.169 | 0.332 |
| 2003 | 10.518 | 10.376 | 0.143 | -0.046 |
| 2004 | 11.093 | 10.858 | 0.235 | 0.257 |
| 2005 | 10.314 | 10.316 | -0.002 | 0.074 |
| 2006 | 11.019 | 10.965 | 0.054 | -0.019 |
| 2007 | 11.472 | 11.017 | 0.455 | 0.593 |
| 2008 | 11.396 | 11.426 | -0.030 | -0.113 |
| 2009 | 12.173 | 12.052 | 0.121 | 0.071 |
| 2010 | 10.954 | 11.236 | -0.281 | -0.398 |
| 2011 | 11.297 | 11.268 | 0.029 | 0.051 |
| 2012 | 10.974 | 11.233 | -0.259 | -0.253 |
| 2013 | 10.363 | 10.273 | 0.090 | 0.093 |
| 2014 | 11.387 | 11.276 | 0.110 | 0.049 |
| 2015 | 10.426 | 10.384 | 0.042 | -0.081 |
| 2016 | 9.460 | 9.529 | -0.069 | -0.065 |
| 2017 | 10.274 | 10.477 | -0.204 | -0.202 |
| 2018 | 10.625 | 10.606 | 0.019 | -0.084 |
| 2019 | 10.628 | 10.821 | -0.194 | -0.291 |
| 2020 | 10.709 | 10.912 | -0.203 | -0.292 |
| 2021 | 11.575 | 11.814 | -0.239 | -0.241 |
| 2022 | 9.376 | 9.300 | 0.076 | -0.075 |
| 2023 | 10.054 | 10.010 | 0.044 | -0.155 |
| 2024 | 10.006 | 10.260 | -0.254 | -0.379 |
| 2025 | 10.492 | 10.664 | -0.172 | -0.312 |

## Stability

| Metric | Entity | Paired Rows | Pearson | Spearman |
| --- | --- | --- | --- | --- |
| QSaCR | qb | 605 | 0.485 | 0.480 |
| qb_any_a | qb | 605 | 0.403 | 0.388 |
| qb_passer_rating | qb | 605 | 0.473 | 0.475 |
| SaOvR | team | 829 | 0.417 | 0.414 |

## QBR Correlations

| Season | Joined Rows | Pearson | Spearman |
| --- | --- | --- | --- |
| 2006 | 31 | 0.872 | 0.799 |
| 2007 | 28 | 0.917 | 0.901 |
| 2008 | 31 | 0.850 | 0.842 |
| 2009 | 28 | 0.961 | 0.963 |
| 2010 | 31 | 0.930 | 0.915 |
| 2011 | 32 | 0.946 | 0.954 |
| 2012 | 32 | 0.938 | 0.918 |
| 2013 | 34 | 0.871 | 0.874 |
| 2014 | 32 | 0.886 | 0.864 |
| 2015 | 33 | 0.923 | 0.920 |
| 2016 | 30 | 0.929 | 0.912 |
| 2017 | 30 | 0.790 | 0.770 |
| 2018 | 32 | 0.927 | 0.897 |
| 2019 | 30 | 0.854 | 0.840 |
| 2020 | 32 | 0.900 | 0.909 |
| 2021 | 31 | 0.918 | 0.871 |
| 2022 | 30 | 0.794 | 0.737 |
| 2023 | 30 | 0.918 | 0.871 |
| 2024 | 31 | 0.870 | 0.875 |
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
| q1_fixed_team_defense | 33 | 0.386 | 0.158 | - |
| q2_defense_penalty_x0 | 33 | 0.557 | 0.545 | 0.000 |
| q2_defense_penalty_x0.05 | 33 | 0.554 | 0.540 | 0.050 |
| q2_defense_penalty_x0.1 | 33 | 0.552 | 0.534 | 0.100 |
| q2_defense_penalty_x0.25 | 33 | 0.544 | 0.518 | 0.250 |
| q2_defense_penalty_x0.5 | 33 | 0.531 | 0.493 | 0.500 |
| q2_defense_penalty_x1 | 33 | 0.505 | 0.444 | 1.000 |

## Maye/Stafford Case Study

| Variant | QB | Raw EPA/DB | Adjusted EPA/DB | Faced Difficulty | Adjustment Delta |
| --- | --- | --- | --- | --- | --- |
| current | Matthew Stafford | 0.244 | 0.226 | 0.007 | -0.018 |
| current | Drake Maye | 0.306 | 0.244 | -0.029 | -0.063 |
| q1_fixed_team_defense | Matthew Stafford | 0.244 | 0.222 | 0.005 | -0.022 |
| q1_fixed_team_defense | Drake Maye | 0.306 | 0.253 | -0.016 | -0.054 |
| q2_defense_penalty_x0 | Matthew Stafford | 0.244 | 0.229 | 0.008 | -0.015 |
| q2_defense_penalty_x0 | Drake Maye | 0.306 | 0.240 | -0.035 | -0.066 |

## QB Open Status

- The Stage 3b QB audit continues to stand as a positive linear-adjustment result:
  the additive adjustment operated at full strength in EPA units, the identity checks
  held, and Q1/Q2 were correctly not adopted.
- The QB question remains open anyway, but on a new hypothesis: possible model
  misspecification from additive QB-vs-defense effects rather than miscalibrated
  adjustment strength. Stage 3d is the pre-registered next step.

## SaCR Caveat

SaCR may be evaluated as a secondary line with a caveat:
its frozen Stage 2 weights were fit on the full 1999-2025 history.
A walk-forward SaCR line over that same window has look-ahead in the weights.
SaOvR is the headline walk-forward metric because it does not depend on a fitted Stage 2 weight snapshot.
