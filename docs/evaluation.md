# Evaluation definitions

## Paired allocations

The three-year horizon is 1,095 days. Patients are sorted by their fixed input identifiers. For cohort index `c` and repeat `r`, the random generator uses seed `2026101601 + 100000*c + r`. Calibration samples are nested prefixes of the resulting permutation. SCAN-B and METABRIC use 25, 50, 100, and 200 patients; GSE42568 uses 25 and 50.

The primary evaluation uses every patient outside the calibration prefix. The common-evaluation sensitivity uses the suffix after the largest calibration prefix, giving 455, 1,778, and 54 patients. Predictors and updating strategies share allocations.

## Updating strategies

For fixed risk `p`, let `x = log(-log(1-p))`. Updating produces `q = 1-exp(-exp(a+b*x))`.

| Strategy | Parameters |
|---|---|
| Unchanged | Original `p` |
| Ordinary intercept | Fit `a`, fix `b=1` |
| Penalized intercept | Fit `a` with summed IPCW loss plus `kappa*a*a/2`, fix `b=1` |
| Intercept and slope | Fit both `a` and `b` |

The main penalty is `kappa=5`; sensitivities use 2.5 and 10 and a nonnegative slope. Missing outcome classes, separation, and non-finite two-parameter fits use the declared ordinary-intercept fallback. Censoring weights are estimated separately within the calibration and evaluation samples.

## Reported metrics

`delta = Brier(updated) - Brier(unchanged)` is calculated on the same evaluation patients. A negative value indicates improvement. Worsening frequency is the fraction of allocations with `delta > 0`. The upper-decile mean is the average of the 30 largest signed differences among 300 allocations; negative values are retained.

Cohort displays first average the three predictor-specific scores within an allocation. They summarize prediction errors, rather than averaging patient risks into an ensemble. Figure 2 plots `-1000*delta`; Figure 3 plots `1000*delta`. Their opposite signs follow the stated axis labels.

## Calibration-only selection

At the main sizes, 100 patient bootstraps fit each candidate and generate out-of-bag predictions. Patient–replicate records are pooled. The second stage draws 1,000 paired samples of records from that pool, each of calibration-sample size. Repeated records from a patient are retained; they are not averaged or treated as a patient-level confidence interval.

The minimum-error rule chooses the lowest median resampled Brier score. The preference rule checks earlier candidates in the order unchanged, penalized, ordinary, choosing the first whose paired 2.5th–97.5th percentile interval includes zero. These intervals are operational selection thresholds. Selection uses calibration data; evaluation follows the persisted decisions. The exact seed structure and weighting are recorded in `selection/selection_protocol.json`.

## Pilot and sensitivity checks

The first 20 allocations preceded extension to 300. Continuation required penalized-minus-unchanged mean Brier at most −0.001 in METABRIC and at most 0.001 in SCAN-B at the main sizes, plus lower mean Brier than ordinary updating in at least two cohorts. The original numerical gate is preserved in `benchmark/pilot_gate.json`.

`analysis/sensitivity_tables.py` recomputes those conditions and excludes the first 20 allocations. The resulting 280-allocation upper-decile summary uses 28 differences. The same patients recur across allocations; these summaries describe the fixed cohorts and predictors.
