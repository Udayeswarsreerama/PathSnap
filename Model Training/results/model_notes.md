# Member 2 - Model comparison notes (final, clean data)

Final run: trips 00000033 and 00000054 excluded (mismatched ground truth -
see bayesian_optimization.md data note). Training: 660,434 rows / 78 trips.
Evaluation: 70,335 rows / 10 student_test trips, touched once. Seed 42.
Earlier 80-trip numbers preserved in the v2 outputs.

## Headline (student_test, fitted on 78 clean train trips)

| model | top-1 | top-1 (winnable) | macro F1 | log loss | fit | predict |
|---|---|---|---|---|---|---|
| lightgbm (untuned) | 0.8010 | 0.8105 | 0.8500 | 0.2759 | 5 s | 0.31 s |
| random_forest | 0.8008 | 0.8103 | 0.8509 | 0.2777 | 81 s | 0.52 s |
| mlp | 0.7997 | 0.8092 | 0.8498 | 0.2757 | 26 s | 0.02 s |
| logistic_regression | 0.7926 | 0.8020 | 0.8314 | 0.3196 | 1 s | 0.02 s |
| knn | 0.7891 | 0.7985 | 0.8393 | 0.4330 | 0 s* | 16.2 s |
| **lightgbm_tuned** | **0.8028** | - | 0.8471 | 0.2826 | 2.6 s | 0.12 s |

*kNN: 500-rows-per-trip subsample (~39k rows), full-row evaluation.

## Old (contaminated) vs new (clean)

- Test top-1 barely moved (<0.5 pt per model): trips 33/54 were never in
  student_test, and their ~35k mislabeled rows were only ~5% of training.
- CV means jumped ~2 points (LightGBM 0.7589 -> 0.7784): trips 33/54 were
  pure label noise inside CV folds. CV now sits just below test, the normal
  pattern.
- Rankings are stable: boosting/forest/MLP tied at top, linear baseline
  ~1 pt behind, kNN last and slowest.
- The viva story got cleaner: "we caught two mislabeled trips because their
  candidate recall was 30%/76% - 3+ sigma below the 98.5% median - verified
  with Member 1 that the ground truth was offset, and excluded them."

## Caveats

1. kNN subsample as before (documented in run_config.json).
2. Labels are Member 1's nearest-route-arc proxy.
3. Median imputation for rare NaN first-of-trip rows.
4. student_test used once, final evaluation only.
5. Tuned LightGBM's log loss (0.2826) is slightly worse than untuned
   (0.2759) - the objective was top-1, not calibration. If the HMM proves
   sensitive to calibration, re-tune with a log-loss objective or calibrate
   (Platt/isotonic) on a held-out trip fold.

Everything else: see report_section3_models.md (team-PDF section) and
bayesian_optimization.md (full opt writeup).
