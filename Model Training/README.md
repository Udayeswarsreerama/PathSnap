# Member 2 - Model training and comparison

Five models from different families, trained on Member 1's per-(GPS point,
candidate arc) feature rows, with group-aware cross-validation and a final
held-out evaluation.

## Layout

- `models/train_models.py` - trains all 5 models, runs the CV and the final
  evaluation, saves models + per-candidate test scores. Checkpoints after
  every fold, so a restart resumes where it stopped.
- `evaluation/model_comparison.py` - builds the comparison figures and prints
  the final table from the results CSVs.
- `results/` - all measured outputs (see below).
- `saved_models/` - fitted models (`joblib`), fitted on all 80 train trips.

## Method

- **Features** (Member 1's schema): `distance_m, rank, heading_diff_deg,
  heading_diff_bidir_deg, speed_mps, dist_prev_m, dt_prev_s, gap_before,
  arc_length_m, connected_to_prev`. `is_true` is the label only, never an
  input. Median imputation for the rare NaNs (317 of 695k rows, first points
  of trips with no heading/speed yet).
- **Split**: `GroupKFold(n_splits=5)` on `trip_id` over the 80 train trips -
  whole trips always stay together, no row-level leakage. Final evaluation on
  the 10 `student_test` trips, used only after all training decisions.
- **Models** (seed 42 everywhere):
  logistic regression, kNN (k=15, distance-weighted), random forest
  (100 trees, max depth 20), LightGBM (300 rounds), MLP (64-32, early stop).
- **kNN subsampling**: exact kNN over 695k training rows is too slow for the
  one-day budget, so kNN fits on a capped random sample of 500 rows per trip
  (~40k rows, seed 42). Evaluation still covers every held-out row.
  All other models train on full folds.
- **Top-1 accuracy**: for each GPS point, the candidate with the highest
  predicted probability is the model's map-match; correct iff it is the proxy
  true arc. `top1_accuracy_winnable` restricts to points where the true arc
  was among the candidates at all (~96% of train points, ~99% of test
  points - the rest are candidate-generation misses no model can fix).

## Results files

- `results/model_comparison.csv` - final student-test metrics + CV means.
- `results/cv_results.csv` - per-fold metrics for every model.
- `results/fig_model_comparison.png` - grouped bar chart of test metrics.
- `results/fig_cv_vs_test.png` - CV vs test top-1 (generalisation check).
- `results/test_scores_<model>.csv` - per-candidate probabilities on
  student_test (Member 3's HMM emission inputs).
- `results/run_config.json` - seeds, feature list, CV scheme, runtimes.
- `results/model_notes.md` - per-model interpretation.

## Reproduce

```
python3 models/train_models.py        # ~15-20 min on 2 cores
python3 evaluation/model_comparison.py
```
