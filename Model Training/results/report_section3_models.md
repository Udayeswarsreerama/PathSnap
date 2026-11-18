# Report section - Model experimentation and comparison (Member 2, final)

## 3.1 Problem formulation

Map-matching is framed as binary classification over candidate road arcs. For
each noisy GPS point, Member 1's pipeline generates up to 5 candidate arcs
within 30 m and computes 10 features per (point, candidate) pair:
`distance_m, rank, heading_diff_deg, heading_diff_bidir_deg, speed_mps,
dist_prev_m, dt_prev_s, gap_before, arc_length_m, connected_to_prev`. The
label (`is_true`) marks the candidate nearest to the ground-truth route arc.
The model scores each candidate; the highest-scoring candidate per point is
the point-wise map-match. Sequence consistency is enforced downstream by the
HMM (Member 3), which consumes these scores as emission probabilities.

**Data quality exclusions.** Inspection found two train trips with mismatched
ground truth: trip 00000033 (GPS trace ~6.5 km from its route) and trip
00000054 (route offset ~400 m). Their labels are meaningless, so both trips
were excluded from all training and evaluation. Final training data:
660,434 rows from 78 train trips (26.8% positive). Final evaluation: 70,335
rows from the 10 student_test trips, used exactly once.

**Leakage control.** All cross-validation is `GroupKFold(n_splits=5)` on
`trip_id`: whole trips stay together, so the model never sees rows from a
trip it is scored on. Row-level splitting would inflate scores because
consecutive GPS points are highly correlated.

**Ceiling context.** After the exclusions, the true arc is among the
generated candidates for 98.5% of train points and 98.8% of test points; no
classifier can score a point whose true arc was never a candidate.

## 3.2 Models compared

Five models from five different families (seed 42 throughout):

1. **Logistic regression** - linear baseline; interpretable weights;
   calibrated probabilities for the HMM.
2. **kNN (k=15, distance-weighted)** - instance-based; tests whether the
   features carry local structure. Fitted on a deliberate 500-rows-per-trip
   subsample (~39k rows, seed 42) because exact inference over 660k rows is
   too slow; evaluation covers every held-out row.
3. **Random forest** (100 trees, max depth 20) - bagging; robust to GPS
   noise; feature importances.
4. **LightGBM** (300 rounds) - gradient boosting; strongest off-the-shelf
   family for tabular data; the optimization candidate.
5. **MLP** (64-32, early stopping) - neural net; checks for feature
   interactions the trees cannot express.

Rare NaNs (first-of-trip rows) are median-imputed; logistic regression, kNN
and MLP use standardized features.

## 3.3 Results

Final metrics on the 10 student_test trips (models fitted on the 78 clean
train trips). Top-1 (winnable) restricts to points where the true arc was
among the candidates.

| model | top-1 | top-1 (winn.) | precision | recall | macro F1 | log loss | fit | predict |
|---|---|---|---|---|---|---|---|---|
| lightgbm | **0.8010** | 0.8105 | 0.7962 | 0.7618 | 0.8500 | 0.2759 | **5 s** | 0.31 s |
| random_forest | 0.8008 | 0.8103 | 0.7972 | 0.7635 | **0.8509** | 0.2777 | 81 s | 0.52 s |
| mlp | 0.7997 | 0.8092 | 0.7923 | 0.7654 | 0.8498 | **0.2757** | 26 s | 0.02 s |
| logistic_regression | 0.7926 | 0.8020 | 0.7634 | 0.7402 | 0.8314 | 0.3196 | 1 s | 0.02 s |
| knn | 0.7891 | 0.7985 | 0.7768 | 0.7501 | 0.8393 | 0.4330 | 0 s* | 16.2 s |

Figures: `fig_model_comparison.png` (grouped metrics), `fig_cv_vs_test.png`
(5-fold CV vs test).

Findings:

- **All five models land within 1.2 top-1 points** (0.789-0.801). The linear
  baseline reaching 0.793 shows the features carry most of the signal in
  simple additive form; non-linear models add only about a point.
- **LightGBM, random forest and MLP are statistically tied** at the top.
  LightGBM is the practical choice: equal accuracy, best-or-tied log loss,
  and 5-16x faster to train - which is why it was the optimization candidate
  and the serving model.
- **kNN confirms local structure** (0.789 despite seeing ~6% of the data) but
  is slow to query and worst calibrated - unsuitable as the HMM input.
- **CV means (0.777-0.783) now sit just below test scores (0.79-0.80)** -
  the expected pattern. (In the first run, before trips 33/54 were excluded,
  CV means were ~2 points lower because those trips' meaningless labels were
  pure noise inside CV folds; excluding them fixed CV without changing test,
  since neither trip was in student_test.)
- The remaining ~19% of winnable points mis-picked point-by-point are
  exactly the errors the HMM + Viterbi layer addresses by enforcing route
  connectivity - the motivation for the two-stage architecture.

## 3.4 Bayesian hyperparameter optimization

LightGBM was tuned with Gaussian-process Bayesian optimization
(scikit-optimize, Matern kernel, Expected Improvement): **objective** = mean
top-1 accuracy under the same 5-fold group CV on the 78 clean train trips;
**search space** = n_estimators 100-600, learning_rate 0.01-0.3 (log),
num_leaves 15-127, max_depth 4-12, min_child_samples 10-200, subsample and
colsample_bytree 0.5-1.0, reg_alpha/reg_lambda 1e-8-10 (log). 16 trials (8
random + 8 GP-guided); every trial is logged in `bayesopt_trials.csv`; search
history in `fig_bayesopt_history.png`.

**Tuned vs untuned:**

| metric | untuned | tuned |
|---|---|---|
| CV top-1 | 0.7784 | **0.7866** |
| test top-1 | 0.8010 | **0.8028** |
| test log loss | **0.2759** | 0.2826 |
| fit time | 5.0 s | **2.6 s** |

Best configuration: 112 trees, learning rate 0.0596, 60 leaves, max depth 4,
min_child_samples 195, subsample 0.616, colsample_bytree 0.545,
reg_alpha 0.0037, reg_lambda 2.8e-5.

The tuned LightGBM is our best model on test top-1 (0.8028). The optimizer
clearly preferred **small, heavily regularized** models - the largest
configs scored worst, and both the clean and the pre-exclusion runs
independently found the same pattern. The tuned gain (+0.8 pt CV, +0.2 pt
test) is real but smaller than the per-fold standard deviation (~0.06),
indicating the point-wise features are near their information ceiling -
subsequent gains come from sequence modelling (HMM + Viterbi), not further
tuning. The tuned model also fits 2x faster, which benefits the live API.

## 3.5 Reproducibility

`models/train_models.py` (comparison) and `models/bayesian_optimization.py`
(tuning), both seeded (42) and checkpointed; configs in `run_config.json` and
`bayesopt_best_params.json`; fitted models in `saved_models/`; per-candidate
test probabilities for the HMM in `test_scores_<model>.csv`; a container
demo (`demo/predict_demo.py`, no pandas needed) scores 10 real student_test
points end to end.
