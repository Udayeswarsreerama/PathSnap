# Bayesian hyperparameter optimization - LightGBM (final, clean data)

Rubric checklist: search space stated below, objective stated, group-aware CV
on the 78 clean training trips, every trial logged (`bayesopt_trials.csv`),
tuned vs untuned comparison at the end. Seed 42 everywhere.

**Data note:** trips 00000033 and 00000054 were excluded after Member 1 found
their ground truth is mismatched (GPS trace ~6.5 km from its route on trip
33; route offset ~400 m on trip 54). All numbers below are from the clean
78-trip set (660,434 rows). The earlier 80-trip run is kept for reference in
the v2 outputs.

## Objective

Maximize **mean top-1 per-point accuracy** under 5-fold `GroupKFold` on
`trip_id` over the 78 clean train trips. Top-1 = for each GPS point, the
highest-scoring candidate arc is correct iff it is the proxy true arc.
`student_test` was NOT used anywhere in the search - only for the final
tuned vs untuned comparison.

## Method

Gaussian-process Bayesian optimization (scikit-optimize `Optimizer`, Matern
kernel surrogate, Expected Improvement acquisition), 16 trials: 8 random
initial points + 8 GP-guided. Budget chosen so one trial stays under ~90 s.

## Search space

| parameter | range | prior |
|---|---|---|
| n_estimators | 100-600 | uniform |
| learning_rate | 0.01-0.3 | log-uniform |
| num_leaves | 15-127 | uniform |
| max_depth | 4-12 | uniform |
| min_child_samples | 10-200 | uniform |
| subsample | 0.5-1.0 | uniform |
| colsample_bytree | 0.5-1.0 | uniform |
| reg_alpha | 1e-8-10 | log-uniform |
| reg_lambda | 1e-8-10 | log-uniform |

## Trial log

All 16 trials with full parameters, CV top-1 mean/std, CV log loss and
wall-clock time: `results/bayesopt_trials.csv`. Search history:
`results/fig_bayesopt_history.png`.

## Best configuration (trial 2)

```
n_estimators=112, learning_rate=0.0596, num_leaves=60, max_depth=4,
min_child_samples=195, subsample=0.616, colsample_bytree=0.545,
reg_alpha=0.0037, reg_lambda=2.8e-5
```

## Tuned vs untuned (untuned = comparison-table LightGBM: 300 trees,
lr 0.08, 63 leaves, no explicit regularization)

| metric | untuned | tuned | delta |
|---|---|---|---|
| CV top-1 (5-fold, by trip) | 0.7784 | **0.7866** | +0.0082 |
| test top-1 (student_test) | 0.8010 | **0.8028** | +0.0018 |
| test macro F1 | 0.8500 | 0.8471 | -0.0029 |
| test log loss | **0.2759** | 0.2826 | +0.0067 |
| fit time (full train) | 5.0 s | **2.6 s** | -48% |

The tuned LightGBM is the best model we produced on test top-1 (0.8028),
edging out random forest (0.8008) and MLP (0.7997) at a fraction of their
training cost.

## What the search found (viva notes)

1. **The optimizer again preferred small, heavily regularized models** -
   112 trees, depth 4, min_child_samples 195, aggressive row/column
   subsampling. The biggest, deepest configs scored worst. With ~80 noisy
   GPS trips, variance reduction beats raw capacity. (The contaminated-data
   run independently found the same pattern, which strengthens the point.)
2. **Learning rate / n_estimators trade-off**: the good region is ~0.06 lr
   with ~110 trees; slow-learning high-tree corners were explored early and
   abandoned by the surrogate.
3. **The gain is real but modest** (+0.8 pt CV, +0.2 pt test), below the
   per-fold standard deviation (~0.06). The point-wise features are near
   their information ceiling - further gains come from sequence structure
   (Member 3's HMM + Viterbi), not more tuning.
4. **Clean data changed CV, not test.** Removing trips 33/54 raised every
   model's CV mean by ~2 points (0.759 -> 0.778 for LightGBM) because those
   trips' meaningless labels were pure noise in the CV folds; test scores
   moved <0.5 pt since trips 33/54 were never in student_test.

Artifacts: `bayesopt_trials.csv`, `bayesopt_best_params.json`,
`tuned_result.csv`, `test_scores_lightgbm_tuned.csv` (HMM emission inputs),
`saved_models/lightgbm_tuned.joblib`, `fig_bayesopt_history.png`.
