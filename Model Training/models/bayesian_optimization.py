"""PathSnap Member 2 - Bayesian hyperparameter optimization of LightGBM.

Gaussian-process Bayesian optimization (scikit-optimize, Matern kernel,
Expected Improvement) over 9 LightGBM hyperparameters. Objective: mean top-1
per-point accuracy under the SAME 5-fold GroupKFold-by-trip split used for
the model comparison, on the 80 train trips only. student_test is touched
once, at the end, for the untuned vs tuned comparison.

Budget (one-day hackathon): 16 trials (8 random initial + 8 GP-guided) over a
bounded space capped at 600 trees / 127 leaves so a trial stays under ~90s.

Every trial is appended to results/bayesopt_trials.csv the moment it
finishes; on restart the script replays the log into a fresh optimizer
(same seed -> identical ask sequence) and resumes where it stopped.

Usage: python3 bayesian_optimization.py [n_trials]   (default 16)
"""
import glob, json, os, sys, time
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from skopt import Optimizer
from skopt.space import Integer, Real

SEED = 42
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 16
FEATURES = ["distance_m", "rank", "heading_diff_deg", "heading_diff_bidir_deg",
            "speed_mps", "dist_prev_m", "dt_prev_s", "gap_before",
            "arc_length_m", "connected_to_prev"]
LABEL = "is_true"

BASE = "/downloads/pathsnap/member2_clean"
RES = os.path.join(BASE, "results")
MOD = os.path.join(BASE, "saved_models")
os.makedirs(RES, exist_ok=True)
os.makedirs(MOD, exist_ok=True)
LOG = os.path.join(RES, "bayesopt_trials.csv")

SPACE = [
    Integer(100, 600, name="n_estimators"),
    Real(0.01, 0.3, prior="log-uniform", name="learning_rate"),
    Integer(15, 127, name="num_leaves"),
    Integer(4, 12, name="max_depth"),
    Integer(10, 200, name="min_child_samples"),
    Real(0.5, 1.0, name="subsample"),
    Real(0.5, 1.0, name="colsample_bytree"),
    Real(1e-8, 10.0, prior="log-uniform", name="reg_alpha"),
    Real(1e-8, 10.0, prior="log-uniform", name="reg_lambda"),
]


def load(split):
    files = sorted(glob.glob(f"/downloads/pathsnap/features_out_clean/{split}_*.csv"))
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df["gap_before"] = df["gap_before"].astype(np.int8)
    df["connected_to_prev"] = df["connected_to_prev"].astype(np.int8)
    df[LABEL] = df[LABEL].astype(bool)
    return df


def make_model(params):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", LGBMClassifier(n_jobs=2, random_state=SEED, verbose=-1,
                               **params)),
    ])


def top1(ev_df, proba):
    ev = ev_df[["trip_id", "point_idx", LABEL]].copy()
    ev["score"] = proba
    top = ev.sort_values("score", ascending=False).drop_duplicates(
        ["trip_id", "point_idx"])
    return float(top[LABEL].mean())


def cv_score(train_df, params):
    """Mean top-1 over the same 5 group folds as the model comparison."""
    gkf = GroupKFold(n_splits=5)
    top1s, losses = [], []
    t0 = time.perf_counter()
    for itr, iev in gkf.split(train_df, groups=train_df["trip_id"]):
        ftr, fev = train_df.iloc[itr], train_df.iloc[iev]
        model = make_model(params)
        model.fit(ftr[FEATURES], ftr[LABEL])
        proba = model.predict_proba(fev[FEATURES])[:, 1]
        top1s.append(top1(fev, proba))
        losses.append(log_loss(fev[LABEL], np.clip(proba, 1e-7, 1 - 1e-7)))
    return float(np.mean(top1s)), float(np.std(top1s)), float(np.mean(losses)), \
        time.perf_counter() - t0


def main():
    train_df, test_df = load("train"), load("student_test")
    done = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame()
    opt = Optimizer(SPACE, random_state=SEED, n_initial_points=8,
                    acq_func="EI", base_estimator="GP")
    if len(done):  # resume: replay the log (same seed -> same ask sequence)
        names = [d.name for d in SPACE]
        opt.tell(done[names].values.tolist(), (-done["cv_top1_mean"]).tolist())
        print(f"resumed with {len(done)} logged trials", flush=True)

    for i in range(len(done), N_TRIALS):
        x = opt.ask()
        params = dict(zip([d.name for d in SPACE], x))
        m, s, ll, secs = cv_score(train_df, params)
        opt.tell(x, -m)  # maximize top-1
        row = {"trial": i, **params, "cv_top1_mean": round(m, 5),
               "cv_top1_std": round(s, 5), "cv_log_loss": round(ll, 5),
               "eval_time_s": round(secs, 1)}
        pd.DataFrame([row]).to_csv(LOG, mode="a",
                                   header=not os.path.exists(LOG), index=False)
        best = -min(opt.yi)
        print(f"trial {i}: top1={m:.4f} logloss={ll:.4f} ({secs:.0f}s) "
              f"best_so_far={best:.4f}", flush=True)

    # ---- best config: refit on all train, evaluate on student_test ----
    trials = pd.read_csv(LOG)
    names = [d.name for d in SPACE]
    best_row = trials.loc[trials.cv_top1_mean.idxmax()]
    best_params = {n: best_row[n] for n in names}
    for n in ("n_estimators", "num_leaves", "max_depth", "min_child_samples"):
        best_params[n] = int(best_params[n])
    print("BEST", json.dumps(best_params), flush=True)

    model = make_model(best_params)
    t0 = time.perf_counter()
    model.fit(train_df[FEATURES], train_df[LABEL])
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = model.predict_proba(test_df[FEATURES])[:, 1]
    inf_s = time.perf_counter() - t0
    y = test_df[LABEL].to_numpy()
    pred = proba >= 0.5
    tuned = {
        "model": "lightgbm_tuned",
        "top1_accuracy": top1(test_df, proba),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "log_loss": log_loss(y, np.clip(proba, 1e-7, 1 - 1e-7)),
        "cv_top1_mean": float(best_row.cv_top1_mean),
        "train_time_s": round(fit_s, 2), "infer_time_s": round(inf_s, 2),
        "n_fit_rows": len(train_df),
    }
    pd.DataFrame([tuned]).to_csv(os.path.join(RES, "tuned_result.csv"),
                                 index=False)
    with open(os.path.join(RES, "bayesopt_best_params.json"), "w") as f:
        json.dump({"best_params": best_params,
                   "cv_top1_mean": float(best_row.cv_top1_mean),
                   "n_trials": len(trials)}, f, indent=2)
    joblib.dump(model, os.path.join(MOD, "lightgbm_tuned.joblib"))
    out = test_df[["trip_id", "point_idx", "arc_id", LABEL]].copy()
    out["score"] = proba
    out.to_csv(os.path.join(RES, "test_scores_lightgbm_tuned.csv"),
               index=False)
    print("FINAL tuned:", {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in tuned.items()}, flush=True)
    print("BAYESOPT DONE", flush=True)


if __name__ == "__main__":
    main()
