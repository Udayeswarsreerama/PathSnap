"""PathSnap Member 2 - model training and comparison.

Trains 5 models from different families on Member 1's per-(point, candidate)
feature rows, with group-aware 5-fold CV by trip (GroupKFold on trip_id) and a
final evaluation on the 10 student_test trips. All seeds fixed and logged.

Usage: python3 train_models.py
Inputs:  /downloads/pathsnap/features_out_clean/{train,student_test}_*.csv
Outputs: /downloads/pathsnap/member2_clean/results/ + /downloads/pathsnap/member2_clean/saved_models/

kNN note: exact kNN inference over 695k training rows is too slow for a one-day
budget, so kNN fits on a capped random sample of 500 rows per trip (~40k rows,
seed 42) and is still EVALUATED on every held-out row. Everything else trains
on full folds.
"""
import glob, json, os, time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                             precision_score, recall_score)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

SEED = 42
FEATURES = ["distance_m", "rank", "heading_diff_deg", "heading_diff_bidir_deg",
            "speed_mps", "dist_prev_m", "dt_prev_s", "gap_before",
            "arc_length_m", "connected_to_prev"]
LABEL = "is_true"
KNN_ROWS_PER_TRIP = 500

BASE = "/downloads/pathsnap/member2_clean"
RES = os.path.join(BASE, "results")
MOD = os.path.join(BASE, "saved_models")
os.makedirs(RES, exist_ok=True)
os.makedirs(MOD, exist_ok=True)


def load(split):
    files = sorted(glob.glob(f"/downloads/pathsnap/features_out_clean/{split}_*.csv"))
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df["gap_before"] = df["gap_before"].astype(np.int8)
    df["connected_to_prev"] = df["connected_to_prev"].astype(np.int8)
    df[LABEL] = df[LABEL].astype(bool)
    return df


def make_models():
    return {
        "logistic_regression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]),
        "knn": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance",
                                         algorithm="ball_tree", n_jobs=2)),
        ]),
        "random_forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=20,
                                           min_samples_leaf=2, n_jobs=2,
                                           random_state=SEED)),
        ]),
        "lightgbm": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", LGBMClassifier(n_estimators=300, learning_rate=0.08,
                                   num_leaves=63, n_jobs=2, random_state=SEED,
                                   verbose=-1)),
        ]),
        "mlp": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                                  batch_size=1024, early_stopping=True,
                                  validation_fraction=0.1, max_iter=60,
                                  random_state=SEED)),
        ]),
    }


def subsample_knn(df):
    parts = []
    for _, g in df.groupby("trip_id"):
        parts.append(g.sample(n=min(len(g), KNN_ROWS_PER_TRIP),
                              random_state=SEED))
    return pd.concat(parts, ignore_index=True)


def evaluate(model, ev_df, fit_time):
    X = ev_df[FEATURES]
    y = ev_df[LABEL].to_numpy()
    t0 = time.perf_counter()
    proba = model.predict_proba(X)[:, 1]
    infer_time = time.perf_counter() - t0
    pred = (proba >= 0.5)
    # Top-1 per GPS point: highest-scoring candidate vs proxy true arc
    ev = ev_df[["trip_id", "point_idx", LABEL]].copy()
    ev["score"] = proba
    top = ev.sort_values("score", ascending=False).drop_duplicates(
        ["trip_id", "point_idx"])
    winnable = ev.groupby(["trip_id", "point_idx"])[LABEL].max()
    top = top.set_index(["trip_id", "point_idx"])
    top1 = top[LABEL].mean()
    top1_win = top.loc[winnable[winnable].index, LABEL].mean()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "log_loss": log_loss(y, np.clip(proba, 1e-7, 1 - 1e-7)),
        "top1_accuracy": top1,
        "top1_accuracy_winnable": top1_win,
        "winnable_frac": winnable.mean(),
        "train_time_s": round(fit_time, 2),
        "infer_time_s": round(infer_time, 2),
        "n_eval_rows": len(ev_df),
    }, proba


def main():
    t_start = time.perf_counter()
    train_df, test_df = load("train"), load("student_test")
    print(f"train {len(train_df)} rows / {train_df.trip_id.nunique()} trips, "
          f"test {len(test_df)} rows / {test_df.trip_id.nunique()} trips", flush=True)

    cv_path = os.path.join(RES, "cv_results.csv")
    fin_path = os.path.join(RES, "model_comparison.csv")
    done_cv = set()
    if os.path.exists(cv_path):
        prev = pd.read_csv(cv_path)
        done_cv = set(zip(prev.model, prev.fold))
    done_fin = set()
    if os.path.exists(fin_path):
        done_fin = set(pd.read_csv(fin_path).model)

    gkf = GroupKFold(n_splits=5)

    for name in make_models():
        # ---- group-aware 5-fold CV on train trips ----
        fold_metrics = []
        for fold, (itr, iev) in enumerate(gkf.split(
                train_df, groups=train_df["trip_id"])):
            if (name, fold) in done_cv:
                continue
            ftr, fev = train_df.iloc[itr], train_df.iloc[iev]
            fit_df = subsample_knn(ftr) if name == "knn" else ftr
            model = make_models()[name]
            t0 = time.perf_counter()
            model.fit(fit_df[FEATURES], fit_df[LABEL])
            fit_time = time.perf_counter() - t0
            m, _ = evaluate(model, fev, fit_time)
            m.update(model=name, fold=fold, n_fit_rows=len(fit_df))
            pd.DataFrame([m]).to_csv(cv_path, mode="a",
                                     header=not os.path.exists(cv_path),
                                     index=False)
            print(f"CV {name} fold {fold}: top1={m['top1_accuracy']:.4f} "
                  f"f1={m['f1_macro']:.4f} fit={fit_time:.0f}s", flush=True)

        # ---- final: fit on all train, evaluate on student_test ----
        if name in done_fin:
            continue
        cv_done = pd.read_csv(cv_path)
        fm = cv_done[cv_done.model == name]
        fit_df = subsample_knn(train_df) if name == "knn" else train_df
        model = make_models()[name]
        t0 = time.perf_counter()
        model.fit(fit_df[FEATURES], fit_df[LABEL])
        fit_time = time.perf_counter() - t0
        m, proba = evaluate(model, test_df, fit_time)
        m.update(model=name, n_fit_rows=len(fit_df),
                 cv_top1_mean=fm.top1_accuracy.mean(),
                 cv_top1_std=fm.top1_accuracy.std(),
                 cv_f1_macro_mean=fm.f1_macro.mean(),
                 cv_log_loss_mean=fm.log_loss.mean())
        pd.DataFrame([m]).to_csv(fin_path, mode="a",
                                 header=not os.path.exists(fin_path),
                                 index=False)
        print(f"FINAL {name}: top1={m['top1_accuracy']:.4f} "
              f"winnable={m['top1_accuracy_winnable']:.4f} "
              f"f1={m['f1_macro']:.4f} logloss={m['log_loss']:.4f} "
              f"fit={fit_time:.0f}s", flush=True)
        joblib.dump(model, os.path.join(MOD, f"{name}.joblib"))
        # per-candidate probabilities for Member 3's HMM
        out = test_df[["trip_id", "point_idx", "arc_id", LABEL]].copy()
        out["score"] = proba
        out.to_csv(os.path.join(RES, f"test_scores_{name}.csv"), index=False)

    meta = {"seed": SEED, "knn_rows_per_trip": KNN_ROWS_PER_TRIP,
            "features": FEATURES, "cv": "GroupKFold(n_splits=5) on trip_id",
            "train_rows": len(train_df), "test_rows": len(test_df),
            "total_runtime_s": round(time.perf_counter() - t_start, 1)}
    with open(os.path.join(RES, "run_config.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"ALL DONE in {meta['total_runtime_s']}s", flush=True)


if __name__ == "__main__":
    main()
