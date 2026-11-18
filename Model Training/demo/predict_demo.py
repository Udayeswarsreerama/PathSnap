"""PathSnap predict demo - runs in a plain Python container.

Needs: joblib, scikit-learn, lightgbm (numpy comes with them).
NO pandas required - the sample rows are read with the csv module.

Run:  python3 predict_demo.py
Files next to this script: lightgbm_tuned.joblib, sample_rows.csv

What it shows: for 10 real GPS points from the student_test trips, the model
scores every candidate road arc; the highest-scoring arc is the predicted
map-match. We print predicted vs true arc and the model's confidence.
"""
import csv
import os
import warnings

import joblib
import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "lightgbm_tuned.joblib")
CSV_PATH = os.path.join(HERE, "sample_rows.csv")

# The 10 input features, in training order (is_true is the label, never input)
FEATURES = ["distance_m", "rank", "heading_diff_deg", "heading_diff_bidir_deg",
            "speed_mps", "dist_prev_m", "dt_prev_s", "gap_before",
            "arc_length_m", "connected_to_prev"]


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def to_float(v):
    if v in ("", "None", "nan", "NaN"):
        return np.nan  # the pipeline's imputer handles these
    if v == "True":
        return 1.0
    if v == "False":
        return 0.0
    return float(v)


def main():
    model = joblib.load(MODEL_PATH)
    clf = model.named_steps["clf"]
    print("=" * 64)
    print("PathSnap map-matching demo - tuned LightGBM classifier")
    print("=" * 64)
    print(f"Loaded pipeline: SimpleImputer -> {clf.__class__.__name__}")
    p = clf.get_params()
    print("Key parameters: n_estimators={n_estimators}, "
          "learning_rate={learning_rate:.4f}, num_leaves={num_leaves}, "
          "max_depth={max_depth},".format(**p))
    print("                min_child_samples={min_child_samples}, "
          "subsample={subsample:.3f}, colsample_bytree={colsample_bytree:.3f}, "
          "reg_alpha={reg_alpha:.3f}".format(**p))

    rows = load_rows(CSV_PATH)
    X = np.array([[to_float(r[c]) for c in FEATURES] for r in rows])
    proba = model.predict_proba(X)[:, 1]  # P(candidate is the true arc)

    # group candidate scores per GPS point, highest score = predicted arc
    points = {}
    for r, p_ in zip(rows, proba):
        key = (r["trip_id"], int(r["point_idx"]))
        points.setdefault(key, []).append(
            (int(r["arc_id"]), p_, r["is_true"] == "True"))

    print(f"\nScored {len(rows)} candidate arcs over {len(points)} GPS points "
          "from student_test.\n")
    header = f"{'point':>6} | {'cands':>5} | {'predicted':>9} | {'true':>6} | {'conf':>5} | verdict"
    print(header)
    print("-" * len(header))
    n_correct = 0
    for (trip, pidx), cands in sorted(points.items()):
        cands.sort(key=lambda c: -c[1])
        pred_arc, conf, _ = cands[0]
        true = next((a for a, _, t in cands if t), None)
        ok = (pred_arc == true)
        n_correct += ok
        print(f"{pidx:>6} | {len(cands):>5} | {pred_arc:>9} | "
              f"{str(true):>6} | {conf:>5.2f} | {'CORRECT' if ok else 'miss'}")
    print("-" * len(header))
    print(f"Top-1 accuracy on this sample: "
          f"{n_correct}/{len(points)} = {n_correct/len(points):.1%}")
    print("\nNote: 'conf' is the model's probability for its top candidate.")
    print("These per-point probabilities are what the HMM/Viterbi layer turns")
    print("into a fully connected route.")


if __name__ == "__main__":
    main()
