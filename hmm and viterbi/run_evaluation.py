"""PathSnap Member 3 - run the full HMM/Viterbi evaluation.

Usage:
    python3 run_evaluation.py <data_root> [out_dir]

<data_root> is the folder containing train/ and student_test/ (the "Student
folder" of the full dataset, or the smaller subset - the code is identical
for both). Outputs go to ./member3_out by default:

    metrics_per_trip.csv      - all metrics for every evaluated trip
    metrics_summary.csv       - point-weighted comparison table
    fig_point_accuracy.png    - classifier-only vs Viterbi accuracy per trip
    fig_continuity.png        - route continuity comparison
    fig_param_sweep.png       - sigma/beta sensitivity on train trips
    map_<trip_id>.html        - interactive Folium maps for two test trips
    hmm_config.json           - the configuration used (copy of input/tuned)

Steps:
  1. train the baseline candidate scorer on train-trip features
     (replace with Member 2's model artifact when ready - see README)
  2. tune sigma/beta on a few train trips (decoder params are fitted on
     training data only; student-test is used only for final evaluation)
  3. decode every student_test trip, compute metrics, plots and maps
"""
import os
import sys
import time
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Member 1's pipeline lives next to this package; make it importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "member1", "pipeline"))

from data_loader import trip_dirs, load_trip
from feature_engineering import extract_features
from preprocessing import preprocess_trajectory

from hmm import HMMConfig
from candidate_scorer import build_scorer
from route_reconstruction import prepare_trip, decode_with_params
from evaluate_decoder import evaluate_trip, summarise, sweep_parameters
from visualize_routes import make_map

HERE = os.path.dirname(os.path.abspath(__file__))
N_TRAIN_FEATURES = 8   # train trips used to fit the baseline scorer
N_TUNE_TRIPS = 4       # train trips used for the sigma/beta sweep
N_MAPS = 2             # student_test trips that get Folium maps


def main():
    data_root = sys.argv[1] if len(sys.argv) > 1 else "../package/dataset"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "member3_out")
    os.makedirs(out_dir, exist_ok=True)
    config = HMMConfig.load(os.path.join(HERE, "hmm_config.json"))

    train_dirs = [d for d in trip_dirs(data_root, "train")
                  if os.path.basename(d) not in config.excluded_trips]
    test_dirs = trip_dirs(data_root, "student_test")
    print(f"[1/4] {len(train_dirs)} train trips, {len(test_dirs)} student_test trips")

    # ---- 1. fit the baseline scorer on train features ---------------------
    t0 = time.time()
    feat_frames = []
    for d in train_dirs[:N_TRAIN_FEATURES]:
        trip = load_trip(d)
        gps_df = preprocess_trajectory(trip["track"])
        feat_frames.append(extract_features(gps_df, trip["nodes"],
                                            trip["arcs"], trip["route"],
                                            trip_id=trip["id"]))
    train_features = pd.concat(feat_frames, ignore_index=True)
    scorer = build_scorer(config, train_features)
    print(f"    scorer fitted on {len(train_features)} rows "
          f"from {len(feat_frames)} trips ({time.time()-t0:.1f}s)")

    # ---- 2. tune sigma/beta on train trips --------------------------------
    t0 = time.time()
    tune_prepared = []
    for d in train_dirs[:N_TUNE_TRIPS]:
        tune_prepared.append(prepare_trip(load_trip(d), scorer, config))
    sigma_sweep = sweep_parameters(tune_prepared, config, "sigma_m",
                                   [5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0])
    beta_sweep = sweep_parameters(tune_prepared, config, "transition_beta_m",
                                  [2.5, 5.0, 10.0, 25.0, 50.0, 100.0])
    best_sigma = sigma_sweep.groupby("value")["viterbi_point_acc"] \
                            .mean().idxmax()
    best_beta = beta_sweep.groupby("value")["viterbi_point_acc"] \
                          .mean().idxmax()
    config.sigma_m = float(best_sigma)
    config.transition_beta_m = float(best_beta)
    config.save(os.path.join(out_dir, "hmm_config.json"))
    print(f"    tuned sigma_m={best_sigma}, beta_m={best_beta} "
          f"({time.time()-t0:.1f}s)")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for sweep, param, a, label in (
            (sigma_sweep, "sigma_m", ax[0], "sigma (m)"),
            (beta_sweep, "transition_beta_m", ax[1], "beta (m)")):
        g = sweep.groupby("value")[["viterbi_point_acc",
                                    "viterbi_continuity"]].mean()
        a.plot(g.index, g["viterbi_point_acc"], "o-", label="point accuracy")
        a.plot(g.index, g["viterbi_continuity"], "s--", label="continuity")
        a.set_xlabel(label); a.set_ylabel("score"); a.set_ylim(0, 1.02)
        a.legend(); a.grid(alpha=0.3)
    fig.suptitle("Decoder parameter sensitivity (train trips)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_param_sweep.png"), dpi=150)
    plt.close(fig)

    # ---- 3. decode all student_test trips ---------------------------------
    t0 = time.time()
    metrics = []
    prepared_tests = []
    for d in test_dirs:
        prepared = prepare_trip(load_trip(d), scorer, config)
        prepared["viterbi"] = decode_with_params(prepared, config)
        prepared_tests.append(prepared)
        metrics.append(evaluate_trip(prepared, config))
    per_trip, summary = summarise(metrics)
    per_trip.to_csv(os.path.join(out_dir, "metrics_per_trip.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)
    print(f"    decoded {len(prepared_tests)} test trips ({time.time()-t0:.1f}s)")
    print(summary.T.to_string(header=False))

    # ---- 4. comparison plots + maps ---------------------------------------
    x = np.arange(len(per_trip))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - 0.2, per_trip["clf_top1_point_acc"], 0.4,
           label="classifier-only")
    ax.bar(x + 0.2, per_trip["viterbi_point_acc"], 0.4,
           label="classifier + HMM/Viterbi")
    ax.set_xticks(x); ax.set_xticklabels(per_trip["trip_id"], rotation=45)
    ax.set_ylabel("point accuracy"); ax.set_ylim(0, 1.02)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_point_accuracy.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - 0.2, per_trip["clf_top1_continuity"], 0.4,
           label="classifier-only")
    ax.bar(x + 0.2, per_trip["viterbi_continuity"], 0.4,
           label="classifier + HMM/Viterbi")
    ax.set_xticks(x); ax.set_xticklabels(per_trip["trip_id"], rotation=45)
    ax.set_ylabel("route continuity"); ax.set_ylim(0, 1.02)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_continuity.png"), dpi=150)
    plt.close(fig)

    for prepared in prepared_tests[:N_MAPS]:
        make_map(prepared, os.path.join(out_dir, f"map_{prepared['trip_id']}.html"),
                 title=f"Trip {prepared['trip_id']}")
    print(f"    outputs written to {out_dir}")


if __name__ == "__main__":
    main()
