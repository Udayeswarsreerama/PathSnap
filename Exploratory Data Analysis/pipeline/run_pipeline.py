"""Run the Member 1 pipeline on the dataset and produce real evaluation outputs.

Usage: python3 run_pipeline.py [n_trips]
  n_trips = how many train trips to fully process (default 12, for hackathon speed).

Outputs in ./pipeline_out:
  trip_statistics.csv      - one row per processed trip (preprocessing summary)
  candidate_metrics.csv    - candidate generation recall per processed trip
  features_sample.csv      - full feature dataset for the first processed trips
  fig_candidates.png       - candidate visualisation for one trip
  summary.json             - aggregate numbers
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import trip_dirs, load_trip
from preprocessing import preprocess_trajectory, trip_statistics
from candidate_generation import (evaluate_candidates, build_trip_index,
                                  generate_candidates, proxy_true_arc,
                                  SEARCH_RADIUS_M, TOP_K)
from feature_engineering import extract_features

# Paths: pass as argv, set env vars, or place the dataset in ./data
#   python3 run_pipeline.py [n_trips] [data_root] [out_dir]
DATA = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("PATHSNAP_DATA", "data")
OUT = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("PATHSNAP_OUT", "pipeline_out")
os.makedirs(OUT, exist_ok=True)

n_trips = int(sys.argv[1]) if len(sys.argv) > 1 else 12
dirs = trip_dirs(DATA, "train")[:n_trips]

# ---- radius tuning: recall per search radius on the first 3 trips ----
RADII = [10, 15, 20, 25, 30, 40, 50, 75]
radius_rows = []
for d in dirs[:3]:
    trip = load_trip(d)
    for rad in RADII:
        t_start = time.time()
        cm = evaluate_candidates(trip["track"], trip["nodes"], trip["arcs"], trip["route"],
                                 radius_m=rad, top_k=TOP_K)
        radius_rows.append({"trip_id": trip["id"], "radius_m": rad,
                            "candidate_recall": cm["candidate_recall"],
                            "avg_candidates": cm["avg_candidates"],
                            "zero_cand_points": cm["points_zero_candidates"],
                            "runtime_s": round(time.time() - t_start, 2)})
        print(f"radius sweep {trip['id']} r={rad}: recall={cm['candidate_recall']:.3f}", flush=True)
radius_df = pd.DataFrame(radius_rows)
radius_df.to_csv(os.path.join(OUT, "radius_tuning.csv"), index=False)

stats_rows, cand_rows = [], []
features_frames = []
t0 = time.time()

for d in dirs:
    trip = load_trip(d)
    tid = trip["id"]
    gps = preprocess_trajectory(trip["track"])
    stats_rows.append(trip_statistics(gps, tid))
    cm = evaluate_candidates(trip["track"], trip["nodes"], trip["arcs"], trip["route"])
    cm["trip_id"] = tid
    cand_rows.append(cm)
    if len(features_frames) < 3:  # full feature export for the first 3 trips
        fdf = extract_features(gps, trip["nodes"], trip["arcs"], trip["route"], trip_id=tid)
        features_frames.append(fdf)
    print(f"{tid}: pts={len(gps)} gaps={int(gps['gap_before'].sum())} "
          f"recall={cm['candidate_recall']:.3f} zero_cand={cm['points_zero_candidates']}",
          flush=True)

stats_df = pd.DataFrame(stats_rows)
cand_df = pd.DataFrame(cand_rows)
stats_df.to_csv(os.path.join(OUT, "trip_statistics.csv"), index=False)
cand_df.to_csv(os.path.join(OUT, "candidate_metrics.csv"), index=False)
feat_df = pd.concat(features_frames, ignore_index=True)
feat_df.to_csv(os.path.join(OUT, "features_sample.csv"), index=False)

# ---- candidate visualisation for one trip ----
trip = load_trip(dirs[0])
tid = trip["id"]
gps = preprocess_trajectory(trip["track"])
projector, grid = build_trip_index(trip["nodes"], trip["arcs"])
nodes, arcs = trip["nodes"], trip["arcs"]
fig, ax = plt.subplots(figsize=(9, 9))
step = max(1, len(arcs) // 40000)
for a, b in arcs[::step]:
    ax.plot([nodes[a, 0], nodes[b, 0]], [nodes[a, 1], nodes[b, 1]], "-", lw=0.3,
            color="lightgray", alpha=0.6)
# ground truth route
rt = []
for aid in trip["route"]:
    a, b = arcs[aid]; rt.append(nodes[a]); rt.append(nodes[b])
rt = np.array(rt)
ax.plot(rt[:, 0], rt[:, 1], "-", lw=1.8, color="black", alpha=0.8, label="ground-truth route")
# sample GPS points and their candidates
rng = np.random.default_rng(1)
sample = rng.choice(len(gps), size=120, replace=False)
shown_cands = set()
for i in sample:
    r = gps.iloc[i]
    cands = generate_candidates(r["lon"], r["lat"], nodes, arcs,
                                projector=projector, grid=grid)
    ax.plot(r["lon"], r["lat"], ".", ms=4, color="steelblue")
    for c in cands:
        aid = c["arc_id"]
        if aid in shown_cands:
            continue
        shown_cands.add(aid)
        a, b = arcs[aid]
        ax.plot([nodes[a, 0], nodes[b, 0]], [nodes[a, 1], nodes[b, 1]], "-", lw=2.0,
                color="darkorange", alpha=0.8)
ax.plot([], [], ".", color="steelblue", label="sampled GPS points")
ax.plot([], [], "-", color="darkorange", label=f"candidate arcs (r={SEARCH_RADIUS_M:.0f} m, k={TOP_K})")
ax.set_title(f"Candidate generation on trip {tid}: GPS points and their candidate road arcs")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.legend()
ax.set_aspect("equal", adjustable="datalim")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_candidates.png")); plt.close(fig)

radius_pivot = radius_df.groupby("radius_m")[["candidate_recall", "avg_candidates", "runtime_s"]].mean().round(4)
summary = {
    "n_trips_processed": len(dirs),
    "radius_tuning": radius_pivot.reset_index().to_dict("records"),
    "runtime_s": round(time.time() - t0, 1),
    "search_radius_m": SEARCH_RADIUS_M,
    "top_k": TOP_K,
    "candidate_recall_mean": float(cand_df["candidate_recall"].mean()),
    "candidate_recall_min": float(cand_df["candidate_recall"].min()),
    "candidate_recall_max": float(cand_df["candidate_recall"].max()),
    "avg_candidates_per_point": float(cand_df["avg_candidates"].mean()),
    "points_zero_candidates_total": int(cand_df["points_zero_candidates"].sum()),
    "points_total": int(cand_df["n_points"].sum()),
    "gaps_total": int(stats_df["n_gaps"].sum()),
    "speed_unreliable_total": int(stats_df["n_speed_unreliable"].sum()) if "n_speed_unreliable" in stats_df else None,
    "feature_rows_exported": len(feat_df),
    "feature_label_balance": float(feat_df["is_true"].mean()) if len(feat_df) else None,
}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
