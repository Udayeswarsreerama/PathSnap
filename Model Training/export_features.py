"""Export Member 1 features for a list of trips -> one CSV per trip.

Usage: python3 export_features.py <split> <trip_dir> [<trip_dir> ...]
Writes /downloads/pathsnap/features_out/<split>_<trip_id>.csv
Skips trips missing any of the 4 files. Logs progress to stdout.
"""
import os, sys, time
import pandas as pd

PIPE = "/downloads/pathsnap/member1/pipeline"
sys.path.insert(0, PIPE)
from data_loader import load_trip
from preprocessing import preprocess_trajectory
from feature_engineering import extract_features

OUT = "/downloads/pathsnap/features_out"
os.makedirs(OUT, exist_ok=True)

split = sys.argv[1]
total_rows = 0
t0 = time.time()
for d in sys.argv[2:]:
    d = d.rstrip("/")
    tid = os.path.basename(d)
    need = [f"{tid}.{ext}" for ext in ("track", "nodes", "arcs", "route")]
    if not all(os.path.exists(os.path.join(d, f)) for f in need):
        print(f"SKIP {split}/{tid}: incomplete", flush=True)
        continue
    outp = os.path.join(OUT, f"{split}_{tid}.csv")
    if os.path.exists(outp):
        print(f"SKIP {split}/{tid}: already done", flush=True)
        continue
    trip = load_trip(d)
    gps = preprocess_trajectory(trip["track"])
    fdf = extract_features(gps, trip["nodes"], trip["arcs"], trip["route"], trip_id=tid)
    fdf["split"] = split
    fdf.to_csv(outp, index=False)
    total_rows += len(fdf)
    n_true = int(fdf["is_true"].sum()) if fdf["is_true"].notna().any() else 0
    pts = fdf["point_idx"].nunique()
    rec = n_true / pts if pts else 0
    print(f"{split}/{tid}: pts={pts} rows={len(fdf)} recall={rec:.3f} "
          f"elapsed={time.time()-t0:.0f}s", flush=True)
print(f"DONE split={split} rows={total_rows} total_elapsed={time.time()-t0:.0f}s", flush=True)
