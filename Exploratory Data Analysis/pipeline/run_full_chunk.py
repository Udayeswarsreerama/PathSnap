"""Chunked full-dataset run: python3 run_full_chunk.py START END [data_root] [out_dir]
Processes trips [START, END) of the ordered list (80 train + 10 test),
writes chunk outputs to <out_dir>/chunk_START_END_*.csv.
Paths default to env vars PATHSNAP_DATA / PATHSNAP_OUT, then ./data and
./pipeline_out_full relative to the current directory."""
import os, sys, json, time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import trip_dirs, load_trip
from preprocessing import preprocess_trajectory, trip_statistics
from candidate_generation import evaluate_candidates
from feature_engineering import extract_features

DATA = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("PATHSNAP_DATA", "data")
OUT = sys.argv[4] if len(sys.argv) > 4 else os.environ.get("PATHSNAP_OUT", "pipeline_out_full")
os.makedirs(OUT, exist_ok=True)

start, end = int(sys.argv[1]), int(sys.argv[2])
all_dirs = [(d, "train") for d in trip_dirs(DATA, "train")] + \
           [(d, "student_test") for d in trip_dirs(DATA, "student_test")]
chunk = all_dirs[start:end]

stats_rows, cand_rows, feat_parts = [], [], []
t0 = time.time()
for d, split in chunk:
    trip = load_trip(d)
    tid = trip["id"]
    gps = preprocess_trajectory(trip["track"])
    st = trip_statistics(gps, tid); st["split"] = split
    stats_rows.append(st)
    if trip["route"] is not None:
        cm = evaluate_candidates(trip["track"], trip["nodes"], trip["arcs"], trip["route"])
        cm["trip_id"] = tid; cm["split"] = split
        cand_rows.append(cm)
        rec = f"{cm['candidate_recall']:.3f}"
    else:
        rec = "n/a"
    fdf = extract_features(gps, trip["nodes"], trip["arcs"], trip["route"], trip_id=tid)
    fdf["split"] = split
    feat_parts.append(fdf)
    print(f"{split}/{tid}: pts={len(gps)} recall={rec} feat_rows={len(fdf)} "
          f"elapsed={time.time()-t0:.0f}s", flush=True)

tag = f"chunk_{start}_{end}"
pd.DataFrame(stats_rows).to_csv(os.path.join(OUT, f"{tag}_stats.csv"), index=False)
pd.DataFrame(cand_rows).to_csv(os.path.join(OUT, f"{tag}_cand.csv"), index=False)
pd.concat(feat_parts, ignore_index=True).to_csv(os.path.join(OUT, f"{tag}_features.csv"), index=False)
print(f"CHUNK {start}-{end} DONE in {time.time()-t0:.0f}s")
