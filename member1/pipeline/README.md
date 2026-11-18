# Member 1 - Data pipeline: preprocessing, candidate generation, feature engineering

Modules (import these, do not copy code):

| Function | Module | Who uses it |
|---|---|---|
| `load_dataset(data_root)` / `load_trip(dir)` | `data_loader.py` | everyone |
| `preprocess_trajectory(track)` | `preprocessing.py` | Members 2, 3, 4 |
| `build_trip_index(nodes, arcs)` + `generate_candidates(lon, lat, nodes, arcs, projector, grid)` | `candidate_generation.py` | Members 2, 3, 4 |
| `extract_features(gps_df, nodes, arcs, route, trip_id)` | `feature_engineering.py` | Member 2 (training), Member 3 (HMM inputs) |
| `get_road_connectivity` | use `arcs` directly: arc A connects to arc B iff `arcs[A][1] == arcs[B][0]` | Members 3, 4 |

`data_root` is the folder containing `train/` and `student_test/`.

## Preprocessing output columns (per GPS point)

`lon, lat, ts, valid, dist_prev_m, dt_prev_s, speed_mps, heading_deg, gap_before, speed_unreliable, segment_id`

- `gap_before`: previous interval > 10 s (justified: 98.65% of all intervals are <= 2 s).
- `speed_unreliable`: speed > 55 m/s (~200 km/h) - GPS jump, do not trust that speed.
- `segment_id`: increments at every gap; the HMM should not stitch a route across segments.

## Candidate generation

- Uniform-grid spatial index per trip (50 m cells) - no all-pairs search.
- Distances in metres via per-trip equirectangular projection (dataset spans many latitudes).
- Defaults: radius 30 m, top-k 5. Tuned on real recall numbers, see below.

## Feature schema (one row per point x candidate)

`trip_id, point_idx, arc_id, distance_m, rank, heading_diff_deg, heading_diff_bidir_deg, speed_mps, dist_prev_m, dt_prev_s, gap_before, arc_length_m, connected_to_prev, is_true`

- `is_true` is the TRAINING LABEL ONLY (candidate == nearest ground-truth route arc). Never feed it to a model as input.
- Split train/validation BY TRIP, not by point, to avoid leakage.

## Measured results (first 12 train trips, 26,206 GPS points)

- Candidate recall (true route arc among candidates): mean 96.7%, min 84.6% (trip 00000002), max 100%.
- Average 3.7 candidates per point; 924 points (3.5%) had zero candidates at r=30.
- Radius tuning (recall, averaged over 3 trips): 10 m -> 0.82, 15 m -> 0.92, 20 m -> 0.93, 25 m -> 0.94, 30 m -> 0.94, 40 m -> 0.944, 50 m -> 0.945, 75 m -> 0.947. Knee at ~20-30 m; default 30 m. Note recall is computed against the nearest-route-arc proxy label.
- 268 large gaps and 280 unreliable speeds flagged across the 12 trips - real phenomena, kept and flagged rather than deleted.
- Feature export for 3 trips: 36,765 rows, 22.4% positive labels (see `pipeline_out/features_sample.csv`).

Full outputs in `pipeline_out/`: `trip_statistics.csv`, `candidate_metrics.csv`, `radius_tuning.csv`, `features_sample.csv`, `fig_candidates.png`, `summary.json`.

## Setup and re-run (Windows / Mac / Linux)

```
python -m pip install -r requirements.txt
python run_pipeline.py            # first 12 train trips, dataset expected in ./data
python run_pipeline.py 12 "path/to/Student folder" my_out_dir
```

- `data_root` defaults to `./data`; override with the second argument or the `PATHSNAP_DATA` environment variable. It must be the folder containing `train/` and `student_test/`.
- Output dir defaults to `./pipeline_out`; override with the third argument or `PATHSNAP_OUT`.
- On Windows use `python` (or `py`), not `python3`.
- `run_full_chunk.py START END [data_root] [out_dir]` processes trips [START, END) of the ordered 90-trip list (80 train + 10 student_test) - used for the full-dataset run in chunks.

Full-dataset (90-trip) results: 765,414 feature rows; median per-trip candidate recall 98.5%, macro 95.2%; point-weighted recall excluding the two ground-truth-mismatched trips (00000033, 00000054) is 96.7%; label balance 26%. Exports in `pipeline_out_full/`.
