# PathSnap - Member 1 (Obaid): Dataset, EDA & Feature Pipeline

This package is Member 1's complete part for the PathSnap hackathon: the EDA
(Task 1) and the data pipeline (Tasks 2-4: preprocessing, candidate
generation, feature engineering).

## Contents

- `EDA.ipynb` - EDA notebook (24 cells, follows the required 9-section
  checklist). Set `DATA_ROOT` to your `data` folder and run top to bottom.
- `eda_analysis.py` - script version of the EDA.
- `figures/` - all EDA figures (trajectories, distributions, timestamp gaps,
  road network, GPS-vs-route overlay, noise, coordinate distributions) plus
  `fig_candidates.png` (candidate arcs vs ground-truth route).
- `maps/` - interactive Folium maps (open in a browser):
  `folium_map.html` (one trip), `folium_map_multi_trip.html` (4 trips,
  toggleable layers - best demo visual).
- `report/` - `report_section2_dataset.md` (Dataset section of the final
  report, written with real numbers), `presentation_summary.md` (talking
  points), `eda_results.json` (every computed stat).
- `pipeline/` - the Tasks 2-4 code:
  - `data_loader.py` - trip + road-network loading and validation
  - `preprocessing.py` - sorting, haversine distances/speeds, gap & jump
    flagging (flags, never deletes)
  - `candidate_generation.py` - candidate road arcs per GPS point (spatial
    index, 30m default radius, top-5)
  - `feature_engineering.py` - per (point, candidate) features + labels,
    trip-level train/val split
  - `run_pipeline.py` - runs the whole chain
  - `pipeline_out/` - results: full-run candidate metrics (all 90 trips),
    trip statistics, summary.json, radius tuning table, plus
    `features_sample.csv` (36,765-row feature export from 3 trips). The full
    765,414-row feature export is 112MB - regenerate it with
    `run_full_chunk.py` in ~2 minutes.

## Key numbers (real, from the data)

- 90 trips (80 train / 10 test), 227,577 GPS points.
- Candidate recall (all 90 trips): median trip 98.5%, mean 95.2%
  (r=30m, top_k=5). Radius tuning knee at 20-30m; defaults: radius 30m,
  top_k 5.
- DATA ANOMALY: trips 00000033 and 00000054 have mismatched ground truth
  (GPS trace ~6.5km from its route / route offset ~400m). Exclude them from
  training - per-point labels there are meaningless. Flag to organizers.
- GPS noise: median 4.2m, p90 9.7m -> supports sigma ~5-10m for the HMM.
- Preprocessing flagged 268 large time gaps and 280 impossible speeds
  (kept, flagged via segment_id / speed_unreliable).

## What each teammate does next

**Member 2 (models):** train the 5+ models on the feature export
(`pipeline/pipeline_out/features_sample.csv` is 3 trips; regenerate the full
765k-row export with `run_full_chunk.py`). IMPORTANT: exclude trips 00000033
and 00000054 - their ground truth is mismatched (see README key numbers). Use the trip-level train/val split already in
feature_engineering.py. Positive label rate is ~22% - watch class balance.

**Member 3 (HMM/Viterbi):** consume `generate_candidates()` output
(distance, heading difference, speed, connectivity per candidate) as
emission/transition inputs. sigma ~5-10m Gaussian emission fits the measured
noise. `segment_id` splits trips at outages - never let Viterbi stitch
across a gap. Connectivity between consecutive candidates is yours
(the pipeline only marks direct node sharing).

**Member 4 (integration/report):** `report/` has the Dataset section ready
to drop in; figures/ and maps/ are presentation material. Interfaces for the
app: `load_dataset()`, `preprocess_trajectory()`, `generate_candidates()`,
`extract_features()` - documented in `pipeline/README.md`.

## Notes

- The 10 student_test trips in our copy also contain .route ground truth;
  confirm with the organizers whether that's intentional.
- Judgment calls (change if the team prefers): radius 30m, top_k 5,
  zero-candidate points (~3.5%) dropped from features.
