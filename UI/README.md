# PathSnap - Integrated API (Members 1 + 2 + 3)

FastAPI backend for the PathSnap demo. It runs the team's complete system:

    GPS trace -> Member 1 preprocessing + candidate generation
              -> Member 2 model (per-candidate probabilities)
              -> Member 3 HMM + Viterbi decoding
              -> predicted route + server-side accuracy metrics

Ground truth (.route) is loaded only on the server to score predictions; it
is never sent to the UI or shown to the prediction pipeline. Every request
is logged to the terminal - that log is what you show the evaluator.

## Folder layout (Laptop 1)

Put these next to each other in one folder:

    api.py                  <- this package
    requirements.txt        <- this package
    pipeline/               <- member 1's modules (unmodified)
    member3/                <- member 3's modules + config + model (unmodified)
        hmm.py  viterbi.py  candidate_scorer.py  route_reconstruction.py
        evaluate_decoder.py  hmm_config.json
        mlp.joblib  mlp.joblib.features.json
    data/student_test/      <- one subfolder per trip
        <trip_id>/
            <trip_id>.track  <trip_id>.nodes  <trip_id>.arcs  <trip_id>.route

At the final evaluation, drop the reserved evaluator trips into
`data/student_test/` in the same per-trip layout (or set DATA_DIR to their
folder) and restart the server.

## Laptop 1: run the API

```bash
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

Leave this terminal visible. It logs every request with timing and accuracy:

```
POST /predict/00000001 - done in 1145.8 ms | 2503 GPS pts -> 248 arcs | point_top1=0.9453 membership=0.9493 continuity=0.9988 lcs=0.6653
```

Check it works: `curl http://localhost:8000/health` should return
`{"status":"ok", ...}` and `curl http://localhost:8000/trips` lists the
trips it found.

## Laptop 2: run the UI

```bash
streamlit run ui.py
```

In the UI's server address field, enter laptop 1's address, e.g.
`http://192.168.1.10:8000` (laptop 1's LAN IP - find it with `ip addr`).
Pick a trip, run prediction.

## Reading the results on screen

- The map shows the GPS trace and the reconstructed predicted route.
- The metrics come from the server scoring against the hidden reference
  route:
  - `point_top1_accuracy` - per-point accuracy of the HMM/Viterbi route
    (0.96 on the sample trip, pooled 0.887 across all 10 test trips)
  - `point_accuracy_direction_agnostic` - same, but a predicted arc also
    counts as correct when it is the exact reverse of the reference arc
    (0.966 on the sample trip). Some test trips retrace the same road in
    the opposite direction, where the direction is ambiguous; this metric
    ignores that ambiguity. Neither metric changes the predicted route.
  - `classifier_only_point_accuracy` - the same trip with NO HMM, just the
    classifier's argmax (0.923 on the sample) - the gap between the two is
    what member 3's decoder adds
  - `route_membership` - fraction of predicted arcs that are on the
    reference route
  - `route_continuity` - fraction of consecutive predictions that are
    connected in the road network (~0.999 with Viterbi vs ~0.95 without)
  - `route_similarity_lcs` - sequence overlap with the reference route

## Swapping in member 2's tuned model (one line)

When the Bayesian-tuned LightGBM lands:

1. Copy the new `.joblib` (and its `.features.json` sidecar, if the feature
   list changed) into `member3/`.
2. Edit one line in `member3/hmm_config.json`:

   ```json
   "artifact_path": "mlp.joblib"   ->   "artifact_path": "lightgbm_tuned.joblib"
   ```

3. Restart uvicorn. Nothing else changes - member 3's scorer adapter loads
   any joblib model with `predict_proba`.

`lightgbm` is already in requirements.txt so this swap needs no reinstall.

## Notes

- Decoded trips are cached in memory: the first prediction for a trip takes
  ~1 s, a repeat of the same trip returns in ~10 ms.
- Endpoints: `GET /health`, `GET /trips`, `POST /predict/{trip_id}`.
- Env overrides: `DATA_DIR`, `PIPELINE_DIR`, `MEMBER3_DIR`, `HMM_CONFIG_PATH`.
