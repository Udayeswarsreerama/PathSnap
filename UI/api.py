"""
PathSnap - FastAPI backend (Laptop 1)
=====================================

Runs on the inference laptop. The Streamlit UI (Laptop 2) calls this API.

INTEGRATION STATUS - COMPLETE (Members 1 + 2 + 3)
-------------------------------------------------
The prediction path is the team's full system:

    .track -> Member 1 preprocess_trajectory()
           -> Member 1 candidate generation (radius 30 m, top_k 5, per-trip
              equirectangular projection + grid spatial index, enriched with
              the projection fraction along each arc by Member 3)
           -> Member 2 model per-candidate probability (via Member 3's
              candidate_scorer adapter - currently the MLP artifact
              member3/mlp.joblib, configured in member3/hmm_config.json)
           -> Member 3 HMM emissions (Gaussian on point-to-arc distance
              + classifier probability) and Newson-Krumm style transitions
              (bounded Dijkstra network distance vs travelled GPS distance)
           -> Member 3 Viterbi decoding per GPS segment (never stitches
              across a >10 s GPS gap; candidate-less points inherit the
              previous decoded arc in the same segment)

The classifier-only per-point argmax is still computed alongside and
reported as classifier_only_point_accuracy, so the evaluator can see what
HMM/Viterbi adds on top of the raw classifier.

SWAPPING IN MEMBER 2'S TUNED MODEL (one line)
---------------------------------------------
When the Bayesian-tuned LightGBM lands, copy the new .joblib (and its
.features.json sidecar if the feature list changed) into member3/ and edit
ONE line in member3/hmm_config.json:

    "artifact_path": "mlp.joblib"   ->   "artifact_path": "lightgbm_tuned.joblib"

Then install lightgbm (already in requirements.txt) and restart uvicorn.
Nothing in api.py changes - candidate_scorer loads any joblib model with
predict_proba.

CONTRACT WITH THE UI (unchanged)
--------------------------------
predict_trip(trip_id) returns:

    {
        "trip_id": str,
        "gps_points": [[lat, lon], ...],
        "predicted_road_sequence": [arc_id, ...],   # chronological, HMM-decoded
        "reconstructed_route": [[lat, lon], ...],   # polyline from arcs
        "metrics": {                                 # SERVER-SIDE only
            "point_top1_accuracy": float | null,     # per-point vs nearest
                                                     # route arc (Viterbi)
            "classifier_only_point_accuracy": float | null,  # argmax, for
                                                     # comparison
            "route_membership": float | null,        # predicted arc on the
                                                     # reference route
            "route_continuity": float | null,        # connected consecutive
                                                     # predictions
            "invalid_transitions": int | null,
            "top1_arc_accuracy": float | null,       # positional seq match
            "route_similarity_lcs": float | null,    # LCS overlap vs reference
            "runtime_ms": float,
            "num_gps_points": int,
            "num_predicted_arcs": int,
            "num_reference_arcs": int | null
        }
    }

Ground truth (.route) is loaded ONLY here on the server to compute metrics.
It is never returned to the UI and never shown to the prediction pipeline.

Decoded trips are cached in memory, so re-requesting a trip the evaluator
already saw returns instantly.

REPO LAYOUT REQUIRED BY THIS FILE
---------------------------------
    api.py
    pipeline/                  # Member 1's modules, imported unmodified
        data_loader.py  preprocessing.py  candidate_generation.py
        feature_engineering.py
    member3/                   # Member 3's modules, imported unmodified
        hmm.py  viterbi.py  candidate_scorer.py  route_reconstruction.py
        evaluate_decoder.py  hmm_config.json
        mlp.joblib  mlp.joblib.features.json
    data/student_test/         # DATA_DIR: one subfolder per trip
        trip_xxx/  *.track  *.nodes  *.arcs  *.route

Overrides: PIPELINE_DIR, MEMBER3_DIR, HMM_CONFIG_PATH, DATA_DIR env vars.
At the final evaluation, drop the 10 reserved evaluator trips into the same
per-trip folder structure (or point DATA_DIR at them) and restart.

Run (Laptop 1):
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import sys
import time
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException

# ---------------------------------------------------------------------------
# Config + visible request logging (shown to the evaluator on this terminal)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", "./data/student_test")
PIPELINE_DIR = os.environ.get("PIPELINE_DIR", os.path.join(BASE_DIR, "pipeline"))
MEMBER3_DIR = os.environ.get("MEMBER3_DIR", os.path.join(BASE_DIR, "member3"))
HMM_CONFIG_PATH = os.environ.get(
    "HMM_CONFIG_PATH", os.path.join(MEMBER3_DIR, "hmm_config.json"))

# Member 1's and Member 3's modules import each other as top-level modules
# (e.g. `from preprocessing import ...`), so both folders go on sys.path.
for _p in (PIPELINE_DIR, MEMBER3_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Teammates' modules - imported, not copied.
from data_loader import load_trip as m1_load_trip                      # noqa: E402
from hmm import HMMConfig                                              # noqa: E402
from candidate_scorer import build_scorer                              # noqa: E402
from route_reconstruction import reconstruct_trip                      # noqa: E402
from evaluate_decoder import evaluate_trip as m3_evaluate_trip         # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PathSnap API] %(message)s",
)
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap API")


# ---------------------------------------------------------------------------
# HMM config + scorer loading (Member 3's HMM, Member 2's model artifact)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_config() -> HMMConfig:
    """Load member3/hmm_config.json once.

    A relative scorer artifact_path in the config is resolved against the
    config file's own folder, so the demo works from any CWD.
    """
    log.info("Loading HMM config from %s", HMM_CONFIG_PATH)
    cfg = HMMConfig.load(HMM_CONFIG_PATH)
    scorer = cfg.scorer or {}
    if scorer.get("type") == "artifact":
        path = scorer.get("artifact_path", "")
        if path and not os.path.isabs(path):
            scorer["artifact_path"] = os.path.join(
                os.path.dirname(HMM_CONFIG_PATH), path)
    log.info(
        "HMM config: sigma=%.1f beta=%.1f radius=%.1f top_k=%d scorer=%s",
        cfg.sigma_m, cfg.transition_beta_m, cfg.search_radius_m, cfg.top_k,
        cfg.scorer,
    )
    return cfg


@lru_cache(maxsize=1)
def load_scorer():
    """Build the candidate scorer named in the HMM config, once.

    With the default config this loads Member 2's MLP (member3/mlp.joblib,
    feature columns from mlp.joblib.features.json) through Member 3's
    ArtifactScorer adapter.
    """
    cfg = load_config()
    scorer = build_scorer(cfg)
    log.info("Scorer loaded: %s (features=%s)",
             type(scorer).__name__, getattr(scorer, "feature_cols", "?"))
    return scorer


# ---------------------------------------------------------------------------
# Trip discovery and loading (Member 1's loader)
# ---------------------------------------------------------------------------
def list_trips():
    """Trip IDs = subfolders of DATA_DIR containing at least a .track file."""
    trips = []
    if not os.path.isdir(DATA_DIR):
        log.warning("DATA_DIR %s does not exist", DATA_DIR)
        return trips
    for name in sorted(os.listdir(DATA_DIR)):
        trip_dir = os.path.join(DATA_DIR, name)
        if os.path.isdir(trip_dir):
            for f in os.listdir(trip_dir):
                if f.endswith(".track"):
                    trips.append(name)
                    break
    return trips


@lru_cache(maxsize=32)
def load_trip(trip_id: str):
    """Load one trip via Member 1's data_loader. Cached in memory.

    Returns dict: id, track (N,3 lon/lat/ts), nodes (M,2), arcs (K,2),
    route (list of arc ids or None - ground truth, server-side only).
    """
    if trip_id not in list_trips():
        raise HTTPException(status_code=404, detail=f"Unknown trip: {trip_id}")
    trip = m1_load_trip(os.path.join(DATA_DIR, trip_id))
    log.info(
        "Loaded trip %s: %d GPS points, %d nodes, %d arcs, route=%s",
        trip_id, len(trip["track"]), len(trip["nodes"]), len(trip["arcs"]),
        "none" if trip["route"] is None else f"{len(trip['route'])} arcs",
    )
    return trip


# ---------------------------------------------------------------------------
# Member 3's full chain, cached per trip
# ---------------------------------------------------------------------------
@lru_cache(maxsize=16)
def decode_trip(trip_id: str):
    """Run Member 3's end-to-end chain for one trip and cache the result.

    preprocess -> candidates -> Member 2 scores -> HMM -> Viterbi, plus the
    classifier-only argmax for comparison. Returns Member 3's `prepared`
    dict; prepared["viterbi"] is the per-point decoded arc sequence (one
    entry per GPS point, None only where a whole GPS segment had no
    candidate arcs). First request for a trip pays the full decode cost;
    repeats are instant.
    """
    trip = load_trip(trip_id)
    t0 = time.perf_counter()
    prepared = reconstruct_trip(trip, load_scorer(), load_config())
    log.info("Decoded trip %s (HMM/Viterbi) in %.1f ms",
             trip_id, (time.perf_counter() - t0) * 1000)
    return prepared


# ---------------------------------------------------------------------------
# Server-side evaluation metrics (ground truth never leaves this process)
# ---------------------------------------------------------------------------
def _lcs_length(a, b) -> int:
    """Longest common subsequence length (arc-id sequences)."""
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def compute_metrics(prepared, predicted_seq, reference_route,
                    runtime_ms: float, num_gps: int) -> dict:
    """All server-side metrics for one decoded trip.

    Per-point and continuity numbers come from Member 3's evaluate_decoder
    (proxy nearest-route-arc labels, route membership, network-continuity
    with a 150 m reachability check). Sequence-level numbers compare the
    collapsed decoded arc sequence with the reference route.
    """
    metrics = {
        "point_top1_accuracy": None,
        "point_accuracy_direction_agnostic": None,
        "classifier_only_point_accuracy": None,
        "route_membership": None,
        "route_continuity": None,
        "invalid_transitions": None,
        "top1_arc_accuracy": None,
        "route_similarity_lcs": None,
        "runtime_ms": round(runtime_ms, 1),
        "num_gps_points": num_gps,
        "num_predicted_arcs": len(predicted_seq),
        "num_reference_arcs": None,
    }
    if reference_route is None or len(reference_route) == 0:
        log.info("No .route file for this trip - accuracy metrics unavailable")
        return metrics

    # Member 3's decoder evaluation (server-side ground truth)
    m3 = m3_evaluate_trip(prepared, load_config())
    for api_key, m3_key in (
        ("point_top1_accuracy", "viterbi_point_acc"),
        ("classifier_only_point_accuracy", "clf_top1_point_acc"),
        ("route_membership", "viterbi_route_membership"),
        ("route_continuity", "viterbi_continuity"),
    ):
        if m3_key in m3:
            metrics[api_key] = round(float(m3[m3_key]), 4)
    if "viterbi_invalid_transitions" in m3:
        metrics["invalid_transitions"] = int(m3["viterbi_invalid_transitions"])

    # Additional observation-level metric; the standard directed-arc metric
    # above is unchanged. Reuse the same already-computed proxy labels.
    true_arcs = prepared.get("true_arcs")
    decoded = prepared["viterbi"]
    if true_arcs is not None and len(prepared["gps_df"]):
        if len(decoded) != len(true_arcs) or len(decoded) != len(prepared["gps_df"]):
            raise ValueError("Per-point predictions and reference labels must align")
        arcs = prepared["trip"]["arcs"]
        correct = 0
        for predicted_arc, true_arc in zip(decoded, true_arcs):
            if predicted_arc is None or true_arc is None:
                continue  # unmatched observations remain in the denominator
            p, t = int(predicted_arc), int(true_arc)
            if not (0 <= p < len(arcs) and 0 <= t < len(arcs)):
                continue
            if p == t or (int(arcs[p, 0]) == int(arcs[t, 1])
                          and int(arcs[p, 1]) == int(arcs[t, 0])):
                correct += 1
        metrics["point_accuracy_direction_agnostic"] = round(
            correct / len(prepared["gps_df"]), 4)

    # Sequence-level comparison against the reference route
    ref = [int(a) for a in reference_route]
    pred = [int(a) for a in predicted_seq]
    metrics["num_reference_arcs"] = len(ref)
    n = min(len(pred), len(ref))
    if n:
        metrics["top1_arc_accuracy"] = round(
            sum(1 for i in range(n) if pred[i] == ref[i]) / n, 4)
    denom = max(len(pred), len(ref))
    if denom:
        metrics["route_similarity_lcs"] = round(_lcs_length(pred, ref) / denom, 4)
    return metrics


# ---------------------------------------------------------------------------
# Contract implementation
# ---------------------------------------------------------------------------
def build_reconstructed_route(predicted_arcs, nodes, arcs):
    """Turn a predicted arc sequence into a [lat, lon] polyline."""
    points = []
    for arc_id in predicted_arcs:
        from_node, to_node = arcs[arc_id]
        flon, flat = nodes[from_node]
        tlon, tlat = nodes[to_node]
        if not points or points[-1] != [flat, flon]:
            points.append([float(flat), float(flon)])
        points.append([float(tlat), float(tlon)])
    return points


def predict_trip(trip_id: str) -> dict:
    t0 = time.perf_counter()
    trip = load_trip(trip_id)
    track, nodes, arcs, route = (trip["track"], trip["nodes"],
                                 trip["arcs"], trip["route"])

    prepared = decode_trip(trip_id)  # Member 3's chain; cached per trip
    decoded = prepared["viterbi"]    # one arc id (or None) per GPS point

    # Chronological arc sequence: collapse consecutive duplicates, skip the
    # rare None points (a whole GPS segment without any candidate arcs).
    predicted = []
    prev_arc = None
    for arc_id in decoded:
        if arc_id is None:
            continue
        if arc_id != prev_arc:
            predicted.append(int(arc_id))
            prev_arc = arc_id

    runtime_ms = (time.perf_counter() - t0) * 1000
    gps_points = [[float(lat), float(lon)] for lon, lat, _ in track]
    return {
        "trip_id": trip_id,
        "gps_points": gps_points,
        "predicted_road_sequence": predicted,
        "reconstructed_route": build_reconstructed_route(predicted, nodes, arcs),
        "metrics": compute_metrics(prepared, predicted, route, runtime_ms,
                                   len(track)),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    trips = list_trips()
    log.info("GET /health -> ok, %d trips available", len(trips))
    return {"status": "ok", "data_dir": os.path.abspath(DATA_DIR),
            "num_trips": len(trips)}


@app.get("/trips")
def trips():
    trip_list = list_trips()
    log.info("GET /trips -> %s", trip_list)
    return {"trips": trip_list}


@app.post("/predict/{trip_id}")
def predict(trip_id: str):
    log.info("POST /predict/%s - request received", trip_id)
    t0 = time.perf_counter()
    result = predict_trip(trip_id)
    m = result["metrics"]
    log.info(
        "POST /predict/%s - done in %.1f ms | %d GPS pts -> %d arcs | "
        "point_top1=%s membership=%s continuity=%s lcs=%s",
        trip_id, (time.perf_counter() - t0) * 1000,
        m["num_gps_points"], m["num_predicted_arcs"],
        m["point_top1_accuracy"], m["route_membership"],
        m["route_continuity"], m["route_similarity_lcs"],
    )
    return result
