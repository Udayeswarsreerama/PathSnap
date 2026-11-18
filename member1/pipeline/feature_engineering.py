"""PathSnap feature engineering (Member 1, Task 4).

Builds the per-(GPS point, candidate arc) feature dataset that Member 2's
models train on and Member 3's HMM reuses.

One row per (point, candidate):
  trip_id, point_idx, arc_id,
  distance_m        - point-to-segment distance (candidate generation)
  rank              - candidate rank by distance (1 = nearest)
  heading_diff_deg  - circular difference between GPS movement bearing and arc bearing
  heading_diff_bidir_deg - min(heading_diff, 180 - heading_diff): alignment ignoring travel direction
  speed_mps         - vehicle speed into this point (from preprocessing)
  dist_prev_m       - distance from the previous GPS point
  dt_prev_s         - time from the previous GPS point
  gap_before        - True when the previous interval exceeded the gap threshold
  arc_length_m      - length of the candidate arc
  connected_to_prev - True when this candidate's from-node equals any previous
                      candidate's to-node (direct connectivity)
  is_true           - training label only: candidate arc == nearest route arc proxy.
                      NEVER use as an input feature. Only available for train trips.

Train/validation must be split by trip, not by point, to avoid leakage.
"""
import numpy as np
import pandas as pd

from preprocessing import bearing_deg
from candidate_generation import (build_trip_index, generate_candidates,
                                  proxy_true_arc, SEARCH_RADIUS_M, TOP_K)


def circular_diff_deg(a, b):
    """Smallest absolute difference between two headings in degrees."""
    d = np.abs((a - b + 180.0) % 360.0 - 180.0)
    return d


def arc_bearings_deg(nodes, arcs, projector):
    """Bearing of every arc (from-node -> to-node), degrees."""
    ax, ay = projector.to_m(nodes[arcs[:, 0], 0], nodes[arcs[:, 0], 1])
    bx, by = projector.to_m(nodes[arcs[:, 1], 0], nodes[arcs[:, 1], 1])
    return (np.degrees(np.arctan2(bx - ax, by - ay)) + 360.0) % 360.0


def arc_lengths_m(nodes, arcs, projector):
    ax, ay = projector.to_m(nodes[arcs[:, 0], 0], nodes[arcs[:, 0], 1])
    bx, by = projector.to_m(nodes[arcs[:, 1], 0], nodes[arcs[:, 1], 1])
    return np.hypot(bx - ax, by - ay)


def extract_features(gps_df, nodes, arcs, route=None, trip_id=None,
                     radius_m=SEARCH_RADIUS_M, top_k=TOP_K):
    """Feature dataset for one trip.

    gps_df: output of preprocessing.preprocess_trajectory
    route:  ground-truth arc ids (train trips) or None
    """
    projector, grid = build_trip_index(nodes, arcs)
    bearings = arc_bearings_deg(nodes, arcs, projector)
    lengths = arc_lengths_m(nodes, arcs, projector)
    arc_to_node = arcs[:, 1]  # to_node per arc

    rows = []
    prev_cand_to_nodes = set()
    for i in range(len(gps_df)):
        r = gps_df.iloc[i]
        cands = generate_candidates(r["lon"], r["lat"], nodes, arcs,
                                    projector=projector, radius_m=radius_m,
                                    top_k=top_k, grid=grid)
        true_arc = proxy_true_arc(r["lon"], r["lat"], nodes, arcs, route, projector) \
            if route is not None else None
        cand_to_nodes = set()
        for c in cands:
            aid = c["arc_id"]
            hd = circular_diff_deg(r["heading_deg"], bearings[aid]) \
                if not np.isnan(r["heading_deg"]) else np.nan
            rows.append({
                "trip_id": trip_id,
                "point_idx": i,
                "arc_id": aid,
                "distance_m": c["distance_m"],
                "rank": c["rank"],
                "heading_diff_deg": hd,
                "heading_diff_bidir_deg": min(hd, 180.0 - hd) if not np.isnan(hd) else np.nan,
                "speed_mps": r["speed_mps"],
                "dist_prev_m": r["dist_prev_m"],
                "dt_prev_s": r["dt_prev_s"],
                "gap_before": bool(r["gap_before"]),
                "arc_length_m": float(lengths[aid]),
                "connected_to_prev": arcs[aid, 0] in prev_cand_to_nodes,
                "is_true": (aid == true_arc) if true_arc is not None else None,
            })
            cand_to_nodes.add(int(arc_to_node[aid]))
        prev_cand_to_nodes = cand_to_nodes
    return pd.DataFrame(rows)
