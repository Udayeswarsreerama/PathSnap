"""PathSnap Member 3 - end-to-end route reconstruction for one trip.

Pipeline per trip:

    raw .track/.nodes/.arcs
      -> Member 1 preprocessing (sort, gaps, speeds, headings, segment ids)
      -> Member 1 candidate generation (spatial grid, radius + top-k)
      -> candidate features (distances, headings, t_frac along arc)
      -> scorer probabilities (Member 2 model via candidate_scorer adapter)
      -> HMM emissions + network transitions (hmm.py)
      -> Viterbi decoding per GPS segment (viterbi.py)
      -> per-point predicted arc ids

The module deliberately reuses Member 1's loading / preprocessing /
spatial-index code (import, not copy) and adds only what the decoder needs
on top: the projection fraction of each GPS point onto each candidate arc.
Per-point candidate distances and classifier probabilities are cached in
the result so decoder-parameter sweeps can re-run decoding without
rebuilding candidates.
"""
import numpy as np
import pandas as pd

from preprocessing import preprocess_trajectory
from candidate_generation import build_trip_index, proxy_true_arc
from feature_engineering import arc_bearings_deg, arc_lengths_m, circular_diff_deg

from hmm import RoadNetwork, RouteHMM, gaussian_logpdf
from viterbi import decode_trip
from candidate_scorer import point_normalised_probs


def build_candidates(gps_df, nodes, arcs, projector, grid, config):
    """Candidate arcs per GPS point, enriched with the projection fraction.

    Same selection as Member 1's generate_candidates (grid query, radius,
    top-k by distance) but each candidate dict also carries:
      t_frac          - fraction along the arc of the point's projection
                        (0 = from-node, 1 = to-node); the transition model
                        needs it to measure network distance between
                        candidate positions
      heading_diff_bidir_deg, arc_length_m, connected_to_prev
                        - feature columns for the scoring model
    """
    nodes_xy = projector.to_m(nodes[:, 0], nodes[:, 1])
    ax = nodes_xy[0][arcs[:, 0]]
    ay = nodes_xy[1][arcs[:, 0]]
    bx = nodes_xy[0][arcs[:, 1]]
    by = nodes_xy[1][arcs[:, 1]]
    bearings = arc_bearings_deg(nodes, arcs, projector)
    lengths = arc_lengths_m(nodes, arcs, projector)

    all_candidates = []
    prev_to_nodes = set()
    for i in range(len(gps_df)):
        row = gps_df.iloc[i]
        px, py = projector.to_m(row["lon"], row["lat"])
        near = grid.query(px, py)
        cands = []
        if len(near) > 0:
            bax, bay = bx[near] - ax[near], by[near] - ay[near]
            L2 = np.where(bax * bax + bay * bay == 0, 1e-12,
                          bax * bax + bay * bay)
            t = np.clip(((px - ax[near]) * bax + (py - ay[near]) * bay) / L2,
                        0.0, 1.0)
            cx = ax[near] + t * bax
            cy = ay[near] + t * bay
            dist = np.hypot(px - cx, py - cy)
            keep = dist <= config.search_radius_m
            order = np.argsort(dist[keep])[:config.top_k]
            near_k, dist_k, t_k = near[keep][order], dist[keep][order], t[keep][order]
            to_nodes = set()
            for rank, (aid, d, tf) in enumerate(zip(near_k, dist_k, t_k), 1):
                hd = circular_diff_deg(row["heading_deg"], bearings[aid]) \
                    if not np.isnan(row["heading_deg"]) else np.nan
                hd_bidir = min(hd, 180.0 - hd) if not np.isnan(hd) else np.nan
                cands.append({
                    "arc_id": int(aid),
                    "distance_m": float(d),
                    "rank": rank,
                    "heading_diff_deg": float(hd) if not np.isnan(hd) else np.nan,
                    "t_frac": float(tf),
                    "heading_diff_bidir_deg": float(hd_bidir) if not np.isnan(hd_bidir) else np.nan,
                    "arc_length_m": float(lengths[aid]),
                    "connected_to_prev": int(arcs[aid, 0]) in prev_to_nodes,
                })
                to_nodes.add(int(arcs[aid, 1]))
            prev_to_nodes = to_nodes
        else:
            prev_to_nodes = set()
        all_candidates.append(cands)
    return all_candidates


def candidates_to_frame(candidates, gps_df):
    """Flat per-(point, candidate) feature frame for the scoring model."""
    rows = []
    for i, cands in enumerate(candidates):
        r = gps_df.iloc[i]
        for c in cands:
            rows.append({
                "point_idx": i,
                "arc_id": c["arc_id"],
                "distance_m": c["distance_m"],
                "rank": c["rank"],
                "heading_diff_deg": c["heading_diff_deg"],
                "heading_diff_bidir_deg": c["heading_diff_bidir_deg"],
                "speed_mps": r["speed_mps"],
                "dist_prev_m": r["dist_prev_m"],
                "dt_prev_s": r["dt_prev_s"],
                "gap_before": bool(r["gap_before"]),
                "arc_length_m": c["arc_length_m"],
                "connected_to_prev": c["connected_to_prev"],
            })
    return pd.DataFrame(rows)


def prepare_trip(trip, scorer, config):
    """Everything about a trip that decoding needs, computed once.

    Returns a dict with the preprocessed trajectory, candidates, cached
    per-point candidate distances and classifier probabilities, the
    classifier-only prediction and the proxy ground truth.
    """
    gps_df = preprocess_trajectory(trip["track"])
    projector, grid = build_trip_index(trip["nodes"], trip["arcs"],
                                       pad_m=config.search_radius_m)
    network = RoadNetwork(trip["nodes"], trip["arcs"], projector)
    candidates = build_candidates(gps_df, trip["nodes"], trip["arcs"],
                                  projector, grid, config)

    frame = candidates_to_frame(candidates, gps_df)
    raw = scorer.score_rows(frame) if len(frame) else np.empty(0)
    probs_iter = point_normalised_probs(frame, raw) if len(frame) else iter([])
    # per-point probability arrays aligned with ALL points
    cand_probs = []
    prob_idx = 0
    probs_list = list(probs_iter) if len(frame) else []
    for cands in candidates:
        if cands:
            cand_probs.append(probs_list[prob_idx])
            prob_idx += 1
        else:
            cand_probs.append(np.empty(0))
    cand_dists = [np.array([c["distance_m"] for c in cands]) if cands
                  else np.empty(0) for cands in candidates]

    clf_top1 = []
    for cands, probs in zip(candidates, cand_probs):
        clf_top1.append(cands[int(np.argmax(probs))]["arc_id"]
                        if cands else None)

    true_arcs = None
    if trip.get("route"):
        true_arcs = [proxy_true_arc(r["lon"], r["lat"], trip["nodes"],
                                    trip["arcs"], trip["route"], projector)
                     for _, r in gps_df.iterrows()]

    return {
        "trip_id": trip["id"],
        "trip": trip,
        "gps_df": gps_df,
        "candidates": candidates,
        "cand_dists": cand_dists,
        "cand_probs": cand_probs,
        "network": network,
        "clf_top1": clf_top1,
        "true_arcs": true_arcs,
    }


def decode_with_params(prepared, config):
    """Run HMM emissions + Viterbi on a prepared trip with the given config."""
    hmm = RouteHMM(config, prepared["network"])
    emissions = []
    for dists, probs in zip(prepared["cand_dists"], prepared["cand_probs"]):
        if len(dists) == 0:
            emissions.append(np.empty(0))
            continue
        spatial = gaussian_logpdf(dists, config.sigma_m)
        emissions.append(config.distance_weight * spatial
                         + config.classifier_weight
                         * np.log(np.clip(probs, 1e-12, None)))
    return decode_trip(prepared["candidates"], emissions,
                       prepared["gps_df"], hmm)


def reconstruct_trip(trip, scorer, config):
    """Full reconstruction: prepare + decode with the config's parameters."""
    prepared = prepare_trip(trip, scorer, config)
    prepared["viterbi"] = decode_with_params(prepared, config)
    return prepared
