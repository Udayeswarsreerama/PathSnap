"""PathSnap Member 3 - decoder evaluation.

Compares classifier-only predictions with classifier+HMM/Viterbi decoding
on every evaluated trip and reports the metrics required by the brief:

  * observation-level Top-1 accuracy (argmax candidate vs proxy true arc)
  * candidate miss rate (true arc absent from the candidate set, or no
    candidates at all)
  * Viterbi (decoded) point accuracy
  * sequence-level performance: fraction of points whose predicted arc lies
    on the reference route at all ("route membership"), and coverage of the
    reference route arcs
  * route continuity: fraction of consecutive same-segment predictions that
    are connected in the road network (same arc, shared node, or reachable
    within a short network distance); invalid transitions are counted
  * GPS-gap behaviour: accuracy on points just after a sampling outage vs
    normal points

Ground truth note (matches Member 1's choice): the .route file is a
sequence of arcs, not per-point labels, so per-point accuracy uses the
nearest-route-arc proxy. With ~4 m median GPS noise this is accurate away
from intersections; we report route-membership metrics alongside so the
proxy's limitations are visible.
"""
import numpy as np
import pandas as pd


def _connected(network, arc_a, arc_b, reach_m=150.0):
    """Continuity check: same arc, shared node, or short network path."""
    if arc_a is None or arc_b is None:
        return False
    if arc_a == arc_b:
        return True
    if int(network.arcs[arc_a, 1]) == int(network.arcs[arc_b, 0]):
        return True
    sp = network.shortest_path(int(network.arcs[arc_a, 1]),
                               int(network.arcs[arc_b, 0]), reach_m)
    return np.isfinite(sp)


def evaluate_trip(prepared, config):
    """Metric dict for one decoded trip (prepare_trip + decode_with_params)."""
    gps_df = prepared["gps_df"]
    true_arcs = prepared["true_arcs"]
    route_set = set(prepared["trip"]["route"]) if prepared["trip"].get("route") else set()
    net = prepared["network"]
    n = len(gps_df)

    out = {"trip_id": prepared["trip_id"], "n_points": n}

    # candidate miss rate (needs ground truth)
    cand_sets = [set(c["arc_id"] for c in cands) for cands in prepared["candidates"]]
    n_zero = sum(1 for s in cand_sets if not s)
    out["zero_candidate_rate"] = n_zero / n if n else 0.0

    seg = gps_df["segment_id"].values
    gap = gps_df["gap_before"].values

    for name, pred in (("clf_top1", prepared["clf_top1"]),
                       ("viterbi", prepared["viterbi"])):
        if true_arcs is not None:
            correct = sum(1 for p, t in zip(pred, true_arcs) if p == t)
            out[f"{name}_point_acc"] = correct / n if n else 0.0
            # accuracy on points right after a GPS gap vs normal points
            for label, mask in (("gap", gap), ("normal", ~gap)):
                idx = np.where(mask)[0]
                if len(idx):
                    c = sum(1 for i in idx if pred[i] == true_arcs[i])
                    out[f"{name}_acc_{label}_pts"] = c / len(idx)
            if route_set:
                member = sum(1 for p in pred if p in route_set)
                out[f"{name}_route_membership"] = member / n if n else 0.0
            # candidate miss rate only needs the candidate sets
            miss = sum(1 for s, t in zip(cand_sets, true_arcs) if t not in s)
            out["candidate_miss_rate"] = miss / n if n else 0.0
        # route continuity over consecutive predictions inside one segment
        pairs = [(pred[i - 1], pred[i]) for i in range(1, n)
                 if seg[i] == seg[i - 1]]
        if pairs:
            ok = sum(1 for a, b in pairs if _connected(net, a, b))
            out[f"{name}_continuity"] = ok / len(pairs)
            out[f"{name}_invalid_transitions"] = len(pairs) - ok
    if route_set:
        for name, pred in (("clf_top1", prepared["clf_top1"]),
                           ("viterbi", prepared["viterbi"])):
            used = set(p for p in pred if p is not None)
            out[f"{name}_route_coverage"] = \
                len(used & route_set) / len(route_set) if route_set else 0.0
    return out


def sweep_parameters(prepared_trips, base_config, param, values):
    """Re-decode prepared trips for each value of one decoder parameter.

    Returns a DataFrame with one row per (parameter value, trip) holding
    Viterbi point accuracy and continuity. Candidates and classifier
    probabilities are reused, so this is cheap.
    """
    import copy
    rows = []
    for v in values:
        cfg = copy.deepcopy(base_config)
        setattr(cfg, param, v)
        for prepared in prepared_trips:
            prepared["viterbi"] = decode_with_params(prepared, cfg)
            m = evaluate_trip(prepared, cfg)
            rows.append({
                "param": param, "value": v, "trip_id": prepared["trip_id"],
                "viterbi_point_acc": m.get("viterbi_point_acc", np.nan),
                "viterbi_continuity": m.get("viterbi_continuity", np.nan),
            })
    return pd.DataFrame(rows)


def summarise(metrics_rows):
    """Aggregate per-trip metrics into the comparison table."""
    df = pd.DataFrame(metrics_rows)
    agg = {}
    for col in df.columns:
        if col in ("trip_id", "n_points"):
            continue
        vals = df[col].astype(float) * df["n_points"]
        agg[col] = vals.sum() / df["n_points"].sum()  # point-weighted mean
    summary = pd.DataFrame([agg])
    return df, summary


# import here to avoid a cycle: sweep needs decode
from route_reconstruction import decode_with_params  # noqa: E402
