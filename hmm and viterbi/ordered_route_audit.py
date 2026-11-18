"""Evaluation-only ordered-route alignment. Never use this to predict a trip.

This is an alternative proxy labeler, NOT measured per-GPS ground truth.
It finds a nondecreasing sequence of reference-route indices minimizing
squared GPS-to-reference-segment distance over the whole trace. Same-segment
projection jitter is allowed. It uses no model outputs. It can remain
ambiguous on repeated/coincident reference sections, stops, and outages.
Do not replace an official metric with this without the evaluator's approval.
"""
import numpy as np


def align_reference_route(gps_df, nodes, arcs, route, projector):
    route = np.asarray(route, dtype=np.int64)
    if not len(route) or not len(gps_df):
        raise ValueError("Nonempty observations and reference route required")
    x, y = projector.to_m(nodes[:, 0], nodes[:, 1])
    px, py = projector.to_m(gps_df["lon"].to_numpy(), gps_df["lat"].to_numpy())
    u, v = arcs[route, 0], arcs[route, 1]
    dx, dy = x[v] - x[u], y[v] - y[u]
    l2 = np.maximum(dx * dx + dy * dy, 1e-12)
    cost = np.empty((len(gps_df), len(route)))
    for i in range(len(gps_df)):
        t = np.clip(((px[i] - x[u]) * dx + (py[i] - y[u]) * dy) / l2, 0, 1)
        cost[i] = (px[i] - x[u] - t * dx) ** 2 + (py[i] - y[u] - t * dy) ** 2
    back = np.zeros(cost.shape, dtype=np.int32)
    prev = cost[0].copy()
    indices = np.arange(len(route), dtype=np.int32)
    for i in range(1, len(gps_df)):
        prefix_min = np.minimum.accumulate(prev)
        # Strict improvements keep the earliest route index on an exact tie.
        improved = np.r_[True, prefix_min[1:] < prefix_min[:-1]]
        best_idx = np.maximum.accumulate(np.where(improved, indices, 0))
        back[i] = best_idx
        prev = cost[i] + prefix_min
    path = np.empty(len(gps_df), dtype=np.int32)
    path[-1] = np.argmin(prev)
    for i in range(len(gps_df) - 2, -1, -1):
        path[i] = back[i + 1, path[i + 1]]
    return route[path], path
