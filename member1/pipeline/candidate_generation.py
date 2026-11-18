"""PathSnap candidate road generation (Member 1, Task 3).

For every GPS point, find the road arcs that could plausibly have generated it:
all arcs whose segment comes within SEARCH_RADIUS_M of the point, ranked by
point-to-segment distance, keeping at most TOP_K.

Distances are computed in metres: each trip is projected with an equirectangular
approximation centred on the trip's mean position (the dataset spans many
cities, so a single global projection is invalid).

Speed: arcs are bucketed into a uniform grid (cell = GRID_CELL_M metres) once per
trip, so each GPS point only tests the arcs in the cells around it instead of
the whole network.

Candidate recall (the key quality metric) is measured against a proxy label:
the route arc nearest to the GPS point. Ground truth is a route (sequence of
arc ids), not per-point labels, so the nearest route arc is the standard proxy;
with a median GPS noise of ~4 m it is accurate away from intersections.
"""
import numpy as np

SEARCH_RADIUS_M = 30.0   # EDA: GPS noise p90 = 9.7 m, max = 27 m -> 30 m covers the noise
TOP_K = 5
GRID_CELL_M = 50.0


class TripProjector:
    """Equirectangular projection centred on one trip."""

    def __init__(self, lons, lats):
        self.lon0 = float(np.mean(lons))
        self.lat0 = float(np.mean(lats))
        self.kx = 111320.0 * np.cos(np.radians(self.lat0))
        self.ky = 110540.0

    def to_m(self, lon, lat):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky


def _point_seg_dist_m(px, py, ax, ay, bx, by):
    """Distance(s) from point (px,py) to segments (a->b), all in metres. Vectorised over segments."""
    bax, bay = bx - ax, by - ay
    L2 = bax * bax + bay * bay
    L2 = np.where(L2 == 0, 1e-12, L2)
    t = ((px - ax) * bax + (py - ay) * bay) / L2
    t = np.clip(t, 0.0, 1.0)
    cx = ax + t * bax
    cy = ay + t * bay
    return np.hypot(px - cx, py - cy)


class ArcGridIndex:
    """Uniform grid spatial index over the arcs of one trip (metre space)."""

    def __init__(self, nodes_xy, arcs, cell_m=GRID_CELL_M, pad_m=SEARCH_RADIUS_M):
        self.cell = cell_m
        self.pad = pad_m
        ax = nodes_xy[0][arcs[:, 0]]
        ay = nodes_xy[1][arcs[:, 0]]
        bx = nodes_xy[0][arcs[:, 1]]
        by = nodes_xy[1][arcs[:, 1]]
        self.ax, self.ay, self.bx, self.by = ax, ay, bx, by
        self.grid = {}
        x0 = np.minimum(ax, bx) - pad_m
        x1 = np.maximum(ax, bx) + pad_m
        y0 = np.minimum(ay, by) - pad_m
        y1 = np.maximum(ay, by) + pad_m
        for aid in range(len(arcs)):
            cx0, cx1 = int(x0[aid] // cell_m), int(x1[aid] // cell_m)
            cy0, cy1 = int(y0[aid] // cell_m), int(y1[aid] // cell_m)
            for gx in range(cx0, cx1 + 1):
                for gy in range(cy0, cy1 + 1):
                    self.grid.setdefault((gx, gy), []).append(aid)

    def query(self, x, y):
        """Arc ids whose padded bbox covers a cell around (x, y)."""
        c = (int(x // self.cell), int(y // self.cell))
        out = []
        for gx in (c[0] - 1, c[0], c[0] + 1):
            for gy in (c[1] - 1, c[1], c[1] + 1):
                out.extend(self.grid.get((gx, gy), ()))
        return np.unique(np.array(out, dtype=np.int64)) if out else np.empty(0, dtype=np.int64)


def generate_candidates(gps_lon, gps_lat, nodes, arcs, projector=None,
                        radius_m=SEARCH_RADIUS_M, top_k=TOP_K, grid=None):
    """Candidates for one GPS point -> list of dicts sorted by distance.

    Each dict: {arc_id, distance_m, rank}. Returns [] when nothing is within radius.
    """
    if projector is None:
        projector = TripProjector(nodes[:, 0], nodes[:, 1])
    if grid is None:
        nodes_xy = projector.to_m(nodes[:, 0], nodes[:, 1])
        grid = ArcGridIndex(nodes_xy, arcs, pad_m=radius_m)
    px, py = projector.to_m(gps_lon, gps_lat)
    cand = grid.query(px, py)
    if len(cand) == 0:
        return []
    d = _point_seg_dist_m(px, py, grid.ax[cand], grid.ay[cand], grid.bx[cand], grid.by[cand])
    keep = d <= radius_m
    cand, d = cand[keep], d[keep]
    order = np.argsort(d)[:top_k]
    return [{"arc_id": int(cand[i]), "distance_m": float(d[i]), "rank": r + 1}
            for r, i in enumerate(order)]


def build_trip_index(nodes, arcs):
    """Precompute the projector + grid for a trip (do this once per trip)."""
    projector = TripProjector(nodes[:, 0], nodes[:, 1])
    nodes_xy = projector.to_m(nodes[:, 0], nodes[:, 1])
    grid = ArcGridIndex(nodes_xy, arcs)
    return projector, grid


def proxy_true_arc(gps_lon, gps_lat, nodes, arcs, route, projector):
    """Nearest route arc to a GPS point (proxy per-point ground truth)."""
    px, py = projector.to_m(gps_lon, gps_lat)
    route = np.asarray(route, dtype=np.int64)
    nodes_xy = projector.to_m(nodes[:, 0], nodes[:, 1])
    d = _point_seg_dist_m(px, py,
                          nodes_xy[0][arcs[route, 0]], nodes_xy[1][arcs[route, 0]],
                          nodes_xy[0][arcs[route, 1]], nodes_xy[1][arcs[route, 1]])
    return int(route[int(np.argmin(d))])


def evaluate_candidates(track, nodes, arcs, route, radius_m=SEARCH_RADIUS_M, top_k=TOP_K):
    """Run candidate generation over one trip and compute recall statistics."""
    projector, grid = build_trip_index(nodes, arcs)
    n = len(track)
    n_with_cand = 0
    n_hit = 0
    cand_counts = []
    for i in range(n):
        cands = generate_candidates(track[i, 0], track[i, 1], nodes, arcs,
                                    projector=projector, radius_m=radius_m,
                                    top_k=top_k, grid=grid)
        cand_counts.append(len(cands))
        if not cands:
            continue
        n_with_cand += 1
        true_arc = proxy_true_arc(track[i, 0], track[i, 1], nodes, arcs, route, projector)
        if any(c["arc_id"] == true_arc for c in cands):
            n_hit += 1
    return {
        "n_points": n,
        "points_with_candidates": n_with_cand,
        "points_zero_candidates": n - n_with_cand,
        "avg_candidates": float(np.mean(cand_counts)),
        "candidate_recall": n_hit / n if n else 0.0,   # fraction of all points whose true arc is a candidate
    }
