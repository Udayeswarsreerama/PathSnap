"""PathSnap EDA - computes all stats and figures. Saves results.json + figures."""
import os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# Paths: set env vars PATHSNAP_DATA / PATHSNAP_OUT, or place the dataset in ./data
DATA = os.environ.get("PATHSNAP_DATA", "data")
OUT = os.environ.get("PATHSNAP_OUT", "eda_outputs")
os.makedirs(OUT, exist_ok=True)

def trip_dirs(split):
    base = os.path.join(DATA, split)
    return sorted(os.path.join(base, d) for d in os.listdir(base))

def read_track(path):
    """Returns (pts Nx3 array, n_malformed, n_invalid_coords)."""
    pts, malformed, invalid = [], 0, 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                parts = line.split()
            if len(parts) != 3:
                malformed += 1
                continue
            try:
                lon, lat, ts = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                malformed += 1
                continue
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                invalid += 1
                continue
            pts.append((lon, lat, ts))
    return np.array(pts) if pts else np.zeros((0, 3)), malformed, invalid

def read_nodes(path):
    nodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split("\t")
            if len(p) != 2:
                p = line.split()
            nodes.append((float(p[0]), float(p[1])))
    return np.array(nodes)

def read_arcs(path):
    arcs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split("\t")
            if len(p) != 2:
                p = line.split()
            arcs.append((int(p[0]), int(p[1])))
    return np.array(arcs, dtype=np.int64)

def read_route(path):
    with open(path) as f:
        return [int(x) for x in f.read().split()]

# ---------- union-find for components ----------
class UF:
    def __init__(self, n):
        self.p = list(range(n))
        self.sz = [1]*n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.sz[ra] += self.sz[rb]

R = {"trips": {}}
all_point_counts, all_gaps = [], []
rows = []

for split in ["train", "student_test"]:
    dirs = trip_dirs(split)
    R["trips"][split] = {"count": len(dirs), "ids": [os.path.basename(d) for d in dirs]}
    for d in dirs:
        tid = os.path.basename(d)
        track, mal, inv = read_track(os.path.join(d, tid + ".track"))
        nodes = read_nodes(os.path.join(d, tid + ".nodes"))
        arcs = read_arcs(os.path.join(d, tid + ".arcs"))
        rpath = os.path.join(d, tid + ".route")
        route = read_route(rpath) if os.path.exists(rpath) else None

        n_pts = len(track)
        all_point_counts.append(n_pts)
        # timestamps
        ts = track[:, 2]
        gaps = np.diff(ts)
        all_gaps.append(gaps)
        nonmono = int(np.sum(gaps <= 0))
        # dup consecutive points
        dups = int(np.sum((np.diff(track[:, 0]) == 0) & (np.diff(track[:, 1]) == 0)))
        # connectivity
        n_nodes = len(nodes)
        uf = UF(n_nodes)
        for a, b in arcs:
            uf.union(int(a), int(b))
        roots = {}
        for i in range(n_nodes):
            r = uf.find(i)
            roots[r] = roots.get(r, 0) + 1
        comp_sizes = sorted(roots.values(), reverse=True)
        n_comp = len(comp_sizes)
        largest = comp_sizes[0]
        outdeg = np.zeros(n_nodes, dtype=np.int64)
        indeg = np.zeros(n_nodes, dtype=np.int64)
        for a, b in arcs:
            outdeg[int(a)] += 1
            indeg[int(b)] += 1
        dead_ends = int(np.sum(outdeg == 0))
        sources = int(np.sum(indeg == 0))
        # route validity
        route_ok = None
        if route is not None:
            route_ok = all(0 <= r < len(arcs) for r in route)
        rows.append(dict(split=split, trip=tid, gps_points=n_pts,
                         nodes=n_nodes, arcs=len(arcs),
                         route_len=len(route) if route else None,
                         malformed=mal, invalid=inv, dup_consec=dups,
                         nonmono_ts=nonmono,
                         duration_s=float(ts[-1]-ts[0]) if n_pts > 1 else 0.0,
                         gap_mean=float(gaps.mean()) if len(gaps) else None,
                         gap_max=float(gaps.max()) if len(gaps) else None,
                         pct_1s=float(np.mean(gaps == 1.0)*100) if len(gaps) else None,
                         components=n_comp, largest_comp=largest,
                         largest_comp_pct=100.0*largest/n_nodes,
                         dead_ends=dead_ends, sources=sources,
                         route_valid=route_ok))

R["per_trip"] = rows

def agg(key, filt=None):
    vals = [r[key] for r in rows if r[key] is not None and (filt is None or filt(r))]
    a = np.array(vals, dtype=float)
    return dict(min=float(a.min()), median=float(np.median(a)), max=float(a.max()), mean=float(a.mean()))

R["summary"] = {
    "total_trips": len(rows),
    "gps_points_per_trip": agg("gps_points"),
    "total_gps_points": int(sum(r["gps_points"] for r in rows)),
    "nodes_per_trip": agg("nodes"),
    "arcs_per_trip": agg("arcs"),
    "route_len_train": agg("route_len", lambda r: r["split"] == "train"),
    "duration_s": agg("duration_s"),
    "malformed_lines_total": int(sum(r["malformed"] for r in rows)),
    "invalid_coords_total": int(sum(r["invalid"] for r in rows)),
    "dup_consec_total": int(sum(r["dup_consec"] for r in rows)),
    "nonmono_ts_total": int(sum(r["nonmono_ts"] for r in rows)),
    "components_per_trip": agg("components"),
    "largest_comp_pct": agg("largest_comp_pct"),
    "dead_ends_per_trip": agg("dead_ends"),
    "sources_per_trip": agg("sources"),
    "route_all_valid": all(r["route_valid"] for r in rows if r["route_valid"] is not None),
    "trips_with_route": sum(1 for r in rows if r["route_len"]),
}

gaps_all = np.concatenate([g for g in all_gaps if len(g)])
R["gaps"] = {
    "total_intervals": int(len(gaps_all)),
    "pct_exactly_1s": float(np.mean(gaps_all == 1.0) * 100),
    "pct_le_1s": float(np.mean(gaps_all <= 1.0) * 100),
    "pct_le_2s": float(np.mean(gaps_all <= 2.0) * 100),
    "pct_gt_5s": float(np.mean(gaps_all > 5.0) * 100),
    "max_gap_s": float(gaps_all.max()),
    "mean_gap_s": float(gaps_all.mean()),
    "median_gap_s": float(np.median(gaps_all)),
}

# ---------------- figures ----------------
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

# Fig 1: GPS trajectories - all train trips overlaid (subsample markers), one highlighted
fig, ax = plt.subplots(figsize=(8, 8))
for i, d in enumerate(trip_dirs("train")):
    tid = os.path.basename(d)
    t, _, _ = read_track(os.path.join(d, tid + ".track"))
    ax.plot(t[:, 0], t[:, 1], ".", ms=0.4, alpha=0.25, color="steelblue")
d0 = trip_dirs("train")[0]
tid0 = os.path.basename(d0)
t0, _, _ = read_track(os.path.join(d0, tid0 + ".track"))
ax.plot(t0[:, 0], t0[:, 1], "-", lw=1.2, color="crimson", label=f"trip {tid0} (connected in time order)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("GPS trajectories: all 80 training trips (blue) + one example trip (red)")
ax.legend(); ax.set_aspect("equal", adjustable="datalim")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_gps_trajectories.png")); plt.close(fig)

# Fig 2: road network of example trip (subsample arcs)
nodes0 = read_nodes(os.path.join(d0, tid0 + ".nodes"))
arcs0 = read_arcs(os.path.join(d0, tid0 + ".arcs"))
fig, ax = plt.subplots(figsize=(8, 8))
step = max(1, len(arcs0)//60000)
for a, b in arcs0[::step]:
    ax.plot([nodes0[a, 0], nodes0[b, 0]], [nodes0[a, 1], nodes0[b, 1]],
            "-", lw=0.3, color="gray", alpha=0.5)
ax.plot(nodes0[::20, 0], nodes0[::20, 1], ".", ms=0.3, color="black", alpha=0.4)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title(f"Road network of trip {tid0}: {len(nodes0):,} nodes, {len(arcs0):,} directed arcs (subsampled)")
ax.set_aspect("equal", adjustable="datalim")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_road_network.png")); plt.close(fig)

# Fig 3: GPS overlaid on network + ground-truth route
route0 = read_route(os.path.join(d0, tid0 + ".route"))
fig, ax = plt.subplots(figsize=(8, 8))
for a, b in arcs0[::step]:
    ax.plot([nodes0[a, 0], nodes0[b, 0]], [nodes0[a, 1], nodes0[b, 1]],
            "-", lw=0.3, color="lightgray", alpha=0.6)
ax.plot(t0[:, 0], t0[:, 1], ".", ms=2.5, color="steelblue", label="noisy GPS points")
# ground truth route polyline
rt = []
for aid in route0:
    a, b = arcs0[aid]
    rt.append(nodes0[a]); rt.append(nodes0[b])
rt = np.array(rt)
ax.plot(rt[:, 0], rt[:, 1], "-", lw=1.6, color="crimson", label="ground-truth route")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title(f"Trip {tid0}: noisy GPS vs road network vs ground-truth route")
ax.legend(); ax.set_aspect("equal", adjustable="datalim")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_gps_vs_route.png")); plt.close(fig)

# Fig 4: distributions
fig, axs = plt.subplots(2, 2, figsize=(11, 8))
axs[0,0].hist(all_point_counts, bins=20, color="steelblue")
axs[0,0].set_title("GPS points per trip"); axs[0,0].set_xlabel("points")
axs[0,1].hist([r["nodes"] for r in rows], bins=20, color="seagreen")
axs[0,1].set_title("Road-network nodes per trip"); axs[0,1].set_xlabel("nodes")
axs[1,0].hist([r["arcs"] for r in rows], bins=20, color="darkorange")
axs[1,0].set_title("Directed arcs per trip"); axs[1,0].set_xlabel("arcs")
rl = [r["route_len"] for r in rows if r["route_len"]]
axs[1,1].hist(rl, bins=20, color="crimson")
axs[1,1].set_title("Ground-truth route length (arcs) per trip"); axs[1,1].set_xlabel("arcs in route")
for a in axs.flat: a.set_ylabel("trips")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_distributions.png")); plt.close(fig)

# Fig 5: timestamp gap histogram
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(gaps_all[gaps_all <= 10], bins=np.arange(0, 10.5, 0.5), color="steelblue")
ax.set_title(f"Sampling interval distribution ({len(gaps_all):,} consecutive-point intervals)")
ax.set_xlabel("gap (seconds)"); ax.set_ylabel("count")
ax.axvline(1.0, color="crimson", ls="--", label="1 s (1 Hz)")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_timestamp_gaps.png")); plt.close(fig)

# Fig 6: GPS noise distance to nearest route arc (example trip) - perpendicular distance
def point_seg_dist(p, a, b):
    # equirectangular approx in meters around lat
    lat0 = math.radians(p[1])
    kx = 111320.0 * math.cos(lat0); ky = 110540.0
    pa = np.array([(a[0]-p[0])*kx, (a[1]-p[1])*ky])
    pb = np.array([(b[0]-p[0])*kx, (b[1]-p[1])*ky])
    ba = pb - pa
    L2 = ba @ ba
    if L2 == 0:
        return float(pa @ pa) ** 0.5
    t = max(0.0, min(1.0, -(pa @ ba) / L2))
    proj = pa + t * ba
    return float(proj @ proj) ** 0.5

# build per-node set of route arcs for speed: sample 800 gps points, check against route arcs only
rng = np.random.default_rng(0)
sample_idx = rng.choice(len(t0), size=min(800, len(t0)), replace=False)
dists = []
route_arc_pairs = [(nodes0[arcs0[aid][0]], nodes0[arcs0[aid][1]]) for aid in route0]
for i in sample_idx:
    p = t0[i]
    dists.append(min(point_seg_dist(p, a, b) for a, b in route_arc_pairs))
dists = np.array(dists)
R["noise"] = {
    "example_trip": tid0,
    "sampled_points": int(len(dists)),
    "dist_to_route_m": dict(min=float(dists.min()), median=float(np.median(dists)),
                            mean=float(dists.mean()), p90=float(np.percentile(dists, 90)),
                            max=float(dists.max())),
}
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(dists, bins=40, color="purple")
ax.set_title(f"GPS noise: distance from raw GPS point to ground-truth route (trip {tid0}, {len(dists)} sampled points)")
ax.set_xlabel("distance to nearest route arc (m)"); ax.set_ylabel("points")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_gps_noise.png")); plt.close(fig)

# Fig 7: Folium map for example trip
import folium
center = [float(np.mean(t0[:, 1])), float(np.mean(t0[:, 0]))]
m = folium.Map(location=center, zoom_start=13, tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}", attr="Tiles &copy; Esri - Source: Esri, DeLorme, NAVTEQ, USGS, Intermap, iPC, NRCAN, Esri Japan, METI, Esri China (Hong Kong), Esri (Thailand), TomTom, 2012")
# network subsample as one feature group
fg_net = folium.FeatureGroup(name="Road network (subsampled)")
net_lines = [[[float(nodes0[b,1]), float(nodes0[b,0])], [float(nodes0[a,1]), float(nodes0[a,0])]]
             for a, b in arcs0[::step*2]]
folium.PolyLine(locations=[pt for line in net_lines for pt in line], color="gray", weight=1, opacity=0.5).add_to(fg_net)
fg_net.add_to(m)
# ground truth route
folium.PolyLine(locations=[[float(p[1]), float(p[0])] for p in rt], color="red", weight=3,
                tooltip="Ground-truth route").add_to(m)
# GPS points
fg_gps = folium.FeatureGroup(name="Noisy GPS points")
for p in t0[::2]:
    folium.CircleMarker(location=[float(p[1]), float(p[0])], radius=2, color="blue",
                        fill=True, fill_opacity=0.7).add_to(fg_gps)
fg_gps.add_to(m)
folium.LayerControl().add_to(m)
m.save(os.path.join(OUT, "folium_map.html"))

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(R, f, indent=1)
print("DONE")
print(json.dumps(R["summary"], indent=1))
print(json.dumps(R["gaps"], indent=1))
print(json.dumps(R["noise"], indent=1))
