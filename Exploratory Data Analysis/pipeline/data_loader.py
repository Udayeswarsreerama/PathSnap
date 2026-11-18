"""PathSnap data loader.

Loads the per-trip files of the PathSnap dataset:
  <trip_id>.track  - noisy GPS trace: "lon lat timestamp" per line
  <trip_id>.nodes  - road network nodes: "lon lat" per line (node id = line index)
  <trip_id>.arcs   - directed road arcs: "from_node to_node" per line (arc id = line index)
  <trip_id>.route  - ground-truth route: one arc id per line (train trips)

All functions return plain numpy arrays / lists so every member can rely on the
same structures.
"""
import os
import numpy as np

TRACK_COLS = ("lon", "lat", "ts")


def trip_dirs(data_root, split):
    """Return sorted trip folder paths for a split ('train' or 'student_test')."""
    base = os.path.join(data_root, split)
    return sorted(os.path.join(base, d) for d in os.listdir(base)
                  if os.path.isdir(os.path.join(base, d)))


def load_track(path):
    """Load a .track file -> (N,3) float array [lon, lat, ts]."""
    pts = []
    with open(path) as f:
        for line in f:
            p = line.strip().split("\t")
            if len(p) != 3:
                p = line.split()
            if len(p) != 3:
                continue
            pts.append((float(p[0]), float(p[1]), float(p[2])))
    return np.array(pts) if pts else np.zeros((0, 3))


def load_nodes(path):
    """Load a .nodes file -> (N,2) float array [lon, lat]. Node id = row index."""
    out = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            p = line.split("\t") if "\t" in line else line.split()
            out.append((float(p[0]), float(p[1])))
    return np.array(out)


def load_arcs(path):
    """Load an .arcs file -> (M,2) int array [from_node, to_node]. Arc id = row index."""
    out = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            p = line.split("\t") if "\t" in line else line.split()
            out.append((int(p[0]), int(p[1])))
    return np.array(out, dtype=np.int64)


def load_route(path):
    """Load a .route file -> list of arc ids, or None if the file does not exist."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [int(x) for x in f.read().split()]


def load_trip(trip_dir):
    """Load one trip folder -> dict with keys id, track, nodes, arcs, route."""
    tid = os.path.basename(trip_dir)
    return {
        "id": tid,
        "track": load_track(os.path.join(trip_dir, tid + ".track")),
        "nodes": load_nodes(os.path.join(trip_dir, tid + ".nodes")),
        "arcs": load_arcs(os.path.join(trip_dir, tid + ".arcs")),
        "route": load_route(os.path.join(trip_dir, tid + ".route")),
    }


def load_dataset(data_root, splits=("train", "student_test")):
    """Load every trip of the given splits -> {split: [trip dict, ...]}."""
    return {split: [load_trip(d) for d in trip_dirs(data_root, split)] for split in splits}
