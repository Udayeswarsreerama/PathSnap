"""PathSnap preprocessing (Member 1, Task 2).

Cleans raw GPS trajectories and adds movement quantities used downstream:
  - coordinate validity check (lon in [-180,180], lat in [-90,90])
  - chronological sort per trip
  - haversine distance between consecutive points (metres)
  - time difference between consecutive points (seconds)
  - approximate speed (m/s)
  - large-gap flag

Gap threshold: 10 s. Justification from EDA: 98.65% of all sampling intervals
are <= 2 s and only 0.55% exceed 5 s, so anything beyond 10 s is a genuine
sampling outage, not normal jitter. Points after a gap keep their observations
(no interpolation); the flag lets the HMM down-weight transition evidence.

Nothing is deleted by default: the EDA found 0 malformed lines, 0 invalid
coordinates and 0 non-monotonic timestamps, so preprocessing adds information
rather than throwing data away. Set drop_invalid=True to hard-drop invalid rows.
"""
import numpy as np
import pandas as pd

EARTH_R = 6371000.0  # metres
GAP_THRESHOLD_S = 10.0
MAX_PLAUSIBLE_SPEED_MPS = 55.0  # ~200 km/h; faster => GPS jump, speed unreliable


def haversine_m(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in metres."""
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 to point 2, degrees in [0, 360)."""
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def preprocess_trajectory(track, gap_threshold_s=GAP_THRESHOLD_S, drop_invalid=False):
    """Clean one trip's track and compute movement columns.

    Input:  (N,3) array [lon, lat, ts] (from data_loader.load_track)
    Output: pandas DataFrame with columns
        lon, lat, ts, valid, dist_prev_m, dt_prev_s, speed_mps, heading_deg, gap_before
    """
    track = np.asarray(track)
    if track.size == 0:
        return pd.DataFrame(columns=["lon", "lat", "ts", "valid", "dist_prev_m",
                                   "dt_prev_s", "speed_mps", "heading_deg",
                                   "gap_before", "speed_unreliable", "segment_id"])
    track = track.reshape((-1, 3))
    df = pd.DataFrame({"lon": track[:, 0], "lat": track[:, 1], "ts": track[:, 2]})
    df["valid"] = (df["lat"].between(-90, 90) & df["lon"].between(-180, 180)
                   & df["lat"].notna() & df["lon"].notna() & df["ts"].notna())
    if drop_invalid:
        df = df[df["valid"]].reset_index(drop=True)
    df = df.sort_values("ts").reset_index(drop=True)

    lat_prev = df["lat"].shift(1)
    lon_prev = df["lon"].shift(1)
    dist = np.concatenate([[np.nan], haversine_m(df["lat"].values[:-1], df["lon"].values[:-1],
                                                 df["lat"].values[1:], df["lon"].values[1:])])
    df["dist_prev_m"] = dist
    df["dt_prev_s"] = df["ts"].diff()
    # speed only where the time step is usable
    with np.errstate(divide="ignore", invalid="ignore"):
        df["speed_mps"] = np.where(df["dt_prev_s"] > 0,
                                   df["dist_prev_m"] / df["dt_prev_s"], np.nan)
    head = np.concatenate([[np.nan], bearing_deg(df["lat"].values[:-1], df["lon"].values[:-1],
                                                 df["lat"].values[1:], df["lon"].values[1:])])
    df["heading_deg"] = head
    df["gap_before"] = df["dt_prev_s"] > gap_threshold_s
    # speed that implies a GPS jump rather than real motion (kept, but flagged)
    df["speed_unreliable"] = df["speed_mps"] > MAX_PLAUSIBLE_SPEED_MPS
    # segment id: increments at every large gap, so the HMM can avoid
    # stitching a route across a multi-minute/hour sampling outage
    df["segment_id"] = df["gap_before"].cumsum()
    return df


def trip_statistics(df, trip_id):
    """One-row summary of a preprocessed trip."""
    return {
        "trip_id": trip_id,
        "n_points": len(df),
        "n_invalid": int((~df["valid"]).sum()),
        "duration_s": float(df["ts"].iloc[-1] - df["ts"].iloc[0]) if len(df) > 1 else 0.0,
        "path_length_m": float(df["dist_prev_m"].sum()),
        "n_gaps": int(df["gap_before"].sum()),
        "n_segments": int(df["segment_id"].nunique()),
        "n_speed_unreliable": int(df["speed_unreliable"].sum()),
        "median_speed_mps": float(df["speed_mps"].median()),
        "max_speed_mps": float(df["speed_mps"].max()),
    }
