"""PathSnap Member 3 - Folium visualisation of reconstructed routes.

Builds an interactive map for one trip with toggleable layers:

  * GPS observations        - blue dots (the raw noisy trace)
  * road network            - thin grey lines (arcs near the trajectory)
  * reference route         - green line (ground truth, where available)
  * classifier-only route   - orange dashed line (argmax candidate per point)
  * HMM/Viterbi route       - thick red line (decoded route)

Consecutive points decoded onto the same arc are collapsed into runs, so
the predicted route is drawn arc-by-arc rather than as per-point jumps.
"""
import numpy as np


def _arc_latlon(nodes, arcs, arc_id):
    a, b = int(arcs[arc_id, 0]), int(arcs[arc_id, 1])
    return [(float(nodes[a, 1]), float(nodes[a, 0])),
            (float(nodes[b, 1]), float(nodes[b, 0]))]


def _runs(pred):
    """Collapse a per-point arc sequence into (arc_id, start, end) runs."""
    runs = []
    for i, a in enumerate(pred):
        if a is None:
            continue
        if runs and runs[-1][0] == a and runs[-1][2] == i - 1:
            runs[-1][2] = i
        else:
            runs.append([a, i, i])
    return runs


def make_map(prepared, out_html, title=None):
    """Write the interactive Folium map for one decoded trip."""
    import folium

    trip = prepared["trip"]
    nodes, arcs = trip["nodes"], trip["arcs"]
    gps = prepared["gps_df"]
    centre = [float(gps["lat"].mean()), float(gps["lon"].mean())]
    m = folium.Map(location=centre, zoom_start=14, tiles="OpenStreetMap")

    # road network, restricted to the trajectory's bounding box
    pad = 0.01
    lat0, lat1 = gps["lat"].min() - pad, gps["lat"].max() + pad
    lon0, lon1 = gps["lon"].min() - pad, gps["lon"].max() + pad
    fg_net = folium.FeatureGroup(name="Road network", show=True)
    in_view = ((nodes[arcs[:, 0], 1] >= lat0) & (nodes[arcs[:, 0], 1] <= lat1) &
               (nodes[arcs[:, 0], 0] >= lon0) & (nodes[arcs[:, 0], 0] <= lon1))
    for aid in np.where(in_view)[0]:
        folium.PolyLine(_arc_latlon(nodes, arcs, aid), color="#bbbbbb",
                        weight=1.5, opacity=0.6).add_to(fg_net)
    fg_net.add_to(m)

    # raw GPS observations
    fg_gps = folium.FeatureGroup(name="GPS observations", show=True)
    for _, r in gps.iterrows():
        folium.CircleMarker([r["lat"], r["lon"]], radius=2, color="#1f77b4",
                            fill=True, fill_opacity=0.7).add_to(fg_gps)
    fg_gps.add_to(m)

    # reference route
    if trip.get("route"):
        fg_ref = folium.FeatureGroup(name="Reference route", show=True)
        for aid in trip["route"]:
            folium.PolyLine(_arc_latlon(nodes, arcs, aid), color="#2ca02c",
                            weight=3, opacity=0.8).add_to(fg_ref)
        fg_ref.add_to(m)

    # classifier-only route (per-point argmax, drawn as arc runs)
    fg_clf = folium.FeatureGroup(name="Classifier-only route", show=False)
    for aid, _, _ in _runs(prepared["clf_top1"]):
        folium.PolyLine(_arc_latlon(nodes, arcs, aid), color="#ff7f0e",
                        weight=3, opacity=0.8, dash_array="6").add_to(fg_clf)
    fg_clf.add_to(m)

    # HMM/Viterbi route
    fg_vit = folium.FeatureGroup(name="HMM/Viterbi route", show=True)
    for aid, _, _ in _runs(prepared["viterbi"]):
        folium.PolyLine(_arc_latlon(nodes, arcs, aid), color="#d62728",
                        weight=5, opacity=0.9).add_to(fg_vit)
    fg_vit.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    if title:
        folium.map.Marker(
            centre,
            icon=folium.DivIcon(html=f'<div style="font-size:14pt;'
                                     f'font-weight:bold">{title}</div>'),
        ).add_to(m)
    m.save(out_html)
    return out_html
