"""
PathSnap - Evaluator UI (Laptop 2)
==================================

Streamlit interface for the evaluator. Talks to the FastAPI backend over the
network - set the API address to Laptop 1's IP in the sidebar (or via the
API_BASE_URL env var). Ground truth never reaches this UI; the API computes
accuracy server-side and returns only the results.

Run (Laptop 2):
    streamlit run ui.py
"""

import os

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

DEFAULT_API = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="PathSnap Evaluator", layout="wide")
st.title("PathSnap - Route Reconstruction Evaluator")

# --- Sidebar: API connection -------------------------------------------------
st.sidebar.header("API connection")
api_base = st.sidebar.text_input(
    "API base URL (Laptop 1 address)", value=DEFAULT_API,
    help="e.g. http://192.168.1.10:8000").rstrip("/")

try:
    health = requests.get(f"{api_base}/health", timeout=5).json()
    st.sidebar.success(f"API online - {health['num_trips']} trips available")
except Exception:
    st.sidebar.error("API not reachable. Check the address and that uvicorn "
                     "is running on Laptop 1 with --host 0.0.0.0.")
    st.stop()

# --- Trip selection ----------------------------------------------------------
try:
    trip_ids = requests.get(f"{api_base}/trips", timeout=10).json()["trips"]
except Exception as e:
    st.error(f"Could not load trips from the API: {e}")
    st.stop()

if not trip_ids:
    st.warning("No trips found on the API server. Check DATA_DIR on Laptop 1.")
    st.stop()

trip_id = st.selectbox("Select an evaluation trip", trip_ids)
run = st.button("Run prediction", type="primary")

if run and trip_id:
    with st.spinner(f"Running prediction for {trip_id} on Laptop 1..."):
        try:
            result = requests.post(f"{api_base}/predict/{trip_id}",
                                   timeout=300).json()
        except Exception as e:
            st.error(f"Prediction request failed: {e}")
            st.stop()
    st.session_state["result"] = result

result = st.session_state.get("result")
if not result:
    st.info("Select a trip and press Run prediction.")
    st.stop()

# --- Metrics -----------------------------------------------------------------
m = result["metrics"]
st.subheader(f"Results - {result['trip_id']}")
def percentage_metric(column, label, key):
    value = m.get(key)
    column.metric(label, "n/a" if value is None else f"{value:.1%}")


c1, c2, c3, c4 = st.columns(4)
percentage_metric(c1, "Point accuracy (HMM)", "point_top1_accuracy")
percentage_metric(c2, "Point accuracy (direction ignored)",
                  "point_accuracy_direction_agnostic")
percentage_metric(c3, "Route membership", "route_membership")
percentage_metric(c4, "Route continuity", "route_continuity")
st.caption("Point accuracy (HMM) uses the original directed-arc proxy. "
           "Direction ignored also accepts the same road's exact reversed "
           "endpoints. These are different metrics; neither changes the "
           "predicted route. Route membership checks whether each predicted "
           "arc appears anywhere in the reference route.")

comparison = st.columns(4)
percentage_metric(comparison[0], "Point accuracy (classifier only)",
                  "classifier_only_point_accuracy")
percentage_metric(comparison[1], "Route similarity (LCS)",
                  "route_similarity_lcs")
comparison[2].metric("Predicted arcs", m["num_predicted_arcs"])
comparison[3].metric("Runtime", f"{m['runtime_ms']:.0f} ms")

with st.expander("Predicted road sequence (arc IDs)"):
    st.code(", ".join(str(a) for a in result["predicted_road_sequence"]))

# --- Folium map: raw GPS vs reconstructed route -------------------------------
gps = result["gps_points"]
route = result["reconstructed_route"]
center = gps[len(gps) // 2] if gps else [0, 0]

fmap = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")
folium.PolyLine(route, color="#2ca02c", weight=5, opacity=0.9,
                tooltip="Reconstructed route").add_to(fmap)
for lat, lon in gps:
    folium.CircleMarker([lat, lon], radius=3, color="#1f77b4", fill=True,
                        fill_opacity=0.7).add_to(fmap)
if gps:
    fmap.fit_bounds(gps + route)

# Legend
fmap.get_root().html.add_child(folium.Element("""
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
     background: white; padding: 10px 14px; border: 1px solid #888;
     border-radius: 6px; font-size: 14px;">
  <b>Legend</b><br>
  <span style="color:#1f77b4;">&#9679;</span> Raw GPS observations<br>
  <span style="color:#2ca02c;">&#9473;</span> Reconstructed route
</div>
"""))

st_folium(fmap, use_container_width=True, height=560, returned_objects=[])
st.caption("Blue dots: raw GPS observations. Green line: reconstructed route "
           "returned by the API. Accuracy is computed on Laptop 1 against "
           "the server-side reference route.")
