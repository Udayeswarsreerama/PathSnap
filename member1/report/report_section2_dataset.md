# 2. Dataset Understanding and Exploratory Data Analysis

## 2.1 Dataset structure

The PathSnap dataset consists of 90 self-contained trips in two supplied partitions: 80 training trips (`train/`) and 10 student-testing trips (`student_test/`). (A third, evaluator-testing partition is held by the organizers and was never touched.)

**Partitions used per analysis, as required by the submission guidelines:** dataset structure and inventory statistics (Section 2.2) are reported across both supplied partitions and labelled; data-quality, sampling-rate and connectivity analyses (Sections 2.3-2.5) were computed on all 90 supplied trips, since these describe the raw data rather than any learned model; the GPS noise estimate (Section 2.6) and the radius tuning for candidate generation used training trips only; candidate recall is reported on the 80 training trips, with the student-testing partition used only as a development-stage sanity check (98.2% macro recall, consistent with training). No evaluator-testing data was used anywhere. Each trip is a folder named by its trip id and contains four files:

| File | Contents | Format |
|---|---|---|
| `<id>.track` | Noisy GPS trace | `longitude  latitude  timestamp` per row |
| `<id>.nodes` | Road-network nodes | `longitude  latitude` per row; node id = line index |
| `<id>.arcs` | Directed road arcs | `from_node  to_node` (indices into the nodes file) |
| `<id>.route` | Ground-truth route | Ordered list of arc ids actually travelled |

Every trip carries its own road network sized to the region the trace covers, so the network size varies strongly between trips. The trips are drawn from many different cities worldwide (mostly Europe, with some in North America and Asia), which means no single coordinate reference can be assumed; all distance computations must be done in metres (we use an equirectangular approximation).

## 2.2 Dataset statistics

Computed over all 90 supplied trips (80 train + 10 student-testing):

| Quantity | min | median | mean | max |
|---|---|---|---|---|
| GPS points per trip | 262 | 1,496 | 2,529 | 19,313 |
| Road-network nodes per trip | 299 | 9,305 | 22,491 | 155,003 |
| Directed arcs per trip | 649 | 18,762 | 45,133 | 307,392 |
| Ground-truth route length (arcs, train) | 59 | 280 | 392 | 2,266 |
| Track duration (s) | 265 | 1,710 | 5,274 | 154,030 |

The dataset contains **227,577 GPS observations** in total. All 90 trips include a valid, non-empty ground-truth `.route`; every route arc id is a valid index into that trip's arc list. (Note: the 10 `student_test` trips in the distributed copy also include `.route` files.)

## 2.3 Missing and invalid values

The dataset is remarkably clean. Across all 90 trips we found **zero malformed lines, zero out-of-range coordinates, and zero non-monotonic timestamps**. Only 6 consecutive duplicate GPS fixes exist in the entire dataset. No imputation or coordinate repair is required; preprocessing effort can focus on map-matching features rather than cleaning.

## 2.4 Sampling rate and timestamp gaps

The documentation states that roughly 97% of samples are recorded at 1 Hz. We verified this directly: **96.68% of the 227,487 consecutive-point intervals are exactly 1 second** (median gap 1.0 s, mean 2.09 s), and 98.65% of intervals are at most 2 s. Only 0.55% of intervals exceed 5 s, but a small number of trips contain very long pauses (the largest single gap is about 153,064 s, i.e. a multi-hour stop inside one trace). The pipeline must therefore tolerate occasional large gaps, for example by down-weighting transition evidence across them.

## 2.5 Road-network connectivity

Treating each trip's network as a directed graph and analysing it with union-find:

- Weakly connected components per trip: median 19 (min 1, max 421)
- The largest component covers a median of **98.0% of nodes** (min 69.3%) — each network is essentially one connected region with small dangling fragments
- Dead ends (out-degree 0): median 5.5 nodes per trip (max 81)
- Source nodes (in-degree 0): median 5 nodes per trip

For the HMM reconstruction this means transition probabilities must respect arc direction and connectivity, and candidate generation can safely focus on the main component.

## 2.6 GPS noise level

We estimated the noise on training trip 00000001 by measuring the perpendicular distance from 800 randomly sampled raw GPS points to the nearest ground-truth route arc:

- median **4.2 m**, mean **4.8 m**, 90th percentile **9.7 m**, maximum **27.2 m**

This is typical consumer-GPS accuracy and directly informs the HMM design: a Gaussian emission probability with sigma in the 5-10 m range and a candidate search radius of about 25-30 m per point are appropriate.

## 2.7 Visualisations

Figure 1 shows the worldwide distribution of trip locations and one example raw trajectory. Figure 2 renders a trip's full road network. Figure 3 overlays the noisy GPS trace and the ground-truth route on that network, illustrating both the noise level and the sparsity of the trace relative to the network. Figure 4 shows the distributions of points, nodes, arcs and route lengths across trips, and Figure 5 the sampling-interval distribution. An interactive Folium map (`folium_map.html`) allows zooming into the example trip with the network, GPS points and ground truth as toggleable layers.

One limitation found during validation: two training trips (00000033, 00000054) show systematic offsets between the GPS trace and the supplied reference route (about 400 m and about 6.5 km respectively), so their per-point reference labels are unreliable. Following the dataset guide's advice that supplied reference routes may have limitations, we exclude these two trips from training and report candidate recall both with and without them rather than overstating label certainty.

## 2.8 Implications for the approach

1. Distances must be computed in metres, not degrees (trips span many latitudes).
2. Candidate arcs per GPS point: spatial lookup within ~30 m of the point, restricted to the main network component.
3. Emission model: Gaussian in distance with sigma ~5-10 m.
4. Transition model: must respect directed-arc connectivity and handle rare large time gaps.
5. No data cleaning is needed; the effort goes into candidate generation and sequence reconstruction.
