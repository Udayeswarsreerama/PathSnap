# PathSnap EDA - plain-English summary for the 11:30 evaluation

What we did: loaded all 90 trips (80 train, 10 test), parsed every file, computed statistics, checked data quality, verified the sampling rate, analysed the road networks as directed graphs, measured the GPS noise, and produced plots plus an interactive map.

Key numbers to quote:

- **90 trips, 227,577 GPS points total.** Each trip brings its own road network (median ~9,300 nodes and ~18,800 directed arcs; the biggest has 155k nodes / 307k arcs).
- **Trips come from many cities worldwide** (mostly Europe, some US/Asia), so we do all distance math in metres, not degrees.
- **The data is clean**: zero malformed lines, zero invalid coordinates, zero out-of-order timestamps, 6 duplicate fixes in the whole dataset. No cleaning needed.
- **The 1 Hz claim checks out**: 96.7% of consecutive GPS points are exactly 1 second apart; 98.7% within 2 seconds. A few trips have long pauses (worst gap ~42 hours), so our model must handle occasional gaps.
- **Road networks are essentially connected**: the largest connected component covers ~98% of nodes on average; only a handful of dead ends (median 5-6 per trip).
- **GPS noise is small**: typical error ~4 m (median), 90% of points within ~10 m of the true road, worst case ~27 m. This tells us the HMM emission sigma should be ~5-10 m and we only need to look for candidate roads within ~30 m of each GPS point.
- **Ground-truth routes are valid** in every trip (all arc ids check out). Heads-up: the test trips in our copy also include the ground-truth route files - we'll confirm with the organizers whether that's intended.

What this means for the model (one line): the problem is well-posed for an HMM - noisy points sit within ~10 m of the true road, the network is connected, and 1-second sampling gives strong transition evidence between consecutive points.

Deliverables ready: EDA.ipynb (all code + figures), 6 figures, folium_map.html interactive map.
