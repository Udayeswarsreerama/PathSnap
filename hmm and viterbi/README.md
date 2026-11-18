# PathSnap - Member 3 (Srikanth): HMM + Viterbi Route Reconstruction

Member 3's part of the PathSnap hackathon: the HMM definition, Viterbi
decoder, end-to-end route reconstruction and the decoder evaluation
required by section 5 of the Deliverables document ("HMM and Viterbi Route
Reconstruction").

## Files

| File | What it is |
|---|---|
| `hmm.py` | HMM definition: hidden states, emission model, transition model, road-network shortest paths, `HMMConfig` |
| `viterbi.py` | Viterbi decoder: the recurrence, missing-candidate handling, GPS-gap handling |
| `candidate_scorer.py` | Adapter where Member 2's model plugs in, plus a baseline logistic scorer used until that model is ready |
| `route_reconstruction.py` | End-to-end per-trip chain: preprocessing -> candidates -> scores -> HMM -> Viterbi |
| `evaluate_decoder.py` | All comparison metrics: classifier-only vs classifier+HMM/Viterbi |
| `visualize_routes.py` | Folium maps: GPS points, road network, reference route, classifier route, decoded route |
| `run_evaluation.py` | Runs everything and writes all tables/plots/maps |
| `hmm_config.json` | HMM configuration (all decoder parameters in one place) |
| `member3_out/` | Generated results on the dataset subset (see below) |

## How to run

Layout: this `member3/` folder expects Member 1's pipeline next to it at
`../member1/pipeline/` (i.e. unzip `PathSnap_Member1_Obaid_v2.zip` into a
folder named `member1` beside this one - the pipeline code is imported,
not copied, per the team agreement).

```bash
pip install -r requirements.txt
python3 run_evaluation.py <data_root> [out_dir]
```

`<data_root>` is the folder containing `train/` and `student_test/`.
The code is identical for the subset and for the full Student folder -
just point it at the full folder:

```bash
python3 run_evaluation.py "/path/to/Student folder" member3_out
```

(The subset in this package has 15 train + 10 student_test trips; the full
dataset has 80 train + 10 student_test. Runtime on the subset is ~10 s.)

## The model (answers to Deliverables section 5)

1. **Hidden states.** At each GPS point t, the hidden states are that
   point's candidate road arcs from Member 1's candidate generation
   (30 m radius, top-5 by distance). The candidate-to-road-segment
   correspondence is preserved: each state IS an arc id.
2. **Emission probabilities.** Two evidence terms combined in log space:
   a Gaussian on the perpendicular point-to-arc distance
   (sigma from config; EDA measured GPS noise median 4.2 m, p90 9.7 m)
   and the classifier's probability for that (point, candidate) pair,
   normalised across the point's candidate set.
   `log P(obs|state) = w_dist * log N(d;0,sigma) + w_clf * log p_clf`.
3. **Transition probabilities.** Newson-Krumm style: the driving distance
   along the road network between two candidate positions (bounded
   Dijkstra on the arc graph, using each candidate's projection fraction
   along its arc) is compared with the distance the GPS trace actually
   travelled: `log P(a->b) = -|network_dist - gps_dist| / beta`.
4. **Missing candidates / GPS gaps.** Points with no candidates (~0.5% on
   the test trips) are skipped during decoding; the travelled distance is
   summed across the skip so the transition still compares like with like.
   After decoding, skipped points inherit the previous decoded arc in the
   same segment. Gaps > 10 s split the trip into segments that are decoded
   independently - the decoder never stitches a route across an outage.
5. **Disconnected segments.** Arc pairs with no network path within the
   search cutoff get a fixed large penalty instead of -inf, so one bad
   pair cannot kill the chain; Viterbi strongly prefers connected routes.
6. **Did HMM decoding help?** Yes - see the numbers below and
   `member3_out/metrics_summary.csv`. Cases where it does NOT help: points
   right after GPS outages (the travelled distance is unknown, so the
   transition evidence is meaningless) and points whose true arc is not in
   the candidate set at all (candidate miss rate ~1.7% - no decoder can
   recover those).

## Results on the FULL dataset (10 student_test trips, Member 2's final MLP)

Point-weighted over all test points, with Member 2's MLP classifier plugged
in as the emission model (Member 2's clean-deliverables model,
retrained with trips 00000033/00000054 excluded; run on all 88 trips,
2026-11-18):

| Metric | Classifier-only (Member 2 MLP) | Classifier + HMM/Viterbi |
|---|---|---|
| Point accuracy (vs nearest-route-arc proxy) | 0.795 | **0.842** |
| Route membership (predicted arc on reference route) | 0.844 | **0.911** |
| Reference-route coverage | 0.890 | **0.913** |
| Route continuity (connected consecutive predictions) | 0.955 | **0.998** |
| Invalid transitions (avg per trip) | 131.5 | **5.8** |

Candidate miss rate (true arc absent from candidate set): 1.7%.
Decoder parameters re-tuned on train trips only after the model swap:
sigma_m = 5, beta_m = 10 (sensitivity in `member3_out/fig_param_sweep.png`).
Points right after GPS gaps remain the hard case (~8%) - the travelled
distance is unknown there, so transition evidence is weak for ANY decoder.

Plots: `fig_point_accuracy.png`, `fig_continuity.png`,
`fig_param_sweep.png`. Interactive maps: `map_00000000.html`,
`map_00000007.html` (toggle the "Classifier-only route" layer against
"HMM/Viterbi route" to show the improvement).

## Member 2's model is plugged in

`mlp.joblib` (Member 2's MLP - best log loss and macro F1 in his comparison,
top-1 tied with the random forest) is wired in through `candidate_scorer.py`
in `artifact` mode: `hmm_config.json` points at the artifact and
`mlp.joblib.features.json` fixes the feature columns. The baseline logistic
scorer remains as a fallback (`"scorer": {"type": "baseline_logistic"}`).
Member 2's random forest scored 0.8022 top-1 but its artifact was too large
to share; swapping it in later is the same config change if he exports it.

The artifact is scored by a small numpy reimplementation of his
impute -> scale -> MLP pipeline (in `candidate_scorer.py`), so the model
works with ANY installed scikit-learn version - his was saved with 1.7.2
and newer versions cannot unpickle its internals reliably. Verified to
match the original pipeline's probabilities exactly.

Note on labels: per-point accuracy uses the nearest-route-arc proxy (the
.route file is an arc sequence, not per-point labels), same convention as
Member 1's candidate-recall numbers. Trips 00000033 and 00000054 are
excluded from ALL training, tuning and evaluation (mismatched ground
truth, confirmed in Member 1's v4 audit) - the exclusion list is in
`hmm_config.json`. Ships with Member 1's v4 pipeline (path fixes,
requirements.txt, Esri map tiles); core interfaces unchanged.
