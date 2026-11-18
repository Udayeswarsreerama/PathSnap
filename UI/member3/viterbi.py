"""PathSnap Member 3 - Viterbi decoder.

Decodes the most probable sequence of road arcs (hidden states) for one
trip, given per-candidate emission log-probabilities and the transition
model from hmm.py.

Recurrence (standard Viterbi, in log space):

    delta_t(j) = emit_t(j) + max_i [ delta_{t-1}(i) + log P(i -> j) ]
    psi_t(j)   = argmax_i [ ... ]            (backpointer)

Handling of the dataset's real-world quirks:

  * Missing candidates: points whose candidate set is empty (~3.5% of
    points at r=30 m in Member 1's measurements) cannot be states. They are
    skipped during decoding; the GPS distance across the skip is summed so
    the transition between the surrounding decoded points still compares
    network distance with the actual travelled distance. After decoding,
    every skipped point inherits the arc of the previous decoded point
    inside the same segment (a leading run inherits the first decoded arc),
    so a prediction is returned for EVERY relevant GPS observation.

  * GPS gaps: preprocessing splits each trip into segments at intervals
    longer than the gap threshold (10 s). Segments are decoded as
    independent chains - the decoder never stitches a route across a GPS
    outage, where the travelled distance is unknown.

  * Disconnected transitions: arc pairs with no network path within the
    cutoff receive a fixed large penalty (config.unreachable_log_penalty)
    instead of -inf, so one bad candidate pair cannot kill the whole chain.
"""
import numpy as np


def viterbi_decode(emissions, transitions):
    """Core Viterbi algorithm on a dense candidate grid.

    emissions    : list of T arrays; emissions[t][j] = log P(obs_t | state j)
    transitions  : callable (t, i, j) -> log P(state i at t-1 -> state j at t),
                   where t indexes the points WITH candidates
    Returns the list of chosen candidate indices (one per point with
    candidates).
    """
    T = len(emissions)
    if T == 0:
        return []
    delta = emissions[0].copy()
    back = []
    for t in range(1, T):
        prev_k = len(delta)
        cur_k = len(emissions[t])
        # score[i, j] = delta[i] + trans(i -> j)
        score = np.empty((prev_k, cur_k))
        for j in range(cur_k):
            for i in range(prev_k):
                score[i, j] = delta[i] + transitions(t, i, j)
        best = np.argmax(score, axis=0)
        delta = emissions[t] + score[best, np.arange(cur_k)]
        back.append(best)
    # backtrace
    path = [int(np.argmax(delta))]
    for t in range(T - 2, -1, -1):
        path.append(int(back[t][path[-1]]))
    path.reverse()
    return path


def decode_trip(candidates, emissions, gps_df, hmm):
    """Decode one trip, honouring gaps and missing candidates.

    candidates : list over ALL points of candidate lists (may be empty)
    emissions  : matching list of emission log-prob arrays
    gps_df     : preprocessed trajectory (needs dist_prev_m and segment_id)
    hmm        : RouteHMM instance (provides transition_log_prob)

    Returns a list over ALL points: the chosen arc id, or None for points
    inside a segment that contains no candidates at all.
    """
    predicted = [None] * len(candidates)
    segment_ids = gps_df["segment_id"].values
    dist_prev = gps_df["dist_prev_m"].values

    # process one segment at a time -> never stitch across GPS gaps
    seg_order = []
    seg_points = {}
    for idx, sid in enumerate(segment_ids):
        if sid not in seg_points:
            seg_points[sid] = []
            seg_order.append(sid)
        seg_points[sid].append(idx)

    for sid in seg_order:
        pts = seg_points[sid]
        # keep only points that actually have candidates
        kept = [i for i in pts if len(candidates[i]) > 0]
        if not kept:
            continue  # whole segment unmatched: predictions stay None

        kept_emissions = [emissions[i] for i in kept]
        kept_candidates = [candidates[i] for i in kept]

        # travelled GPS distance between consecutive kept points: sum the
        # per-step distances across any skipped (candidate-less) points
        gap_dist = {}
        for n in range(1, len(kept)):
            d = 0.0
            for i in range(kept[n - 1] + 1, kept[n] + 1):
                step = dist_prev[i]
                if not np.isnan(step):
                    d += step
            gap_dist[n] = d

        def transitions(t, i, j):
            return hmm.transition_log_prob(kept_candidates[t - 1][i],
                                           kept_candidates[t][j],
                                           gap_dist[t])

        path = viterbi_decode(kept_emissions, transitions)

        # map back to point indices; fill candidate-less points with the
        # previous decoded arc within the segment
        decoded_arc = {kept[n]: kept_candidates[n][path[n]]["arc_id"]
                       for n in range(len(kept))}
        last_arc = None
        kept_set = set(kept)
        first_kept = kept[0]
        for i in pts:
            if i in kept_set:
                last_arc = decoded_arc[i]
                predicted[i] = last_arc
            elif i < first_kept:
                predicted[i] = decoded_arc[first_kept]  # leading run
            else:
                predicted[i] = last_arc  # inherit previous decoded arc
    return predicted
