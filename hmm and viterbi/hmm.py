"""PathSnap Member 3 - HMM definition for route reconstruction.

The Hidden Markov Model used for map-matching:

  Hidden state at time t  : one of the candidate road arcs generated for GPS
                            point t (candidate set from Member 1's pipeline).
                            The candidate -> road-segment correspondence is
                            preserved: each state IS an arc id.
  Observation at time t   : the raw GPS point (lon, lat) plus the movement
                            context (heading, speed, distance travelled).

  Emission probability    : how likely a candidate arc is, given the
                            observation. Two independent evidence terms are
                            combined in log space:
                              (1) spatial: Gaussian on the perpendicular
                                  point-to-arc distance, sigma from config
                                  (EDA measured GPS noise median ~4.2 m,
                                  p90 ~9.7 m, so sigma ~5-10 m is justified).
                              (2) classifier: Member 2's model probability
                                  for this (point, candidate) pair, obtained
                                  through the CandidateScorer adapter.
                            log P(obs | state) = w_dist * log N(d; 0, sigma)
                                               + w_clf  * log p_clf

  Transition probability  : how likely the vehicle moved from arc a (time t-1)
                            to arc b (time t). Based on road-network
                            connectivity, following the classic Newson-Krumm
                            idea but with our own implementation: we compute
                            the driving distance along the network between
                            the two candidate positions and compare it with
                            the distance the GPS trace actually travelled:
                              log P(a -> b) = -|network_dist - gps_dist| / beta
                            Unreachable pairs (no path within the search
                            cutoff) get a fixed large penalty instead of -inf,
                            so a prediction is always returned even when the
                            network is disconnected.

  GPS gaps                : preprocessing flags intervals > 10 s and assigns
                            segment ids. The decoder NEVER stitches a route
                            across a gap: each segment is decoded as its own
                            chain (see viterbi.py).

Only standard-library + numpy code here; the HMM math is implemented from
scratch so every team member can explain it.
"""
import heapq
import json
import math

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class HMMConfig:
    """All tunable HMM/decoder parameters in one place (see hmm_config.json)."""

    def __init__(self, sigma_m=7.5, transition_beta_m=25.0,
                 classifier_weight=1.0, distance_weight=1.0,
                 unreachable_log_penalty=-20.0, max_route_dist_m=2000.0,
                 route_dist_margin_m=50.0, route_dist_factor=3.0,
                 search_radius_m=30.0, top_k=5, gap_threshold_s=10.0,
                 excluded_trips=("00000033", "00000054"),
                 scorer=None, same_arc_backtrack_cost_m=50.0):
        # emission: Gaussian width on point-to-arc distance (metres)
        self.sigma_m = float(sigma_m)
        # emission: weight of the classifier probability term
        self.classifier_weight = float(classifier_weight)
        # emission: weight of the Gaussian distance term
        self.distance_weight = float(distance_weight)
        # transition: scale of the |network_dist - gps_dist| penalty (metres)
        self.transition_beta_m = float(transition_beta_m)
        # transition: log-prob given to arc pairs with no connecting path
        self.unreachable_log_penalty = float(unreachable_log_penalty)
        # transition search cutoff: never route further than this (metres)
        self.max_route_dist_m = float(max_route_dist_m)
        # cutoff for one transition: factor*gps_dist + margin, capped by max
        self.route_dist_factor = float(route_dist_factor)
        self.route_dist_margin_m = float(route_dist_margin_m)
        # candidate generation settings (passed to Member 1's pipeline)
        self.search_radius_m = float(search_radius_m)
        self.top_k = int(top_k)
        # points separated by more than this many seconds start a new segment
        self.gap_threshold_s = float(gap_threshold_s)
        # trips with known mismatched ground truth (found in Member 1's EDA)
        self.excluded_trips = tuple(excluded_trips)
        # scorer settings: {"type": "baseline_logistic"} or
        # {"type": "artifact", "artifact_path": ..., "feature_list": [...]}
        self.scorer = scorer or {"type": "baseline_logistic"}
        # Separate projection jitter from a real network U-turn.
        self.same_arc_backtrack_cost_m = float(same_arc_backtrack_cost_m)

    @staticmethod
    def load(path):
        with open(path) as f:
            return HMMConfig(**json.load(f))

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)


# ---------------------------------------------------------------------------
# Road network with bounded shortest-path queries
# ---------------------------------------------------------------------------


class RoadNetwork:
    """Directed road graph of one trip, with bounded Dijkstra queries.

    Arcs are directed (from_node -> to_node) exactly as given in the .arcs
    file. Lengths come from the trip's equirectangular projection, so all
    distances are in metres.
    """

    def __init__(self, nodes, arcs, projector):
        self.nodes = nodes
        self.arcs = arcs
        self.projector = projector
        ax, ay = projector.to_m(nodes[arcs[:, 0], 0], nodes[arcs[:, 0], 1])
        bx, by = projector.to_m(nodes[arcs[:, 1], 0], nodes[arcs[:, 1], 1])
        self.arc_lengths = np.hypot(bx - ax, by - ay)
        # adjacency: from_node -> list of (to_node, length)
        self.adj = {}
        for aid in range(len(arcs)):
            u, v = int(arcs[aid, 0]), int(arcs[aid, 1])
            self.adj.setdefault(u, []).append((v, float(self.arc_lengths[aid])))

    def arc_length(self, arc_id):
        return float(self.arc_lengths[arc_id])

    def directly_connected(self, arc_a, arc_b):
        """True if arc a's to-node is arc b's from-node (or same arc)."""
        return arc_a == arc_b or int(self.arcs[arc_a, 1]) == int(self.arcs[arc_b, 0])

    def shortest_path(self, from_node, to_node, cutoff_m):
        """Bounded Dijkstra: network distance from one node to another.

        Returns the distance in metres, or math.inf if to_node cannot be
        reached within cutoff_m. The bound keeps per-transition cost tiny.
        """
        if from_node == to_node:
            return 0.0
        dist = {from_node: 0.0}
        heap = [(0.0, from_node)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == to_node:
                return d
            if d > dist.get(u, math.inf):
                continue
            if d > cutoff_m:
                break
            for v, w in self.adj.get(u, ()):
                nd = d + w
                if nd < dist.get(v, math.inf) and nd <= cutoff_m:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return dist.get(to_node, math.inf)


# ---------------------------------------------------------------------------
# Emission and transition models
# ---------------------------------------------------------------------------


def gaussian_logpdf(distance_m, sigma_m):
    """Log of N(d; 0, sigma^2) up to the normalising constant.

    The constant is the same for every candidate of one GPS point, so it
    cancels inside Viterbi and is dropped.
    """
    return -(distance_m ** 2) / (2.0 * sigma_m ** 2)


class RouteHMM:
    """Emission + transition scoring for one trip, per the model above."""

    def __init__(self, config, network):
        self.cfg = config
        self.net = network

    # -- emissions ---------------------------------------------------------

    def emission_log_probs(self, candidates, classifier_probs):
        """log P(observation | state) for every candidate of every point.

        candidates       : list over points of lists of candidate dicts
                           (each has "distance_m")
        classifier_probs : same shape; probability from the scoring model
        Returns a list over points of np.float64 arrays (one entry per
        candidate). Points without candidates yield an empty array.
        """
        out = []
        for cands, probs in zip(candidates, classifier_probs):
            if not cands:
                out.append(np.empty(0))
                continue
            dist = np.array([c["distance_m"] for c in cands])
            spatial = gaussian_logpdf(dist, self.cfg.sigma_m)
            probs = np.asarray(probs, dtype=np.float64)
            # guard against a classifier outputting an exact 0
            probs = np.clip(probs, 1e-12, None)
            emit = (self.cfg.distance_weight * spatial
                    + self.cfg.classifier_weight * np.log(probs))
            out.append(emit)
        return out

    # -- transitions ---------------------------------------------------------

    def _network_distance(self, a, b, gps_dist_m):
        """Driving distance along the network between two candidate positions.

        Candidate positions are given as (arc, t_frac) where t_frac is the
        fraction along the arc (0 = from-node, 1 = to-node). The network
        distance is: rest of arc a + shortest path between the nodes +
        entry part of arc b. Same-arc transitions measure progress along
        the arc; going backwards on the same arc counts as a U-turn detour.
        """
        arc_a, ta = a["arc_id"], a["t_frac"]
        arc_b, tb = b["arc_id"], b["t_frac"]
        len_a = self.net.arc_length(arc_a)
        len_b = self.net.arc_length(arc_b)
        if arc_a == arc_b:
            if tb >= ta:
                return (tb - ta) * len_a
            # GPS projection can regress on the same arc without a U-turn.
            # Keep the historical cost as a configurable default; zero avoids
            # charging a fixed manoeuvre cost for every jittering observation.
            return (ta - tb) * len_a + self.cfg.same_arc_backtrack_cost_m
        cutoff = min(self.cfg.route_dist_factor * gps_dist_m
                     + self.cfg.route_dist_margin_m,
                     self.cfg.max_route_dist_m)
        sp = self.net.shortest_path(int(self.net.arcs[arc_a, 1]),
                                    int(self.net.arcs[arc_b, 0]), cutoff)
        if math.isinf(sp):
            return math.inf
        return (1.0 - ta) * len_a + sp + tb * len_b

    def transition_log_prob(self, cand_a, cand_b, gps_dist_m):
        """log P(arc a at t-1 -> arc b at t) given the travelled GPS distance."""
        net = self._network_distance(cand_a, cand_b, gps_dist_m)
        if math.isinf(net):
            # disconnected network: heavily penalised but not impossible,
            # so the decoder still returns a prediction for every point
            return self.cfg.unreachable_log_penalty
        return -abs(net - gps_dist_m) / self.cfg.transition_beta_m
