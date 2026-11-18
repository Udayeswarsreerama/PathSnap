"""PathSnap Member 3 - candidate scoring adapter.

The HMM's emission model needs a probability per (GPS point, candidate arc)
from the team's classification model. Member 2 trains that model; this
module is the single adapter point where it plugs in.

Two scorers are provided:

  BaselineLogisticScorer
      A simple logistic regression trained on Member 1's feature export,
      used ONLY until Member 2's model artifact is available, so the HMM
      pipeline can be developed and evaluated end-to-end. Features:
      distance_m, rank, heading_diff_bidir_deg, speed_mps, arc_length_m,
      connected_to_prev. Class weighting handles the ~22% positive rate.
      This is NOT the team's final classifier - Member 2's comparison of
      5+ models decides that.

  ArtifactScorer
      Loads any joblib/pickle model Member 2 exports, as long as it
      implements predict_proba(X) (or decision_function(X), which is then
      squashed with a sigmoid). The feature column list is stored next to
      the artifact so training and inference features always match.

Both produce RAW per-candidate scores; this module then normalises them
per GPS point (softmax over that point's candidates) so the emission term
is a proper probability distribution over the candidate set.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

# feature columns used by the baseline scorer (subset of Member 1's schema)
BASELINE_FEATURES = ["distance_m", "rank", "heading_diff_bidir_deg",
                     "speed_mps", "arc_length_m", "connected_to_prev"]


def _prepare_X(df, feature_cols):
    """Feature matrix with light, explainable imputation.

    heading_diff_bidir_deg is NaN for the first point and for stationary
    intervals (no usable GPS heading) -> impute 90 degrees (uninformative
    middle of the 0-180 range). speed_mps NaN -> 0. Booleans -> 0/1.
    """
    X = pd.DataFrame(index=df.index)
    for col in feature_cols:
        v = df[col]
        if v.dtype == bool:
            v = v.astype(float)
        X[col] = v.astype(float)
    if "heading_diff_bidir_deg" in X:
        X["heading_diff_bidir_deg"] = X["heading_diff_bidir_deg"].fillna(90.0)
    if "speed_mps" in X:
        X["speed_mps"] = X["speed_mps"].fillna(0.0)
    X = X.fillna(0.0)
    return X.values


class BaselineLogisticScorer:
    """Stand-in classifier: logistic regression on Member 1's features."""

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(class_weight="balanced",
                                        max_iter=1000, C=1.0)
        self.feature_cols = list(BASELINE_FEATURES)
        self._scaler_mean = None
        self._scaler_std = None

    def fit(self, features_df):
        df = features_df[features_df["is_true"].notna()].copy()
        X = _prepare_X(df, self.feature_cols)
        y = df["is_true"].astype(int).values
        self._scaler_mean = X.mean(axis=0)
        self._scaler_std = X.std(axis=0) + 1e-9
        self.model.fit((X - self._scaler_mean) / self._scaler_std, y)
        return self

    def score_rows(self, features_df):
        X = _prepare_X(features_df, self.feature_cols)
        X = (X - self._scaler_mean) / self._scaler_std
        return self.model.predict_proba(X)[:, 1]


class _NumpyPipelineScorer:
    """Version-proof scorer for Member 2's impute -> scale -> MLP pipeline.

    Reimplements predict_proba with plain numpy, so the exported model works
    with ANY installed scikit-learn version (the joblib artifact only needs
    to unpickle; no sklearn internals are called at scoring time).
    """

    def __init__(self, pipe):
        steps = dict(pipe.steps)
        imp, scl, clf = steps["impute"], steps["scale"], steps["clf"]
        self.stat = np.asarray(imp.statistics_, dtype=np.float64)
        self.mean = np.asarray(scl.mean_, dtype=np.float64)
        self.scale = np.asarray(scl.scale_, dtype=np.float64)
        self.coefs = [np.asarray(c, dtype=np.float64) for c in clf.coefs_]
        self.intercepts = [np.asarray(b, dtype=np.float64) for b in clf.intercepts_]
        self.activation = getattr(clf, "activation", "relu")

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64).copy()
        mask = np.isnan(X)
        if mask.any():
            X[mask] = np.broadcast_to(self.stat, X.shape)[mask]
        a = (X - self.mean) / self.scale
        for W, b in zip(self.coefs[:-1], self.intercepts[:-1]):
            z = a @ W + b
            if self.activation == "relu":
                a = np.maximum(z, 0.0)
            elif self.activation == "tanh":
                a = np.tanh(z)
            elif self.activation == "logistic":
                a = 1.0 / (1.0 + np.exp(-z))
            else:
                a = z
        z = (a @ self.coefs[-1] + self.intercepts[-1]).ravel()
        p = 1.0 / (1.0 + np.exp(-z))  # MLPClassifier logistic output
        return np.column_stack([1.0 - p, p])


class ArtifactScorer:
    """Adapter for Member 2's exported model artifact.

    Expects a joblib file containing the fitted model and a JSON sidecar
    ("<artifact>.features.json") listing the training feature columns.
    """

    def __init__(self, artifact_path):
        self.model = joblib.load(artifact_path)
        if hasattr(self.model, "steps") and {"impute", "scale", "clf"} <= set(dict(self.model.steps)):
            # Member 2's pipeline: use the version-proof numpy scorer
            self.model = _NumpyPipelineScorer(self.model)
        sidecar = artifact_path + ".features.json"
        if os.path.exists(sidecar):
            with open(sidecar) as f:
                self.feature_cols = json.load(f)["feature_cols"]
        else:
            self.feature_cols = list(BASELINE_FEATURES)

    def score_rows(self, features_df):
        # bool -> float only; NaNs stay for the model's own imputer
        Xdf = pd.DataFrame(index=features_df.index)
        for col in self.feature_cols:
            v = features_df[col]
            if v.dtype == bool:
                v = v.astype(float)
            Xdf[col] = v.astype(float)
        X = Xdf.values
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        # fall back to a squashed decision function
        z = self.model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))


def build_scorer(config, train_features_df=None):
    """Factory: pick the scorer named in the HMM config and (if needed) fit it."""
    spec = config.scorer or {"type": "baseline_logistic"}
    if spec.get("type") == "artifact":
        return ArtifactScorer(spec["artifact_path"])
    scorer = BaselineLogisticScorer()
    if train_features_df is None or len(train_features_df) == 0:
        raise ValueError("baseline_logistic scorer needs training features")
    return scorer.fit(train_features_df)


def point_normalised_probs(features_df, raw_scores):
    """Softmax-normalise raw per-candidate scores within each GPS point.

    Returns a list over points of np arrays (one entry per candidate, in the
    order the candidates appear in features_df for that point).
    """
    df = features_df[["point_idx"]].copy()
    df["score"] = np.asarray(raw_scores, dtype=np.float64)
    out = []
    for _, group in df.groupby("point_idx", sort=True):
        s = group["score"].values
        s = np.clip(s, 1e-9, None)
        out.append(s / s.sum())
    return out
