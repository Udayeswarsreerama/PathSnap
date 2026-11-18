"""Evaluate the unchanged nearest-reference-arc metric on every trip.

Run from the project folder:
    python evaluate_accuracy_patch.py --data-dir data/student_test

Ground truth is used only for evaluation, never candidate scores or decoding.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "pipeline"), str(ROOT / "member3")]
from data_loader import load_trip
from hmm import HMMConfig
from candidate_scorer import build_scorer
from route_reconstruction import reconstruct_trip
from evaluate_decoder import evaluate_trip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/student_test")
    parser.add_argument("--output", default="accuracy_patch_results.csv")
    args = parser.parse_args()
    config_path = ROOT / "member3" / "hmm_config.json"
    cfg = HMMConfig.load(config_path)
    artifact = Path(cfg.scorer["artifact_path"])
    if not artifact.is_absolute():
        cfg.scorer["artifact_path"] = str(config_path.parent / artifact)
    scorer = build_scorer(cfg)
    rows = []
    for folder in sorted(Path(args.data_dir).iterdir()):
        if not folder.is_dir() or not list(folder.glob("*.track")):
            continue
        p = reconstruct_trip(load_trip(str(folder)), scorer, cfg)
        m = evaluate_trip(p, cfg)
        row = {"trip_id": p["trip_id"], "n_points": len(p["gps_df"]),
               "classifier_accuracy": m.get("clf_top1_point_acc"),
               "hmm_accuracy": m.get("viterbi_point_acc"),
               "candidate_recall": 1 - m.get("candidate_miss_rate", 1),
               "route_continuity": m.get("viterbi_continuity")}
        rows.append(row)
        print(json.dumps(row), flush=True)
    scored = [r for r in rows if r["hmm_accuracy"] is not None]
    if not scored:
        raise SystemExit("No scored trips. Check data directory and .route files.")
    n = sum(r["n_points"] for r in scored)
    print(json.dumps({"pooled_point_accuracy": sum(r["hmm_accuracy"] * r["n_points"] for r in scored) / n,
                      "mean_trip_accuracy": sum(r["hmm_accuracy"] for r in scored) / len(scored),
                      "n_points": n, "n_trips": len(scored)}, indent=2))
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
