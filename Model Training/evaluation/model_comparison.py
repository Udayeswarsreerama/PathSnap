"""Build the model comparison table figure from results CSVs.

Usage: python3 model_comparison.py
Reads results/model_comparison.csv + results/cv_results.csv,
writes results/fig_model_comparison.png and results/fig_cv_top1.png.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")

fin = pd.read_csv(os.path.join(RES, "model_comparison.csv"))
cv = pd.read_csv(os.path.join(RES, "cv_results.csv"))
fin = fin.sort_values("top1_accuracy", ascending=False)

# ---- figure 1: final student-test metrics, grouped bars ----
metrics = ["top1_accuracy", "top1_accuracy_winnable", "f1_macro",
           "precision", "recall"]
labels = ["Top-1 acc", "Top-1 acc (winnable)", "Macro F1",
          "Precision", "Recall"]
x = range(len(fin))
w = 0.16
fig, ax = plt.subplots(figsize=(11, 5.5))
for i, (m, lab) in enumerate(zip(metrics, labels)):
    ax.bar([xi + (i - 2) * w for xi in x], fin[m], width=w, label=lab)
for xi, (_, r) in zip(x, fin.iterrows()):
    ax.text(xi, 0.02, f"logloss {r['log_loss']:.3f}", rotation=90,
            ha="center", va="bottom", fontsize=8, color="white")
ax.set_xticks(list(x))
ax.set_xticklabels(fin["model"])
ax.set_ylim(0, 1.05)
ax.set_ylabel("score")
ax.set_title("PathSnap - 5-model comparison on the 10 student_test trips "
             "(fitted on all 80 train trips)")
ax.legend(ncol=5, fontsize=8, loc="upper center",
          bbox_to_anchor=(0.5, -0.08))
fig.tight_layout()
fig.savefig(os.path.join(RES, "fig_model_comparison.png"), dpi=150)
plt.close(fig)

# ---- figure 2: CV vs test top-1 (generalisation check) ----
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.errorbar(fin["model"], fin["cv_top1_mean"], yerr=fin["cv_top1_std"],
            fmt="o-", capsize=4, label="5-fold group CV (train trips)")
ax.plot(fin["model"], fin["top1_accuracy"], "s--",
        label="student_test (10 trips)")
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("top-1 accuracy")
ax.set_title("Cross-validation vs held-out test - top-1 accuracy by model")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(RES, "fig_cv_vs_test.png"), dpi=150)
plt.close(fig)

# ---- console table for the report ----
cols = ["model", "cv_top1_mean", "top1_accuracy", "top1_accuracy_winnable",
        "precision", "recall", "f1_macro", "log_loss",
        "train_time_s", "infer_time_s"]
print(fin[cols].round(4).to_string(index=False))
