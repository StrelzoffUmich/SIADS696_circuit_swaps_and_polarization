#!/usr/bin/env python3
"""NON-CANONICAL side test -- raw (unscaled) vs log1p MQT-9, IN- and OUT-of-distribution.

Reintroduces the unscaled MQT-9 column Table 5 drops, and shows WHY it's dropped: pre/post the log
transform, in-distribution (left) vs cross-device OOD (right). In-distribution the transform is a
modest lift and even raw counts are serviceable; OUT of distribution raw counts DETONATE (route
R²≈-75) because a linear model predicting a log target extrapolates explosively on out-of-range
circuits, while log1p tames the tail. OOD = mean over the 4 non-Brisbane devices (Berlin/Boston/
Sherbrooke/Torino). Reads the already-computed mqt9_log_lasso.csv; no recompute.
"""
from __future__ import annotations
import sys, pathlib
HERE = pathlib.Path(__file__).resolve(); SUP = HERE.parent.parent; REPO = SUP.parent.parent
sys.path.insert(0, str(SUP))
import pandas as pd, numpy as np                  # noqa: E402
import matplotlib                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402

OUT = REPO / "data" / "results"; FIG = OUT / "figures" / "exploratory"   # non-report exploratory figure
FIG.mkdir(parents=True, exist_ok=True)
CSV = OUT / "mqt9_log_lasso.csv"            # data CSV lives in results/ (written by report_artifacts)
TASKS = ["route", "pol", "pol_z"]
OOD = ["Berlin", "Boston", "Sherbrooke", "Torino"]
REPS = [("mqt9_raw", "MQT-9 raw (pre-log)", "#C44E52"),
        ("mqt9_log", "MQT-9 log1p (post-log)", "#DD8452"),
        ("spectral", "spectral-17 (ref)", "#4C72B0")]


def main():
    d = pd.read_csv(CSV)
    indist = {(t, r): float(d[(d.scheme == "indist") & (d.task == t) & (d.rep == r)].R2.values[0])
              for t in TASKS for r, *_ in REPS}
    ood = {(t, r): float(d[(d.scheme.isin(OOD)) & (d.task == t) & (d.rep == r)].R2.mean())
           for t in TASKS for r, *_ in REPS}

    xs = np.arange(len(TASKS)); nb = len(REPS); bw = 0.8 / nb
    fig, (axI, axO) = plt.subplots(1, 2, figsize=(13.4, 5.2))

    def grouped(ax, data, ylo, yhi, annotate_clipped):
        for k, (rep, lab, col) in enumerate(REPS):
            vals = np.array([data[(t, rep)] for t in TASKS])
            off = (k - (nb - 1) / 2) * bw
            vc = np.clip(vals, ylo, yhi)
            ax.bar(xs + off, vc, bw, color=col, label=lab, edgecolor="white", linewidth=.4)
            for x, v, c in zip(xs + off, vals, vc):
                if annotate_clipped and v < ylo:                 # off-scale: print true value at floor
                    ax.text(x, ylo + 0.06 * (yhi - ylo), f"{v:.0f}", rotation=90, ha="center",
                            va="bottom", fontsize=7.5, color="white", fontweight="bold")
                else:
                    ax.text(x, v + 0.012 * (yhi - ylo), f"{v:.2f}", ha="center", fontsize=7.8)
        ax.axhline(0, color="#888", lw=.8)
        ax.set_xticks(xs); ax.set_xticklabels(TASKS, fontsize=11)
        ax.set_ylim(ylo, yhi); ax.grid(alpha=.25, axis="y")

    grouped(axI, indist, 0, 1.0, annotate_clipped=False)
    axI.set_ylabel("5-fold CV R²"); axI.set_title("In-distribution", fontsize=12.5, fontweight="bold")
    axI.legend(fontsize=8.5, loc="lower right")

    grouped(axO, ood, -6, 1.0, annotate_clipped=True)
    axO.set_ylabel("OOD R²  (mean of 4 cross-devices)")
    axO.set_title("Out-of-distribution (cross-device)", fontsize=12.5, fontweight="bold")

    fig.suptitle("MQT-9: unscaled vs log1p", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "mqt9_raw_vs_log.png", dpi=140); plt.close(fig)
    print("wrote", FIG / "mqt9_raw_vs_log.png")
    print("DONE_MQT9")


if __name__ == "__main__":
    main()
