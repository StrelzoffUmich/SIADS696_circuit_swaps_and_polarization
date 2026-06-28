#!/usr/bin/env python3
"""viz_tests/feature_correlation_ood.py — OOD counterpart of report_artifacts.fig_feature_correlation.

Spearman rank-correlation heatmap among the spectral-17 features, computed on the OUT-of-distribution
validation corpus (the held-out algorithm families) instead of the training corpus. Pairs with the
in-distribution heatmap to show whether the feature-redundancy structure is corpus-stable. The
features are pre-routing structural (device-independent), so "OOD" here means a different circuit-
family distribution, not a different device.

Appendix / exploratory (NOT a report-body figure).
Run:  python viz_tests/feature_correlation_ood.py
"""
from __future__ import annotations
import sys
import pathlib
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))   # src/supervised
import report_artifacts as R                                              # noqa: E402

OOD_CSV = str(R.LABELED / "val_swap_FakeBrisbane.csv")   # validation families; features are device-independent


def main():
    ood = R.load_external(OOD_CSV)
    feats = [c for c in R.FEATURE_SETS["spectral"] if c in ood.columns]
    C = ood[feats].corr(method="spearman")
    D = 1.0 - np.abs(C.values)
    np.fill_diagonal(D, 0.0)
    idx = list(leaves_list(linkage(squareform(D, checks=False), method="average")))
    Cc = C.iloc[idx, idx]
    lab = list(Cc.columns)
    Cc.to_csv(os.path.join(R.RESULTS, "feature_correlation_ood_spearman.csv"))

    M = Cc.values
    n = len(lab)
    mask = np.triu(np.ones_like(M, dtype=bool), k=1)
    Mm = np.ma.masked_where(mask, M)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(11.6, 10.4))
    im = ax.imshow(Mm, cmap=cmap, vmin=-1, vmax=1, aspect="equal")
    tl = [("» " + l if l == "time_to_connected" else l) for l in lab]
    ax.set_xticks(range(n)); ax.set_xticklabels(tl, rotation=45, ha="right", fontsize=9.5)
    ax.set_yticks(range(n)); ax.set_yticklabels(tl, fontsize=9.5)
    for r in range(n):
        for c in range(r + 1):
            v = M[r, c]
            if abs(v) >= 0.3:
                ax.text(c, r, f"{v:+.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(v) > 0.6 else "#222")
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.4)
    ax.tick_params(which="minor", length=0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman ρ")
    ax.set_title("Spearman feature correlation — OUT-of-distribution (validation families), spectral-17",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    p = os.path.join(R.OUT_EXPLORATORY, "feature_correlation_ood.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
