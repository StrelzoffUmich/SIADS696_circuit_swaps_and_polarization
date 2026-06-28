#!/usr/bin/env python3
"""viz_tests/feature_correlation_delta.py — Δ-map of spectral-17 feature correlations,
OOD minus in-distribution (ρ_validation − ρ_train).

Captures BOTH correlation maps in one figure: how the feature-redundancy structure SHIFTS out of
distribution, and whether time_to_connected keeps its in-distribution orthogonality. Cells are
ρ_OOD − ρ_in: red = correlation strengthens OOD, blue = weakens, white = stable. Features are
pre-routing structural (device-independent), so this is a circuit-family-distribution shift
(training families → validation families), not a device effect.

Appendix / exploratory (NOT a report-body figure).
Run:  python viz_tests/feature_correlation_delta.py
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
    train = R.load_data(R.DEFAULT_SWAP_CSV, R.DEFAULT_POL_CSV)
    ood = R.load_external(OOD_CSV)
    feats = [c for c in R.FEATURE_SETS["spectral"] if c in train.columns and c in ood.columns]
    Cin = train[feats].corr(method="spearman")
    Cood = ood[feats].corr(method="spearman")
    # order by the IN-distribution redundancy clustering, so the map is comparable to the in-dist heatmap
    D = 1.0 - np.abs(Cin.values)
    np.fill_diagonal(D, 0.0)
    idx = list(leaves_list(linkage(squareform(D, checks=False), method="average")))
    lab = [feats[i] for i in idx]
    Cin = Cin.iloc[idx, idx]
    Cood = Cood.iloc[idx, idx]
    delta = Cood.values - Cin.values
    pd.DataFrame(delta, index=lab, columns=lab).to_csv(
        os.path.join(R.RESULTS, "delta_correlation_spearman.csv"))

    n = len(lab)
    mask = np.triu(np.ones_like(delta, dtype=bool), k=1)
    Dm = np.ma.masked_where(mask, delta)
    vmax = float(np.nanmax(np.abs(delta[~mask]))) or 0.1
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(11.6, 10.4))
    im = ax.imshow(Dm, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
    tl = [("» " + l if l == "time_to_connected" else l) for l in lab]
    ax.set_xticks(range(n)); ax.set_xticklabels(tl, rotation=45, ha="right", fontsize=9.5)
    ax.set_yticks(range(n)); ax.set_yticklabels(tl, fontsize=9.5)
    for r in range(n):
        for c in range(r + 1):
            v = delta[r, c]
            if abs(v) >= 0.2:
                ax.text(c, r, f"{v:+.2f}", ha="center", va="center", fontsize=7.0,
                        color="white" if abs(v) > 0.6 * vmax else "#222")
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.4)
    ax.tick_params(which="minor", length=0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ Spearman ρ  (OOD − in-distribution)")
    ax.set_title("Δ feature correlation (OOD − in-distribution), spectral-17",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    p = os.path.join(R.OUT_EXPLORATORY, "delta_correlation_spearman.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    ti = lab.index("time_to_connected")
    tccd = np.abs(np.delete(delta[ti], ti))
    allod = np.abs(delta[~np.eye(n, dtype=bool)])
    print(f"TCC |Δρ| mean={tccd.mean():.3f} max={tccd.max():.3f}  |  all-pairs |Δρ| mean={allod.mean():.3f}")
    print("wrote", p)


if __name__ == "__main__":
    main()
