#!/usr/bin/env python3
"""NON-CANONICAL side test — Spearman rank-correlation heatmap among the spectral-17 features
(training set). Demonstrates the intra-feature collinearity that underlies the importance story:
mutually redundant descriptors (|ρ|→1 blocks) are why several large coefficients carry little
unique ablation, and why individual L1 weights are sign-unstable. Diagnostic only; imports the
harness for the feature list + data loader, computes nothing model-related.

Features are reordered by hierarchical clustering on 1−|ρ| so redundant groups form visible blocks.
"""
from __future__ import annotations
import sys, pathlib
HERE = pathlib.Path(__file__).resolve()
SUP = HERE.parent.parent
REPO = SUP.parent.parent
sys.path.insert(0, str(SUP))
import supervised_analysis_run as H               # noqa: E402
import numpy as np, pandas as pd                  # noqa: E402
import matplotlib                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402
from scipy.cluster.hierarchy import linkage, leaves_list   # noqa: E402
from scipy.spatial.distance import squareform     # noqa: E402

OUT = REPO / "data" / "results"
FIG = OUT / "figures" / "exploratory"       # non-report exploratory figure
FIG.mkdir(parents=True, exist_ok=True)


def main():
    train = H.load_data(H.DEFAULT_SWAP_CSV, H.DEFAULT_POL_CSV)
    feats = [c for c in H.FEATURE_SETS["spectral"] if c in train.columns]
    C = train[feats].corr(method="spearman")
    # order by clustering on 1-|rho| so mutually-redundant features sit together
    D = 1.0 - np.abs(C.values); np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    idx = list(leaves_list(Z))
    Cc = C.iloc[idx, idx]
    lab = list(Cc.columns)
    Cc.to_csv(OUT / "feature_correlation_spearman.csv")

    M = Cc.values
    n = len(lab)
    mask = np.triu(np.ones_like(M, dtype=bool), k=1)          # hide the mirror-image upper triangle
    Mm = np.ma.masked_where(mask, M)
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(11.6, 10.4))
    im = ax.imshow(Mm, cmap=cmap, vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n)); ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=9.5)
    ax.set_yticks(range(n)); ax.set_yticklabels(lab, fontsize=9.5)
    for r in range(n):
        for c in range(r + 1):                               # lower triangle + diagonal only
            v = M[r, c]
            if abs(v) >= 0.3:                                # label only the meaningful correlations
                ax.text(c, r, f"{v:+.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(v) > 0.6 else "#222")
    ax.set_xticks(np.arange(-.5, n, 1), minor=True); ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.4); ax.tick_params(which="minor", length=0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman ρ")
    ax.set_title("Spearman rank-correlation between spectral-17 features (training set)",
                 fontweight="bold", fontsize=13)
    fig.text(0.5, 0.012,
             "Lower triangle only;  ρ labelled where |ρ|≥0.3.  Features ordered by hierarchical clustering "
             "on 1−|ρ| — dark off-diagonal blocks = mutually redundant descriptors.",
             ha="center", va="bottom", fontsize=9.5, style="italic", color="#444")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    p = FIG / "feature_correlation_spearman.png"
    fig.savefig(p, dpi=150); plt.close(fig)

    # report the strongest redundant pairs (|rho| highest off-diagonal)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((abs(C.iloc[i, j]), C.iloc[i, j], C.columns[i], C.columns[j]))
    pairs.sort(reverse=True)
    print("strongest |ρ| feature pairs (top 10):")
    for a, rho, f1, f2 in pairs[:10]:
        print(f"  ρ={rho:+.2f}  {f1} <-> {f2}")
    print(f"figure -> {p}")
    print("DONE_CORR")


if __name__ == "__main__":
    main()
