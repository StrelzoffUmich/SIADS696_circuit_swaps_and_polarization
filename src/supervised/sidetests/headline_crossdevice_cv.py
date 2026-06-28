#!/usr/bin/env python3
"""NON-CANONICAL side test -- HEADLINE cross-device family comparison, CROSS-VALIDATED, as a
device-centric faceted line plot (route | pol | pol_z) -- the reorientation of the by-category Table 4.

Protocol (5-fold GroupKFold on the TRAINING set; the test device is never tuned on -> no leakage):
  * in-distribution (Brisbane): standard 5-fold CV -- fit on 4/5, score the held-out 1/5.
  * each cross-device point: fit on the same 4/5 training fold, predict that device's FULL validation
    corpus. 5 folds -> 5 R² estimates -> mean ± sd.
For pol_z the within-N z-stats are fit on the TRAINING FOLD and applied to the held-out fold and to
every OOD device (no normalization leakage). LASSO/ridge tune internally per fold. So every point
(family × device) is a 5-fold mean ± sd -- the "no single split; report mean + sd over CV folds"
requirement, for the cross-device headline.

Each target gets its own panel/y-axis: route is device-invariant (flat ≈0.79 OOD); pol/pol_z LEVEL
collapses on the clean far devices (Berlin/Boston) -- the noise-scale shift. Imports the harness only.
"""
from __future__ import annotations
import sys, pathlib, warnings
HERE = pathlib.Path(__file__).resolve(); SUP = HERE.parent.parent; REPO = SUP.parent.parent
sys.path.insert(0, str(SUP))
import supervised_analysis_run as H               # noqa: E402
import numpy as np, pandas as pd                  # noqa: E402
import matplotlib                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402
from matplotlib import cm                         # noqa: E402
from sklearn.model_selection import GroupKFold    # noqa: E402

warnings.filterwarnings("ignore")
LAB = REPO / "data" / "xdev_out" / "labeled"
OUT = REPO / "data" / "results"; FIG = OUT / "figures" / "exploratory"   # non-report exploratory figure
FIG.mkdir(parents=True, exist_ok=True)
OOD = ["FakeSherbrooke", "FakeTorino", "FakeBerlin", "FakeBoston"]
FAMILIES = ["lasso", "ridge", "rf", "histgb", "knn"]
XKEYS = ["Brisbane", "Sherbrooke", "Torino", "Berlin", "Boston"]
XLABS = ["Brisbane\n(in-dist)", "Sherbrooke", "Torino", "Berlin", "Boston"]
TARGETS = [("route", "route", False, "swap"), ("pol", "pol", False, "pol"),
           ("pol_z", "pol", True, "pol")]


def eval_target(train, base, rel, arm, folds):
    hattr = "has_route" if arm == "swap" else "has_pol"
    oods = {}
    for d in OOD:
        p = LAB / f"val_{arm}_{d}.csv"
        if p.exists():
            o = H.load_external(str(p))
            if o.attrs.get(hattr):
                oods[d] = o
    feats = H.shared_feats("spectral", train, *oods.values())
    X = train[feats].fillna(0).values; grp = train.grp.values
    yb = train[base].values.astype(float); N = train.N.values
    Xod = {d: o[feats].fillna(0).values for d, o in oods.items()}
    ybod = {d: o[base].values.astype(float) for d, o in oods.items()}
    Nod = {d: o.N.values for d, o in oods.items()}

    res = {f: {} for f in FAMILIES}
    for fam in FAMILIES:
        per = {k: [] for k in XKEYS}
        for tr, te in folds:
            if rel:
                stats, gm, gs = H.fit_zstats(yb[tr], N[tr])
                ytr = H.apply_zstats(yb[tr], N[tr], stats, gm, gs)
                yte = H.apply_zstats(yb[te], N[te], stats, gm, gs)
            else:
                ytr, yte = yb[tr], yb[te]
            pipe = H._fit_full(fam, X[tr], ytr, grp[tr])
            per["Brisbane"].append(H.metrics_full(yte, pipe.predict(X[te]))["R2"])
            for d, o in oods.items():
                yo = (H.apply_zstats(ybod[d], Nod[d], stats, gm, gs) if rel else ybod[d])
                per[d.replace("Fake", "")].append(H.metrics_full(yo, pipe.predict(Xod[d]))["R2"])
        for k, v in per.items():
            res[fam][k] = (float(np.mean(v)) if v else np.nan, float(np.std(v)) if v else np.nan)
    return res


def main():
    train = H.load_data(H.DEFAULT_SWAP_CSV, H.DEFAULT_POL_CSV)
    folds = list(GroupKFold(5).split(np.arange(len(train)), groups=train.grp.values))
    all_res, rows = {}, []
    for tok, base, rel, arm in TARGETS:
        print(f"== {tok} ==")
        res = eval_target(train, base, rel, arm, folds)
        all_res[tok] = res
        for fam in FAMILIES:
            print(f"  {fam:7s} " + "  ".join(f"{k}={res[fam][k][0]:+.3f}±{res[fam][k][1]:.3f}" for k in XKEYS))
            for k in XKEYS:
                m, s = res[fam][k]
                rows.append(dict(target=tok, family=fam, device=k,
                                 scheme=("in-dist" if k == "Brisbane" else "cross-device"),
                                 R2_mean=m, R2_sd=s))
    pd.DataFrame(rows).to_csv(OUT / "headline_crossdevice_cv.csv", index=False)
    render(all_res)
    print(f"figure -> {FIG / 'headline_crossdevice_cv.png'}")
    print("DONE_HEADLINE")


def load_csv():
    """Rebuild the all_res structure from the saved CSV (re-render without recomputing)."""
    d = pd.read_csv(OUT / "headline_crossdevice_cv.csv")
    all_res = {}
    for tok, *_ in TARGETS:
        res = {f: {} for f in FAMILIES}
        for f in FAMILIES:
            for k in XKEYS:
                r = d[(d.target == tok) & (d.family == f) & (d.device == k)].iloc[0]
                res[f][k] = (float(r.R2_mean), float(r.R2_sd))
        all_res[tok] = res
    return all_res


def render(all_res):
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.4))               # <2000px @140dpi so it inlines
    xs = np.arange(len(XKEYS))
    palette = [cm.tab10(i) for i in (0, 1, 2, 3, 4)]
    YLO, YHI = -2.0, 1.0                                          # fixed shared scale across panels
    nf = len(FAMILIES); bw = 0.8 / nf
    for ax, (tok, *_ ) in zip(axes, TARGETS):
        res = all_res[tok]
        ax.axvspan(-0.5, 0.5, color="#eef2f7", zorder=0)          # in-dist column (Brisbane)
        for i, fam in enumerate(FAMILIES):
            m = np.array([res[fam][k][0] for k in XKEYS])
            s = np.array([res[fam][k][1] for k in XKEYS])
            off = (i - (nf - 1) / 2) * bw
            mc = np.clip(m, YLO, YHI)                             # clip bars to the shared axis
            ax.bar(xs + off, mc - YLO, bw, bottom=YLO, color=palette[i],
                   label=fam, zorder=3, edgecolor="white", linewidth=0.3)
            ax.errorbar(xs + off, mc, yerr=np.where((m > YLO) & (m < YHI), s, 0),
                        fmt="none", ecolor="#333", elinewidth=.7, capsize=1.5, zorder=4)
            for xi, mv in zip(xs, m):                             # annotate off-scale (clipped) bars
                if mv < YLO:
                    ax.text(xi + off, YLO + 0.07, f"▼{mv:.1f}", rotation=90, ha="center",
                            va="bottom", fontsize=6.2, color="#222", fontweight="bold")
        ax.axvline(0.5, ls="--", color="#888", lw=1)
        ax.axhline(0, color="#888", lw=.8, zorder=2)
        ax.set_xticks(xs); ax.set_xticklabels(XLABS, fontsize=8.5)
        ax.set_ylim(YLO, YHI); ax.set_title(tok, fontweight="bold"); ax.grid(alpha=.25, axis="y")
    axes[0].set_ylabel("R²  (5-fold CV mean ± sd)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Cross-device generalization by model family — 5-fold CV",
                 fontweight="bold", fontsize=13, y=0.985)
    fig.legend(handles, labels, title="family", ncol=5, loc="upper center",
               bbox_to_anchor=(0.5, 0.945), fontsize=9, frameon=False)
    fig.text(0.5, 0.012,
             "In-dist = standard 5-fold CV (shaded Brisbane column); each cross-device point = fit on the "
             "training fold, predict the device (5 folds → mean ± sd).  Shared y-axis clipped at −2; bars "
             "below carry their value.",
             ha="center", va="bottom", fontsize=8, style="italic", color="#444", wrap=True)
    fig.tight_layout(rect=[0, 0.05, 1, 0.88])
    fig.savefig(FIG / "headline_crossdevice_cv.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    if "--rerender" in sys.argv:                                  # re-plot from CSV, no recompute
        render(load_csv())
        print("rerendered from CSV")
    else:
        main()
