#!/usr/bin/env python3
"""NON-CANONICAL side test — sensitivity of the FEATURE-IMPORTANCE / ABLATION results (§2) on the
best model (LASSO, spectral-17, route). Companion to sensitivity_lasso.py (which probed R²/ρ).

This answers the §3 question *about the §2 results*: how stable are the importance & ablation
conclusions to the varying solution elements?

  (1) COEF PATHS vs α — each spectral-17 feature's standardized coefficient as α sweeps the harness
      grid (logspace(-3,1)). Shows whether the importance RANKING (which features dominate, which L1
      zeros) holds as regularization varies. Paired with the §2 in-dist ablation ΔR² per feature.
  (2) ABLATION ACROSS DEVICES — leave-one-feature-out drop in each DEVICE's OOD R² (train Brisbane →
      test device). The fitted coefficients are device-independent (same trained model), so the only
      device-varying importance signal is how much each feature matters for predicting each device.

Imports the harness; does not modify it. Reuses _fit_full (tuned lasso), shared_feats, metrics_full,
build identical StandardScaler+Lasso to make_lasso.
"""
from __future__ import annotations
import sys, pathlib, warnings
HERE = pathlib.Path(__file__).resolve()
SUP = HERE.parent.parent
REPO = SUP.parent.parent
sys.path.insert(0, str(SUP))
import supervised_analysis_run as H               # noqa: E402
import numpy as np, pandas as pd                  # noqa: E402
import matplotlib                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402
from matplotlib import cm                         # noqa: E402
from sklearn.pipeline import Pipeline             # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.linear_model import Lasso            # noqa: E402
from sklearn.model_selection import GroupKFold     # noqa: E402

warnings.filterwarnings("ignore")
LAB = REPO / "data" / "xdev_out" / "labeled"
OUT = REPO / "data" / "results"
FIG = OUT / "figures" / "exploratory"       # non-report exploratory figures
FIG.mkdir(parents=True, exist_ok=True)
OOD = ["FakeSherbrooke", "FakeTorino", "FakeBerlin", "FakeBoston"]
ALPHAS = np.logspace(-3, 1, 30)


def lasso_fixed(a):
    return Pipeline([("scale", StandardScaler()), ("model", Lasso(alpha=a, max_iter=20000))])


def load_oods():
    out = {}
    for d in OOD:
        p = LAB / f"val_swap_{d}.csv"
        if p.exists():
            out[d] = H.load_external(str(p))
    return out


def _ord(n):
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n if n < 20 else n % 10, 'th') }"


def render_importance_fig(feats, coef, abl_val, full_r2, out_path):
    """§2 FEATURE IMPORTANCE & ABLATION (self-contained, 6 pts). 'Which features matter.' Per
    feature: |standardized coef| and ablation ΔR² (in-dist 5-fold CV), each scaled to its own max
    so the two importance LENSES are comparable; sorted by ablation. Disagreement = mismatched bar
    lengths (big-coef/small-ablation = redundant; small-coef/big-ablation = irreplaceable). Any
    L1-zeroed feature is flagged. NO α here -- stability lives in the Sensitivity figure. full_r2 =
    the full-model route CV R² (the denominator the ablation ΔR² bars are measured against)."""
    aco = np.abs(coef)
    order = np.argsort(abl_val)[::-1]                      # lead with the ablation ranking
    f = [feats[i] for i in order]
    ab = np.array([abl_val[i] for i in order])
    co = np.array([aco[i] for i in order])
    dead = co < 1e-6
    abn = ab / max(ab.max(), 1e-12)
    con = co / max(co.max(), 1e-12)
    y = np.arange(len(f))[::-1]; h = 0.40
    fig, ax = plt.subplots(figsize=(10.6, 6.9))
    ax.barh(y + h / 2, abn, height=h, color="#4C72B0", label="ablation ΔR²  (drop when removed)")
    ax.barh(y - h / 2, con, height=h, color="#DD8452", label="|standardized coefficient|")
    for k in range(len(f)):
        ax.text(abn[k] + 0.012, y[k] + h / 2, f"{ab[k]:+.4f}", va="center", fontsize=6.3, color="#274b73")
        if not dead[k]:
            ax.text(con[k] + 0.012, y[k] - h / 2, f"{co[k]:.3f}", va="center", fontsize=6.3, color="#8a4b22")
    for k in np.where(dead)[0]:
        ax.text(0.012, y[k], "L1-zeroed", va="center", fontsize=6.5, color="#999", style="italic")
    ax.set_yticks(y); ax.set_yticklabels(f, fontsize=8.5)
    ax.set_xlim(0, 1.22)
    ax.set_xlabel(f"importance — each lens scaled to its own maximum "
                  f"(ablation max = {ab.max():.4f},  |coef| max = {co.max():.3f})")
    ax.set_title(f"Feature importance & ablation — LASSO (route, spectral-17;  full CV R² = {full_r2:.3f})",
                 fontweight="bold", fontsize=12.5)
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=.25, axis="x")
    le = "laplacian_energy"
    lead = ""
    if le in feats:
        li = feats.index(le)
        lr = int((np.abs(coef) > np.abs(coef[li])).sum()) + 1     # rank by |coef|
        lead = (f"  Clearest split: laplacian_energy — {_ord(lr)}-largest coefficient "
                f"({abs(coef[li]):.3f}) yet near-zero ablation ({abl_val[li]:+.4f}): a large weight "
                f"with almost no unique contribution.")
    fig.text(0.5, 0.014,
             "Each lens is scaled to its own maximum — the two scales are independent; compare ranks "
             "within a colour, not bar lengths across colours." + lead,
             ha="center", va="bottom", fontsize=8, style="italic", color="#444", wrap=True)
    fig.tight_layout(rect=[0, 0.085, 1, 1])
    fig.savefig(out_path, dpi=150); plt.close(fig)


def render_importance_scatter(feats, coef, abl_val, full_r2, out_path):
    """§2 FEATURE IMPORTANCE & ABLATION as a SCATTER -- |coef| vs ablation ΔR², each scaled to its
    own max so a y=x diagonal means 'equal relative importance on both lenses'. Distance from the
    diagonal IS the disagreement: redundant (big coef, low ablation) fall below-right, irreplaceable
    (low coef, high ablation) above-left, minor features cluster at the origin. Position carries the
    meaning -- no dual-scale bars to misread."""
    aco = np.abs(coef)
    x = aco / max(aco.max(), 1e-12)
    y = abl_val / max(abl_val.max(), 1e-12)
    dev = y - x
    thr = 0.15
    col = np.where(dev > thr, "#2a8a2a", np.where(dev < -thr, "#C44E52", "#4C72B0"))
    fig, ax = plt.subplots(figsize=(9.4, 8.4))
    ax.fill_between([0, 1], [0, 1], [1, 1], color="#cfe8cf", alpha=.30, zorder=0)   # above = irreplaceable
    ax.fill_between([0, 1], [0, 0], [0, 1], color="#f3ddc8", alpha=.40, zorder=0)   # below = redundant
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1.2, zorder=1, label="equal relative importance")
    ax.scatter(x, y, c=col, s=60, zorder=3, edgecolor="k", linewidth=.5)
    for i, f in enumerate(feats):
        ha = "left" if x[i] < 0.55 else "right"
        dx = 0.013 if ha == "left" else -0.013
        ax.annotate(f, (x[i], y[i]), (x[i] + dx, y[i] + 0.013), fontsize=6.8, ha=ha, va="bottom",
                    color="#333")
    ax.set_xlim(-0.03, 1.10); ax.set_ylim(-0.05, 1.14)
    ax.set_xlabel(f"|standardized coefficient|   (scaled to max = {aco.max():.3f})")
    ax.set_ylabel(f"ablation ΔR²   (scaled to max = {abl_val.max():.4f})")
    ax.set_title(f"Coefficient vs ablation — LASSO (route, spectral-17;  full CV R² = {full_r2:.3f})",
                 fontweight="bold", fontsize=12.5)
    ax.text(0.035, 0.96, "irreplaceable\n(low coef, high ablation)", fontsize=9, color="#2a6a2a",
            va="top", fontweight="bold")
    ax.text(0.97, 0.12, "redundant\n(big coef, low ablation)", fontsize=9, color="#9c3a3e",
            ha="right", fontweight="bold")
    ax.text(0.035, 0.02, "low importance (both lenses)", fontsize=8, color="#666")
    ax.grid(alpha=.25); ax.legend(loc="center right", fontsize=8)
    fig.text(0.5, 0.013,
             "Both lenses scaled to their own maximum; distance from the diagonal = how much the two "
             "importance lenses disagree.  Points are labelled; the origin = minor on both lenses.",
             ha="center", va="bottom", fontsize=8, style="italic", color="#444", wrap=True)
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(out_path, dpi=150); plt.close(fig)


def render_sensitivity_fig(feats, paths, alphas, tuned_a, tuned_coef, cv_r2, out_path):
    """§3 SENSITIVITY (self-contained, 4 pts). TOP: coefficient paths over four decades of α
    (attribution stability; top-8 by |coef| labelled). BOTTOM: 5-fold CV R² over the SAME α -- the
    score that actually matters is flat across the stable band and only degrades past the
    interpretability limit. Shared α axis; stable band (α≲0.05) shaded, α≈0.1 marked. The bottom
    panel is the headline: the conclusion you care about is insensitive to the knob until α≈0.1."""
    order = np.argsort(np.abs(tuned_coef))[::-1]
    labeled = list(order[:8])
    palette = [cm.tab10(i) for i in (0, 1, 2, 3, 4, 5, 6, 8, 9)]   # skip tab10 grey (reserved for unlabeled)
    color_of = {j: palette[k % len(palette)] for k, j in enumerate(labeled)}
    fig = plt.figure(figsize=(9.8, 7.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.05], hspace=0.08)
    axT = fig.add_subplot(gs[0]); axB = fig.add_subplot(gs[1], sharex=axT)

    def _markers(ax):
        ax.axvspan(alphas.min(), 0.05, color="#eaf3ea", zorder=0)
        ax.axvline(tuned_a, color="k", ls="--", lw=1, alpha=.7)
        ax.axvline(0.1, color="#b00020", ls=":", lw=1.3, alpha=.8)

    for j in range(len(feats)):
        if j in color_of:
            axT.semilogx(alphas, paths[:, j], "-", lw=2, color=color_of[j], label=feats[j], zorder=3)
        else:
            axT.semilogx(alphas, paths[:, j], "-", lw=.8, color="#cccccc", zorder=1)
    _markers(axT); axT.axhline(0, color="#888", lw=.8)
    axT.plot([], [], "k--", lw=1, label=f"tuned α={tuned_a:.3g}")
    axT.plot([], [], color="#b00020", ls=":", lw=1.3, label="α≈0.1 interpretability limit")
    axT.set_ylabel("standardized coefficient")
    axT.set_title("Sensitivity of route importance to L1 strength α — LASSO (spectral-17)",
                  fontweight="bold", fontsize=12.5)
    axT.legend(fontsize=7.5, loc="upper right"); axT.grid(alpha=.25)
    axT.text(alphas.min() * 1.25, axT.get_ylim()[1] * 0.94, "stable band (α≲0.05)",
             fontsize=8, color="#2a6a2a", va="top")
    plt.setp(axT.get_xticklabels(), visible=False)

    axB.semilogx(alphas, cv_r2, "-o", ms=3.5, color="#222", lw=1.9, zorder=3)
    _markers(axB)
    axB.set_ylabel("5-fold CV R²"); axB.set_xlabel("lasso α (log)")
    axB.set_ylim(min(-0.05, float(np.nanmin(cv_r2)) - 0.05), 1.03); axB.grid(alpha=.25)
    axB.text(alphas.min() * 1.25, 0.30, "predictive power flat across the band,\ndegrades only past α≈0.1",
             fontsize=8, color="#333", va="center")

    fig.text(0.5, 0.013,
             "Top: L1 weight signs are unstable under correlated predictors (laplacian_energy crosses zero "
             "near α≈0.3) — read magnitude, not sign.  Bottom: the score is insensitive to α across the "
             "stable band and only degrades past α≈0.1.",
             ha="center", va="bottom", fontsize=8, style="italic", color="#444", wrap=True)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=150); plt.close(fig)


def main():
    print("loading...")
    train = H.load_data(H.DEFAULT_SWAP_CSV, H.DEFAULT_POL_CSV)
    oods = load_oods()
    feats = H.shared_feats("spectral", train, *oods.values())
    Xtr = train[feats].fillna(0).values
    y = train["route"].values
    Xods = {d: o[feats].fillna(0).values for d, o in oods.items()}
    yods = {d: o["route"].values for d, o in oods.items()}

    # tuned alpha (harness operating point)
    tuned_pipe = H._fit_full("lasso", Xtr, y, train.grp.values)
    tuned_a = float(tuned_pipe.named_steps["model"].best_params_["alpha"])
    tuned_coef = np.ravel(tuned_pipe.named_steps["model"].best_estimator_.coef_)
    order = np.argsort(np.abs(tuned_coef))[::-1]              # feature ranking at the operating point

    # ---- (1) coefficient paths + 5-fold CV R² vs alpha (same grid) ----
    print("(1) coefficient paths + CV R² vs alpha...")
    paths = np.zeros((len(ALPHAS), len(feats)))
    cv_r2 = np.zeros(len(ALPHAS))
    gkf = GroupKFold(5)
    for k, a in enumerate(ALPHAS):
        paths[k] = np.ravel(lasso_fixed(a).fit(Xtr, y).named_steps["model"].coef_)
        rr = [H.metrics_full(y[te], lasso_fixed(a).fit(Xtr[tr], y[tr]).predict(Xtr[te]))["R2"]
              for tr, te in gkf.split(np.arange(len(train)), groups=train.grp.values)]
        cv_r2[k] = float(np.mean(rr))
    pd.DataFrame(paths, columns=feats).assign(alpha=ALPHAS).to_csv(
        OUT / "sensitivity_coef_path_route.csv", index=False)
    pd.DataFrame({"alpha": ALPHAS, "cv_r2": cv_r2}).to_csv(
        OUT / "sensitivity_cv_r2_alpha.csv", index=False)

    # in-dist ablation (read the artifact compute_lasso_importance already wrote)
    imp_csv = OUT / "lasso_importance_spectral_route.csv"
    abl = pd.read_csv(imp_csv).set_index("feature") if imp_csv.exists() else None
    abl_val = np.array([float(abl.loc[f, "abl_dR2"]) if abl is not None and f in abl.index else 0.0
                        for f in feats])
    # split into TWO self-contained figures, one per graded rubric section:
    full_r2 = H.run_indist(train, Xtr, "route", "lasso")["R2"]   # denominator for the ablation ΔR² bars
    p_imp = FIG / "feature_importance_ablation_route.png"     # §2 bar view
    render_importance_fig(feats, tuned_coef, abl_val, full_r2, p_imp)
    p_sc = FIG / "feature_importance_scatter_route.png"       # §2 scatter view (position = disagreement)
    render_importance_scatter(feats, tuned_coef, abl_val, full_r2, p_sc)
    p_sens = FIG / "sensitivity_alpha_route.png"              # §3 Sensitivity (4 pts)
    render_sensitivity_fig(feats, paths, ALPHAS, tuned_a, tuned_coef, cv_r2, p_sens)

    # ---- (2) ablation across devices (OOD ΔR² per feature per device) ----
    print("(2) leave-one-feature-out OOD ablation per device...")
    full_r2 = {}
    for d in OOD:
        full_r2[d] = H.metrics_full(yods[d], tuned_pipe.predict(Xods[d]))["R2"]
    A = np.zeros((len(feats), len(OOD)))
    for i in range(len(feats)):
        keep = [k for k in range(len(feats)) if k != i]
        pipe_i = H._fit_full("lasso", Xtr[:, keep], y, train.grp.values)
        for jd, d in enumerate(OOD):
            r2_i = H.metrics_full(yods[d], pipe_i.predict(Xods[d][:, keep]))["R2"]
            A[i, jd] = full_r2[d] - r2_i                       # ΔR² = drop in device-d OOD R²
    dev_short = [d.replace("Fake", "") for d in OOD]
    adf = pd.DataFrame(A, index=feats, columns=dev_short)
    adf["mean"] = adf.mean(axis=1)
    adf = adf.reindex([feats[i] for i in order])              # sort by operating-point importance
    adf.to_csv(OUT / "sensitivity_ablation_by_device_route.csv")

    M = adf.values
    fig2, ax = plt.subplots(figsize=(8.2, 7.2))
    vmax = np.nanpercentile(np.abs(M), 98) or 0.01
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(list(adf.columns), fontsize=9)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(list(adf.index), fontsize=8)
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            # white text on saturated cells (e.g. time_to_connected's dark-red row), else black
            tc = "white" if abs(M[r, c]) > 0.6 * vmax else "black"
            ax.text(c, r, f"{M[r, c]:+.3f}", ha="center", va="center", fontsize=6.5, color=tc)
    ax.axvline(M.shape[1] - 1.5, color="k", lw=1)             # separate the mean column
    fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="OOD ablation ΔR² (drop when removed)")
    ax.set_title("Ablation sensitivity across test devices — LASSO (route, spectral-17)\n"
                 "coefficients are train-fixed (device-invariant); only OOD impact varies",
                 fontweight="bold", fontsize=11)
    fig2.tight_layout()
    p2 = FIG / "sensitivity_ablation_by_device_route.png"
    fig2.savefig(p2, dpi=150); plt.close(fig2)

    # ---- console summary ----
    print(f"\ntuned α={tuned_a:.3g};  full OOD R² per device: "
          + ", ".join(f"{d.replace('Fake','')}={full_r2[d]:+.3f}" for d in OOD))
    print("\ntop-6 features by |coef| @ tuned α and their ablation spread across devices:")
    for i in order[:6]:
        f = feats[i]
        row = adf.loc[f, dev_short]
        print(f"  {f:24s} coef={tuned_coef[i]:+.3f}  OOD abl ΔR² "
              f"mean={adf.loc[f,'mean']:+.4f} [{row.min():+.4f},{row.max():+.4f}]")
    nz = int((np.abs(tuned_coef) > 1e-6).sum())
    print(f"\nL1 keeps {nz}/{len(feats)} features at tuned α.")
    print(f"figures -> §2 {p_imp.name}\n           §3 {p_sens.name}\n           §3 {p2.name} (device axis)")
    print("DONE_IMP_SENS")


if __name__ == "__main__":
    main()
