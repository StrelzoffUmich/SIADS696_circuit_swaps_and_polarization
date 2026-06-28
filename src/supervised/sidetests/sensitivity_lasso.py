#!/usr/bin/env python3
"""NON-CANONICAL side test — SENSITIVITY ANALYSIS on the best model (LASSO), cross-device protocol.

Rubric §3: "how sensitive are your results to choice of (hyper-)parameters, features, or other
varying solution elements?" Best model = LASSO -- it wins the cross-device transfer (the question
that matters here; see the report's overall summary). Everything below trains on the full
in-distribution Brisbane set and tests on each device's held-out validation corpus (the same
run_external protocol the harness/report use), so this is a sensitivity study of the *reported*
(cross-device) result, not the uninformative in-distribution one.

Two axes:
  (A) HYPER-PARAMETER -- sweep lasso's regularization alpha (its one knob; the harness tunes it
      over logspace(-3,1,50) by GroupKFold). For the chosen feature rep (spectral-17) we plot
      in-dist 5-fold CV R2 and pooled-OOD R2/Spearman vs alpha, and mark the harness-tuned alpha.
      Question answered: is the transfer result robust to the regularization choice?
  (B) FEATURE -- the 5 harness feature sets, raw vs log1p(counts). The raw OOD blow-ups are an
      extrapolation artifact: unbounded/extensive features (depth, size, depth_per_qubit, ...)
      run 10-50x outside the TRAIN range on the OOD corpus, and a linear model extrapolates them.
      Logging the count columns isolates how much of the spread is just SCALING; the residual
      (keep4, krystian) is genuine off-support sensitivity, quantified by a support-overlap ratio.

Imports the harness; does not modify it. Reuses StandardScaler+Lasso (identical estimator to
make_lasso) and harness primitives (load_data/load_external, shared_feats, resolve_target,
fit_zstats/apply_zstats, metrics_full, _fit_full).
"""
from __future__ import annotations
import sys, pathlib, warnings
HERE = pathlib.Path(__file__).resolve()
SUP = HERE.parent.parent                                   # src/supervised
REPO = SUP.parent.parent                                   # repo root
sys.path.insert(0, str(SUP))
import supervised_analysis_run as H                         # noqa: E402
import numpy as np, pandas as pd                            # noqa: E402
import matplotlib                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                             # noqa: E402
from sklearn.pipeline import Pipeline                       # noqa: E402
from sklearn.preprocessing import StandardScaler            # noqa: E402
from sklearn.linear_model import Lasso                      # noqa: E402
from sklearn.model_selection import GroupKFold              # noqa: E402

warnings.filterwarnings("ignore")
LAB = REPO / "data" / "xdev_out" / "labeled"
OUT = REPO / "data" / "results"
FIG = OUT / "figures" / "exploratory"       # non-report exploratory figures
FIG.mkdir(parents=True, exist_ok=True)
COUNTS = ["num_qubits", "depth", "size", "num_2q_gates"]
FEATSETS = ["size_only", "basic", "keep4", "spectral", "krystian"]
OOD = ["FakeSherbrooke", "FakeTorino", "FakeBerlin", "FakeBoston"]
TARGETS = [("route", "route", False, "swap"), ("pol", "pol", False, "pol"),
           ("pol_z", "pol", True, "pol")]
ALPHAS = np.logspace(-3, 1, 25)                             # subset of the harness's 50-point grid


def lasso_fixed(alpha):
    return Pipeline([("scale", StandardScaler()),
                     ("model", Lasso(alpha=alpha, max_iter=20000))])


def _ytr_yood(train, ood, base, rel):
    """Train/OOD target vectors with the run_external recipe (z-stats fit on TRAIN for pol_z)."""
    if rel:
        stats, gm, gs = H.fit_zstats(train[base].values, train.N.values)
        ytr = H.apply_zstats(train[base].values, train.N.values, stats, gm, gs)
        yood = H.apply_zstats(ood[base].values, ood.N.values, stats, gm, gs)
    else:
        ytr = np.asarray(train[base].values, float)
        yood = np.asarray(ood[base].values, float)
    return ytr, yood


def load_ood(arm):
    out = {}
    for dev in OOD:
        p = LAB / f"val_{arm}_{dev}.csv"
        if p.exists():
            out[dev] = H.load_external(str(p))
    return out


# ---------- (A) hyper-parameter (alpha) sensitivity, spectral-17 -------------
def alpha_sweep(train, ood_by_arm):
    rows = []
    tuned = {}
    for tok, base, rel, arm in TARGETS:
        oods = ood_by_arm[arm]
        cols = H.shared_feats("spectral", train, *oods.values())
        Xtr = train[cols].fillna(0).values
        Xods = {d: o[cols].fillna(0).values for d, o in oods.items()}
        yt_by = {d: _ytr_yood(train, o, base, rel) for d, o in oods.items()}
        ytr_full = _ytr_yood(train, next(iter(oods.values())), base, rel)[0]
        # harness-tuned alpha (its actual operating point)
        pipe_t = H._fit_full("lasso", Xtr, ytr_full, train.grp.values)
        tuned[tok] = float(pipe_t.named_steps["model"].best_params_["alpha"])
        for a in ALPHAS:
            # in-dist 5-fold CV R2 at fixed alpha
            gkf = GroupKFold(5); r2cv = []
            for tr, te in gkf.split(np.arange(len(train)), groups=train.grp.values):
                tm = np.zeros(len(train), bool); tm[tr] = True
                y = H.resolve_target(train, base, tm, rel)
                p = lasso_fixed(a).fit(Xtr[tr], y[tr])
                r2cv.append(H.metrics_full(y[te], p.predict(Xtr[te]))["R2"])
            # OOD: fit on full train, predict every device, pool the metric per device then mean
            p = lasso_fixed(a).fit(Xtr, ytr_full)
            r2s, rhos = [], []
            for d, o in oods.items():
                _ytr, yood = yt_by[d]
                m = H.metrics_full(yood, p.predict(Xods[d]))
                r2s.append(m["R2"]); rhos.append(m["Spearman"])
            rows.append(dict(target=tok, alpha=a, indist_R2=float(np.mean(r2cv)),
                             ood_R2=float(np.mean(r2s)), ood_rho=float(np.mean(rhos))))
    return pd.DataFrame(rows), tuned


def fig_alpha(df, tuned):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (tok, *_ ) in zip(axes, TARGETS):
        s = df[df.target == tok].sort_values("alpha")
        ax.semilogx(s.alpha, s.indist_R2, "-o", ms=3, color="#4C72B0", label="in-dist CV R²")
        ax.semilogx(s.alpha, s.ood_R2, "-o", ms=3, color="#C44E52", label="OOD R²")
        ax.semilogx(s.alpha, s.ood_rho, "-o", ms=3, color="#55A868", label="OOD Spearman ρ")
        ax.axvline(tuned[tok], color="k", ls="--", lw=1, alpha=.7,
                   label=f"harness-tuned α={tuned[tok]:.3g}")
        ax.axhline(0, color="#999", lw=.8, zorder=0)
        ax.set_title(f"{tok}", fontweight="bold"); ax.set_xlabel("lasso α (log)")
        ax.set_ylim(-1.05, 1.05); ax.grid(alpha=.25); ax.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel("R² / ρ")
    fig.suptitle("Hyper-parameter sensitivity — LASSO (spectral-17), cross-device   "
                 "[clipped at R²=−1 for readability]", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = FIG / "sensitivity_lasso_alpha.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


# ---------- (B) feature sensitivity, raw vs log1p(counts) -------------------
def _logcounts(M, cols):
    M = M.copy()
    for c in COUNTS:
        if c in cols:
            M[c] = np.log1p(M[c].clip(lower=0))
    return M


def feature_sweep(train, ood_by_arm):
    rows = []
    for tok, base, rel, arm in TARGETS:
        oods = ood_by_arm[arm]
        for fs in FEATSETS:
            cols = H.shared_feats(fs, train, *oods.values())
            for variant in ("raw", "logcounts"):
                tr = _logcounts(train[cols], cols) if variant == "logcounts" else train[cols].copy()
                Xtr = tr.fillna(0).values
                ytr_full = _ytr_yood(train, next(iter(oods.values())), base, rel)[0]
                pipe = H._fit_full("lasso", Xtr, ytr_full, train.grp.values)   # harness-tuned lasso
                r2s, rhos = [], []
                for d, o in oods.items():
                    od = _logcounts(o[cols], cols) if variant == "logcounts" else o[cols].copy()
                    _ytr, yood = _ytr_yood(train, o, base, rel)
                    m = H.metrics_full(yood, pipe.predict(od.fillna(0).values))
                    r2s.append(m["R2"]); rhos.append(m["Spearman"])
                rows.append(dict(target=tok, features=fs, variant=variant, dim=len(cols),
                                 ood_R2=float(np.mean(r2s)), ood_rho=float(np.mean(rhos))))
    return pd.DataFrame(rows)


def support_overlap(train, ood_by_arm):
    """Per feature set: the worst OOD/train range-blowup factor (how far OOD pushes a feature
    beyond the training support). Explains which sets extrapolate -- the cause of the raw blow-up."""
    oods = ood_by_arm["swap"]
    rows = []
    for fs in FEATSETS:
        cols = H.shared_feats(fs, train, *oods.values())
        worst, worstcol = 1.0, None
        for c in cols:
            tlo, thi = float(train[c].min()), float(train[c].max())
            span = max(thi - tlo, 1e-9)
            for o in oods.values():
                olo, ohi = float(o[c].min()), float(o[c].max())
                blow = max(ohi - thi, tlo - olo, 0.0) / span      # excess beyond train range, in train-spans
                if blow > worst:
                    worst, worstcol = blow, c
        rows.append(dict(features=fs, worst_blowup_x=round(worst, 1), worst_feature=worstcol))
    return pd.DataFrame(rows)


def main():
    print("loading...")
    train = H.load_data(H.DEFAULT_SWAP_CSV, H.DEFAULT_POL_CSV)
    ood_by_arm = {"swap": load_ood("swap"), "pol": load_ood("pol")}

    print("\n(A) alpha sweep (lasso/spectral-17)...")
    a_df, tuned = alpha_sweep(train, ood_by_arm)
    a_df.to_csv(OUT / "sensitivity_lasso_alpha.csv", index=False)
    p = fig_alpha(a_df, tuned)
    print("=== (A) HYPER-PARAMETER SENSITIVITY — lasso/spectral-17 ===")
    for tok, *_ in TARGETS:
        s = a_df[a_df.target == tok]
        print(f"\n  {tok}: harness-tuned α={tuned[tok]:.3g}")
        print(f"    in-dist CV R²  range over α: [{s.indist_R2.min():+.3f}, {s.indist_R2.max():+.3f}]")
        print(f"    OOD R²         range over α: [{s.ood_R2.min():+.3f}, {s.ood_R2.max():+.3f}]")
        print(f"    OOD ρ          range over α: [{s.ood_rho.min():+.3f}, {s.ood_rho.max():+.3f}]")
        # OOD R2 across the central decade (0.01-1) = the plausible operating band
        band = s[(s.alpha >= 0.01) & (s.alpha <= 1.0)]
        print(f"    OOD R² spread within α∈[0.01,1]: "
              f"[{band.ood_R2.min():+.3f}, {band.ood_R2.max():+.3f}]  (Δ={band.ood_R2.max()-band.ood_R2.min():.3f})")
    print(f"\n  figure -> {p}")

    print("\n(B) feature sweep raw vs log1p(counts) (lasso-tuned)...")
    f_df = feature_sweep(train, ood_by_arm)
    f_df.to_csv(OUT / "sensitivity_lasso_features.csv", index=False)
    sup = support_overlap(train, ood_by_arm)
    print("=== (B) FEATURE SENSITIVITY — OOD R² (ρ), raw vs log1p(counts) ===")
    for tok, *_ in TARGETS:
        print(f"\n  -- {tok} --")
        print(f"    {'feature set':10s} {'dim':>3}  {'raw R²':>9} {'log R²':>9}   {'raw ρ':>7} {'log ρ':>7}")
        for fs in FEATSETS:
            r = f_df[(f_df.target == tok) & (f_df.features == fs)]
            raw = r[r.variant == "raw"].iloc[0]; log = r[r.variant == "logcounts"].iloc[0]
            print(f"    {fs:10s} {int(raw['dim']):>3}  {raw.ood_R2:>+9.2f} {log.ood_R2:>+9.2f}   "
                  f"{raw.ood_rho:>+7.3f} {log.ood_rho:>+7.3f}")
    print("\n  support-overlap diagnostic (worst OOD push beyond train range, in train-spans):")
    for _, r in sup.iterrows():
        print(f"    {r.features:10s} worst_blowup={r.worst_blowup_x:>7}×  via {r.worst_feature}")
    print("\nDONE_SENSITIVITY")


if __name__ == "__main__":
    main()
