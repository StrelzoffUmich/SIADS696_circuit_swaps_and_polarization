#!/usr/bin/env python3
"""NON-CANONICAL side test -- visualize SHERBROOKE's pol_z degeneration (failure mechanism 2:
within-N rank reordering, distinct from the Berlin/Boston noise-scale shift).

Three predicted-vs-true panels (lasso spectral-17, train Brisbane -> test device) that ISOLATE the
failure to the pol_z × Sherbrooke interaction:
  (a) Sherbrooke RAW pol     -> lines up (rank transfers; the device/level is fine)
  (b) Sherbrooke pol_z       -> BLOB (within-N resilience ranking collapses: ρ≈0.18) = the degeneration
  (c) Torino     pol_z       -> lines up (the z-transform is fine elsewhere; it's Sherbrooke-specific)

pol_z z-scores away the across-N gradient (which transfers) and isolates the within-N residual, which
is device-idiosyncratic; Sherbrooke (noise-MATCHED but a different chip) reorders it most. NOT a noise
floor -- the within-N spread is intact. Imports the harness; modifies nothing.
"""
from __future__ import annotations
import sys, pathlib, warnings
HERE = pathlib.Path(__file__).resolve(); SUP = HERE.parent.parent; REPO = SUP.parent.parent
sys.path.insert(0, str(SUP))
import supervised_analysis_run as H               # noqa: E402
import numpy as np                                # noqa: E402
import matplotlib                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402

warnings.filterwarnings("ignore")
LAB = REPO / "data" / "xdev_out" / "labeled"
FIG = REPO / "data" / "results" / "figures" / "exploratory"   # non-report exploratory figure
FIG.mkdir(parents=True, exist_ok=True)


def case(train, dev, relative):
    o = H.load_external(str(LAB / f"val_pol_{dev}.csv"))
    feats = H.shared_feats("spectral", train, o)
    Xtr = train[feats].fillna(0).values
    Xo = o[feats].fillna(0).values
    pt, _ci, pred = H.run_external(train, Xtr, o, Xo, "pol", "lasso", relative=relative, B=0)
    if relative:
        stats, gm, gs = H.fit_zstats(train["pol"].values, train.N.values)
        ytrue = H.apply_zstats(o["pol"].values, o.N.values, stats, gm, gs)
    else:
        ytrue = o["pol"].values.astype(float)
    return ytrue, pred, o.N.values, pt["R2"], pt["Spearman"]


def main():
    train = H.load_data(H.DEFAULT_SWAP_CSV, H.DEFAULT_POL_CSV)
    panels = [("FakeSherbrooke", False, "Sherbrooke — raw pol"),
              ("FakeSherbrooke", True, "Sherbrooke — pol_z"),
              ("FakeTorino", True, "Torino — pol_z")]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8))
    sc = None
    for ax, (dev, rel, title) in zip(axes, panels):
        y, p, N, r2, rho = case(train, dev, rel)
        lo = min(y.min(), p.min()); hi = max(y.max(), p.max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, alpha=.6, zorder=1)
        sc = ax.scatter(y, p, c=N, cmap="viridis", s=26, alpha=.8, edgecolor="white",
                        linewidth=.3, zorder=3)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("true"); ax.set_title(title, fontsize=11, fontweight="bold")
        ax.text(.04, .96, f"ρ = {rho:.2f}\nR² = {r2:.2f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=10,
                bbox=dict(boxstyle="round", fc="white", ec="#bbb", alpha=.9))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("predicted")
    cb = fig.colorbar(sc, ax=axes, fraction=0.026, pad=0.02); cb.set_label("qubit count N")
    fig.savefig(FIG / "sherbrooke_degeneration.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / "sherbrooke_degeneration.png")
    print("DONE_SHER")


if __name__ == "__main__":
    main()
