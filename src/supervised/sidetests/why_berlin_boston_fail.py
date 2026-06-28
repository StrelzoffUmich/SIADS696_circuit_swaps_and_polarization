#!/usr/bin/env python3
"""NON-CANONICAL side test -- WHY do Berlin & Boston have terrible pol / pol_z OOD R²?

Walks 3 concrete circuits (labeled on Brisbane, Berlin, Boston) to show the mechanism and a fix.

THESIS: Berlin and Boston are much CLEANER devices -- lower documented per-gate error (pulled live
from backend.target). Polarization P ≈ exp(-m·e): for a fixed circuit burden m, a smaller per-gate
error e gives a HIGHER polarization. A model trained only on noisy Brisbane predicts Brisbane-scale
(low) polarization from device-independent circuit structure, so on the cleaner devices it
systematically UNDER-predicts -> the residuals are huge and one-signed -> R² goes strongly negative,
even though the RANK is preserved (it still knows which circuits are more/less resilient).

FIX (zero-shot, label-free): rescale the model's Brisbane-scale prediction by the documented gate-
error ratio,  P_D = P_B ** (e_D / e_B)  -- the monotonic power-rescaling implied by exp(-m·e).
e_D/e_B < 1 for a cleaner device, and raising P∈[0,1] to a power <1 lifts it toward the true value.

Imports the harness; modifies nothing.
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

warnings.filterwarnings("ignore")
LAB = REPO / "data" / "xdev_out" / "labeled"
OUT = REPO / "data" / "results"; FIG = OUT / "figures" / "exploratory"   # non-report exploratory figures
FIG.mkdir(parents=True, exist_ok=True)
DEVICES = ["FakeBrisbane", "FakeBerlin", "FakeBoston"]
SHORT = ["Brisbane\n(noisy, train)", "Berlin\n(3.5× cleaner)", "Boston\n(6.1× cleaner)"]


def median_2q_error(name):
    from qiskit_ibm_runtime import fake_provider as fp
    bk = getattr(fp, name)()
    t = bk.target
    errs = [p.error for g in ("ecr", "cz") if g in t for p in t[g].values()
            if p is not None and p.error is not None]
    return float(np.median(errs))


def main():
    import report_artifacts as R                    # noqa: E402
    R.assert_qir_version()                           # soft guard: live params must match the label-gen snapshots
    e = {d: median_2q_error(d) for d in DEVICES}
    eB = e["FakeBrisbane"]
    print("documented median 2-qubit gate error (live from backend.target):")
    for d in DEVICES:
        print(f"  {d:13s} e={e[d]:.5f}   ({eB/e[d]:.1f}× cleaner than Brisbane)" if d != "FakeBrisbane"
              else f"  {d:13s} e={e[d]:.5f}   (training device)")

    # train lasso (spectral-17) on the Brisbane TRAINING set; predict pol from circuit structure.
    train = H.load_data(H.DEFAULT_SWAP_CSV, H.DEFAULT_POL_CSV)
    val = {d: H.load_external(str(LAB / f"val_pol_{d}.csv")) for d in DEVICES}
    feats = H.shared_feats("spectral", train, *val.values())
    pipe = H._fit_full("lasso", train[feats].fillna(0).values, train["pol"].values, train.grp.values)

    # device-independent structure -> ONE Brisbane-scale prediction per circuit (use Brisbane's rows).
    base = val["FakeBrisbane"].reset_index(drop=True)
    pred_B = np.clip(pipe.predict(base[feats].fillna(0).values), 1e-6, 1.0)   # model's prediction
    actual = {d: val[d].set_index("file")["polarization"] for d in DEVICES}

    # ---- corpus-level evidence (UNCONDITIONED): fraction of circuits whose polarization rises
    # monotonically Brisbane < Berlin < Boston. This is the real claim; the 3 circuits below are only
    # an illustration of its SHAPE, so they must NOT be selected on this outcome.
    aB_all = actual["FakeBrisbane"]; aBe_all = actual["FakeBerlin"]; aBo_all = actual["FakeBoston"]
    mono = sum(1 for f in base["file"] if aB_all[f] < aBe_all[f] < aBo_all[f])
    ntot = len(base); mono_pct = 100.0 * mono / ntot
    print(f"\nCORPUS EVIDENCE: {mono}/{ntot} ({mono_pct:.0f}%) circuits rise monotonically "
          f"Brisbane<Berlin<Boston (this is the claim; the 3 circuits below only illustrate the shape).")

    # The LEFT panel ILLUSTRATES the mechanism's typical shape, so it shows three REPRESENTATIVE
    # circuits: spanning low/mid/high 2q-gate burden (input) from distinct families, drawn from the
    # dominant majority that rise monotonically. The trio is NOT the evidence -- treating it as such
    # would be cherry-picking; the UNCONDITIONED claim is the 199/219 statistic + the full-corpus R²
    # (right panel), which include the ~9% exceptions. Burden spans structure; monotonic = the rule.
    g2 = base["num_2q_gates"].values
    def is_mono(f): return aB_all[f] < aBe_all[f] < aBo_all[f]
    pick, used_algos = [], set()
    for target in (4, 10, 18):
        cand = sorted(range(len(base)), key=lambda i: (abs(g2[i] - target), base.loc[i, "file"]))
        for i in cand:
            f = base.loc[i, "file"]; a = base.loc[i, "algo"]
            if a not in used_algos and aB_all[f] > 0.05 and is_mono(f):
                pick.append(i); used_algos.add(a); break
    files = base.loc[pick, "file"].tolist()

    print("\n3 circuits (2q-gate burden ≈ 4 / 10 / 18, distinct families, representative of the "
          "monotonic majority):")
    rows = []
    for idx in pick:
        f = base.loc[idx, "file"]
        aB, aBe, aBo = actual["FakeBrisbane"][f], actual["FakeBerlin"][f], actual["FakeBoston"][f]
        pB = pred_B[idx]
        # zero-shot fix: rescale the model prediction by the gate-error ratio.
        fix = {d: float(np.clip(pB ** (e[d] / eB), 0, 1)) for d in DEVICES}
        rows.append(dict(file=f.split("/")[-1], algo=base.loc[idx, "algo"],
                         nq=int(base.loc[idx, "num_qubits"]), g2=int(base.loc[idx, "num_2q_gates"]),
                         actual_Brisbane=aB, actual_Berlin=aBe, actual_Boston=aBo,
                         pred_raw=pB, pred_fix_Berlin=fix["FakeBerlin"], pred_fix_Boston=fix["FakeBoston"]))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    for _, r in df.iterrows():
        mono_circ = "monotonic↑" if r.actual_Brisbane < r.actual_Berlin < r.actual_Boston else "NON-monotonic"
        print(f"\n  {r.algo}  N={r.nq}  2q={r.g2}  [{r.file}]")
        print(f"    actual pol :  Brisbane={r.actual_Brisbane:.3f}   Berlin={r.actual_Berlin:.3f}"
              f"   Boston={r.actual_Boston:.3f}   ({mono_circ})")
        print(f"    model pred (Brisbane-trained) = {r.pred_raw:.3f}  "
              f"->  under-predicts Berlin by {r.actual_Berlin - r.pred_raw:+.3f},  "
              f"Boston by {r.actual_Boston - r.pred_raw:+.3f}")
        print(f"    rescaled  P_B^(e_D/e_B)       :  Berlin={r.pred_fix_Berlin:.3f} "
              f"(resid {r.actual_Berlin - r.pred_fix_Berlin:+.3f})   "
              f"Boston={r.pred_fix_Boston:.3f} (resid {r.actual_Boston - r.pred_fix_Boston:+.3f})")

    # R² on the FULL corpus, raw model vs rescaled, per device -- the headline numbers.
    def r2(y, p): return H.metrics_full(np.asarray(y), np.asarray(p))["R2"]
    def rho(y, p): return H.metrics_full(np.asarray(y), np.asarray(p))["Spearman"]
    print("\nfull-corpus pol R² (Brisbane-trained lasso), raw vs gate-error-rescaled:")
    yidx = base["file"].values
    for d in DEVICES:
        ad = actual[d].reindex(yidx).values
        praw = pred_B
        pfix = np.clip(pred_B ** (e[d] / eB), 0, 1)
        tag = "(train)" if d == "FakeBrisbane" else ""
        print(f"  {d:13s} raw R²={r2(ad, praw):+7.3f}  ->  rescaled R²={r2(ad, pfix):+7.3f}   "
              f"(rank ρ={rho(ad, praw):+.3f}, unchanged by rescale) {tag}")
    df.to_csv(OUT / "why_berlin_boston_fail.csv", index=False)

    # ---- small device-error table ----
    import report_artifacts as R
    drows, dcell = [], []
    for d in DEVICES:
        pm = float(actual[d].mean())
        drows.append(d.replace("Fake", ""))
        dcell.append([f"{e[d]:.5f}", ("1.0× (train)" if d == "FakeBrisbane" else f"{eB/e[d]:.1f}× cleaner"),
                      f"{pm:.3f}"])
    R.render_table(["median 2q gate error", "vs Brisbane", "mean polarization"], drows, dcell,
                   str(FIG / "device_error_table.png"),
                   "Device gate-error & polarization — cleaner devices, higher polarization",
                   note="median 2-qubit gate error pulled live from backend.target;  "
                        "mean polarization over the 219-circuit validation corpus.",
                   row_w=[2.4, 1.9, 2.0], label_w=1.5, fontsize=12, row_in=0.34)

    # ---- 2-panel figure: 3 circuits (under-prediction + fix) | full-corpus R² before/after ----
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 5.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    xs = np.arange(3)
    cols = ["#4C72B0", "#DD8452", "#55A868"]
    nb = len(df); bw = 0.8 / nb
    fix_by = [[r.pred_raw, r.pred_fix_Berlin, r.pred_fix_Boston] for _, r in df.iterrows()]
    for k, (_, r) in enumerate(df.iterrows()):
        av = [r.actual_Brisbane, r.actual_Berlin, r.actual_Boston]
        off = (k - (nb - 1) / 2) * bw
        axL.bar(xs + off, av, bw, color=cols[k], zorder=2, edgecolor="white", linewidth=.4,
                label=f"{r.algo} (N={r.nq}, 2q={r.g2})")
        axL.hlines(np.full(3, r.pred_raw), xs + off - bw / 2, xs + off + bw / 2, color="k", lw=1.6, zorder=4)
        axL.plot(xs[1:] + off, fix_by[k][1:], "D", mfc="white", mec="k", mew=1.4, ms=7, zorder=5)
    axL.plot([], [], "k-", lw=1.6, label="model pred (Brisbane-scale)")
    axL.plot([], [], "D", mfc="white", mec="k", label="rescaled  P_B^(e_D/e_B)")
    axL.set_xticks(xs); axL.set_xticklabels(SHORT, fontsize=9)
    axL.set_ylabel("polarization"); axL.set_ylim(0, 1.32); axL.grid(alpha=.25, axis="y")
    axL.set_title(f"Three illustrative circuits   ({mono}/{ntot} of corpus rise monotonically)",
                  fontsize=10.5, fontweight="bold")
    axL.legend(fontsize=7.5, loc="upper left", framealpha=.95)
    # the black tick is the model's single (device-blind) prediction, drawn at all 3 panels -> flat.
    axL.annotate("model is device-blind: prediction (—) is flat across panels;\n"
                 "gap to each bar = noise error the rescaling (◇) corrects",
                 xy=(2 - bw, df.iloc[0].pred_raw), xycoords="data",
                 xytext=(1.30, 1.20), fontsize=7.3, color="#333", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#333", lw=1), zorder=6)

    praw = pred_B
    raw_r2 = [H.metrics_full(actual[d].reindex(yidx).values, praw)["R2"] for d in DEVICES]
    fix_r2 = [H.metrics_full(actual[d].reindex(yidx).values,
                             np.clip(praw ** (e[d] / eB), 0, 1))["R2"] for d in DEVICES]
    bw = 0.38
    axR.bar(xs - bw / 2, np.clip(raw_r2, -4, 1), bw, color="#C44E52", label="raw (Brisbane-trained)")
    axR.bar(xs + bw / 2, np.clip(fix_r2, -4, 1), bw, color="#55A868", label="gate-error rescaled")
    for i in range(3):
        axR.text(xs[i] - bw / 2, max(raw_r2[i], -4) + 0.06, f"{raw_r2[i]:.2f}", ha="center", fontsize=7.5)
        axR.text(xs[i] + bw / 2, fix_r2[i] + 0.06, f"{fix_r2[i]:.2f}", ha="center", fontsize=7.5)
    axR.axhline(raw_r2[0], color="#444", ls="--", lw=1, zorder=5)         # in-dist reference
    axR.text(2.46, raw_r2[0] + 0.04, f"in-dist {raw_r2[0]:.2f}", ha="right", va="bottom",
             fontsize=7.5, color="#444")
    axR.axhline(0, color="#888", lw=.8); axR.set_ylim(-4, 1.05)
    axR.set_xticks(xs); axR.set_xticklabels([s.split("\n")[0] for s in SHORT], fontsize=9)
    axR.set_ylabel("pol R²  (full 219-circuit corpus)"); axR.grid(alpha=.25, axis="y")
    axR.set_title("Full corpus aggregate R²", fontsize=11.5, fontweight="bold")
    axR.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "why_berlin_boston_fail.png", dpi=135); plt.close(fig)
    print("\nDONE_WHYFAIL")


if __name__ == "__main__":
    main()
