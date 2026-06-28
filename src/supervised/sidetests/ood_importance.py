#!/usr/bin/env python3
"""NON-CANONICAL side test -- feature importance & ablation on the CROSS-DEVICE OOD metric, rendered
in the table_lasso_importance style (report_artifacts.render_table). Target via argv (default
route): `python sidetests/ood_importance.py route` (or pol / pol_z).

Same recipe as the in-dist importance table, but the ablation column is the drop in cross-device R²
(train full Brisbane -> test the 4 OOD devices, mean over devices) when each feature is removed --
not the in-dist CV drop. Coefficients are the train-fit lasso (device-independent). Spearman Δρ is
shown alongside (the meaningful cross-device signal when level R² is weak/negative). LASSO spectral-17.
Imports the harness; modifies nothing.
"""
from __future__ import annotations
import sys, pathlib, warnings
HERE = pathlib.Path(__file__).resolve(); SUP = HERE.parent.parent; REPO = SUP.parent.parent
sys.path.insert(0, str(SUP))
import supervised_analysis_run as H               # noqa: E402
import report_artifacts as R                      # noqa: E402  (render_table + OUT)
import diagnostics                                # noqa: E402  (model_importances)
import os, numpy as np, pandas as pd              # noqa: E402

warnings.filterwarnings("ignore")
LAB = REPO / "data" / "xdev_out" / "labeled"
OOD = ["FakeSherbrooke", "FakeTorino", "FakeBerlin", "FakeBoston"]

TARGET = sys.argv[1] if len(sys.argv) > 1 else "route"
ARM = {"route": "swap", "pol": "pol", "pol_z": "pol"}[TARGET]
BASE = {"route": "route", "pol": "pol", "pol_z": "pol"}[TARGET]
REL = TARGET == "pol_z"
HASATTR = {"swap": "has_route", "pol": "has_pol"}[ARM]


def ytr_yood(train, o):
    if REL:
        stats, gm, gs = H.fit_zstats(train[BASE].values, train.N.values)
        return (H.apply_zstats(train[BASE].values, train.N.values, stats, gm, gs),
                H.apply_zstats(o[BASE].values, o.N.values, stats, gm, gs))
    return train[BASE].values.astype(float), o[BASE].values.astype(float)


def render_ood_table(target, df, base_r2, base_rho):
    """render_table: std coef + cross-device OOD ablation ΔR²/Δρ, sorted by |coef|.
    Note is wrapped to 3 short lines and names the test devices so the cross-device protocol is
    explicit."""
    cols = ["std. coef", "OOD ablation ΔR²", "OOD ablation Δρ"]
    rlbl, cell, dim = [], [], set()
    for i, (_, r) in enumerate(df.iterrows()):
        rlbl.append(r.feature)
        cell.append([f"{r.coef:+.3f}", f"{r.abl_r2:+.4f}", f"{r.abl_rho:+.4f}"])
        if not r.selected:
            dim |= {(i, 0), (i, 1), (i, 2)}
    caveat = "" if base_r2 > 0 else "   R²<0 ⇒ level doesn't transfer; read ρ."
    note = ("std. coef = per-SD lasso weight (grey = L1-zeroed).   "
            "ablation Δ = drop in the metric when the feature is removed.\n"
            "Cross-device: train Brisbane → test Sherbrooke / Torino / Berlin / Boston "
            "(mean over the 4 held-out devices).\n"
            f"baseline OOD R² = {base_r2:+.3f},  ρ = {base_rho:+.3f}.{caveat}")
    R.render_table(cols, rlbl, cell, os.path.join(R.OUT_EXPLORATORY, f"table_lasso_importance_ood_{target}.png"),
                   f"Feature importance & ablation — LASSO ({target}, spectral-17), CROSS-DEVICE OOD",
                   note=note, dim_cells=dim, row_w=[1.7, 2.2, 2.1], label_w=2.9, fontsize=11, row_in=0.24)


def main():
    train = H.load_data(H.DEFAULT_SWAP_CSV, H.DEFAULT_POL_CSV)
    oods = {}
    for d in OOD:
        p = LAB / f"val_{ARM}_{d}.csv"
        if p.exists():
            o = H.load_external(str(p))
            if o.attrs.get(HASATTR):
                oods[d] = o
    feats = H.shared_feats("spectral", train, *oods.values())
    X = train[feats].fillna(0).values
    ytr = ytr_yood(train, next(iter(oods.values())))[0]
    yoods = {d: ytr_yood(train, o)[1] for d, o in oods.items()}

    def ood_mean(pipe, cols_idx):
        r2, rho = [], []
        for d, o in oods.items():
            Xo = o[[feats[i] for i in cols_idx]].fillna(0).values
            m = H.metrics_full(yoods[d], pipe.predict(Xo))
            r2.append(m["R2"]); rho.append(m["Spearman"])
        return float(np.mean(r2)), float(np.mean(rho))

    base_pipe = H._fit_full("lasso", X, ytr, train.grp.values)
    _lab, _names, coef = diagnostics.model_importances(base_pipe, feats)
    base_r2, base_rho = ood_mean(base_pipe, list(range(len(feats))))

    rows = []
    for i, f in enumerate(feats):
        keep = [k for k in range(len(feats)) if k != i]
        pipe_i = H._fit_full("lasso", X[:, keep], ytr, train.grp.values)
        r2_i, rho_i = ood_mean(pipe_i, keep)
        rows.append(dict(feature=f, coef=float(coef[i]),
                         abl_r2=base_r2 - r2_i, abl_rho=base_rho - rho_i,
                         selected=bool(abs(coef[i]) > 1e-6)))
    df = (pd.DataFrame(rows)
          .reindex(pd.DataFrame(rows).coef.abs().sort_values(ascending=False).index)
          .reset_index(drop=True))
    df.to_csv(str(REPO / "data" / "results" / f"lasso_importance_ood_{TARGET}.csv"), index=False)

    render_ood_table(TARGET, df, base_r2, base_rho)
    print(f"[{TARGET}] baseline OOD R²={base_r2:+.4f}  ρ={base_rho:+.4f};  L1 keeps "
          f"{int(df.selected.sum())}/{len(df)} features. -> table_lasso_importance_ood_{TARGET}.png")
    print("DONE_OOD_IMP")


if __name__ == "__main__":
    main()
