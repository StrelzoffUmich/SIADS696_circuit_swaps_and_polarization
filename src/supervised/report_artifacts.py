#!/usr/bin/env python3
"""
report_artifacts.py -- ONE script that reproduces every table and figure in the supervised
section of the report.

It does NO modeling of its own beyond hyperparameter SELECTION: every fit / metric / CV / OOD
evaluation is delegated to the canonical harness `supervised_analysis_run.py` (imported as H).
Models: kNN (instance/locality), Ridge + Lasso (regularized linear), HistGB + RF (tree ensembles)
-- five diverse families. Metrics: R2 (level), Spearman (rank), and within-N z-scored polarization
(relative resilience). Targets: route = log1p(bare_routed_2q), pol = mirror polarization,
pol_z = within-N z-scored pol (train-fit, no normalization leakage).

Pipeline (main): a preflight check verifies the full analysis has been run (training CSVs,
validation labels, cross_device results), then COMPUTE writes the CSVs each table reads and
RENDER draws the headline summary, the five tables, and the In-Depth-Evaluation importance table.

  RENDER (data/results/figures/, tables as .png + .csv, figures as .png)
    Table 1  in-distribution FakeBrisbane 5-fold CV (spectral-17) -- baseline/context
    Table 2  device held constant, unseen circuits (spectral-17)                 [cross_device.py CSV]
    Table 3  hyperparameters tuned in-distribution, tested once on validation
    Table 4  cross-device generalization by model family, 5-fold CV bar chart  -> headline_crossdevice_cv.png
    Table 5  LASSO feature representation: MQT-9 (log1p) vs spectral-17, route/pol/pol_z
    Table 6  LASSO feature importance & cross-device OOD ablation               -> table_lasso_importance_ood_route.png
    Table 7  per-device OOD ablation heatmap                                    -> sensitivity_ablation_by_device_route.png
    Table 8  device gate-error & polarization                                   -> device_error_table.png
    Fig 5    why Berlin/Boston fail (under-prediction + gate-error rescale)     -> why_berlin_boston_fail.png
    Fig 6    Sherbrooke pol_z degeneration (KNN, predicted-vs-true)             -> sherbrooke_degeneration.png
    Fig 7    MQT-9 unscaled vs log1p                                            -> mqt9_raw_vs_log.png
    +        spectral-17 Spearman correlation heatmap                           -> feature_correlation_spearman.png

Cross-device Table 2 reads cross_device.py's spectral results CSV; if absent it is skipped with a
note. Compute/figure steps read the cached validation labels under data/xdev_out/labeled/ (produced
by cross_device.py --phase label). Non-report exploratory tables/figures (overall OOD summary, the
15x5 OOD grid, in-distribution importance, Option A, few-shot calibration, prediction-vs-truth) live
in viz_tests/exploratory_artifacts.py. Reproduce importance numbers per-feature with the harness:
  supervised_analysis_run.py --model lasso --features spectral --target route --coef / --ablate

Run from anywhere:  python src/supervised/report_artifacts.py
"""
from __future__ import annotations

import ast
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor

# ---- paths + harness import (anchored to this file, runs from any cwd) ------
HERE = Path(__file__).resolve()
HARNESS_DIR = HERE.parent                       # src/supervised
REPO = HARNESS_DIR.parent.parent                # repo root
sys.path.insert(0, str(HARNESS_DIR))
import supervised_analysis_run as H             # noqa: E402  -- the canonical engine
from supervised_analysis_run import (           # noqa: E402
    load_data, load_external, run_indist, run_external, shared_feats,
    fit_zstats, apply_zstats, resolve_target, metrics_full,
    FEATURE_SETS, DEFAULT_SWAP_CSV, DEFAULT_POL_CSV,
)

# Figure outputs are split by audience: OUT = figures that appear IN the report (body + appendix);
# OUT_EXPLORATORY = exploratory side-script figures that are NOT in the report.
OUT = str(REPO / "data" / "results" / "figures" / "report")
OUT_EXPLORATORY = str(REPO / "data" / "results" / "figures" / "exploratory")
os.makedirs(OUT, exist_ok=True)
os.makedirs(OUT_EXPLORATORY, exist_ok=True)
RESULTS = str(REPO / "data" / "results")            # the results dir (CSV sink for the folded generators)
LABELED = REPO / "data" / "xdev_out" / "labeled"
TUNED_CSV = str(REPO / "data" / "results" / "cross_device_indist_tuned.csv")
LOG_LASSO_CSV = os.path.join(RESULTS, "mqt9_log_lasso.csv")   # data CSV -> results/, not figures/
CDV = {"spectral": str(REPO / "data/xdev_out/cross_device_results.csv"),
       "mqt9_raw": str(REPO / "data/xdev_out/mqt9_run/cross_device_results.csv")}

# train device + the four held-out OOD devices (used by preflight and the folded cross-device figures)
TRAIN_DEV = "FakeBrisbane"
OOD_DEVS = ["FakeBerlin", "FakeBoston", "FakeSherbrooke", "FakeTorino"]

MODELS = ["rf", "histgb", "ridge", "lasso", "knn"]
COUNTS = ["num_qubits", "depth", "size", "num_2q_gates"]
DEVICES = ["FakeBrisbane", "FakeBerlin", "FakeBoston", "FakeSherbrooke", "FakeTorino"]
DEV_SHORT = [d.replace("Fake", "") + ("*" if d == "FakeBrisbane" else "") for d in DEVICES]
REP_LABEL = {"mqt9_raw": "MQT-9 (raw)", "mqt9_log": "MQT-9 (log)", "spectral": "spectral-17"}
REP_COLOR = {"mqt9_raw": "#c44e52", "mqt9_log": "#dd8452", "spectral": "#4c72b0"}
ROUTE_DEF = "route = log1p(bare_routed_2q)"
# (token, base, relative) for the three targets
TARGETS3 = [("route", "route", False), ("pol", "pol", False), ("pol_z", "pol", True)]

# MQT-9 = mqt.predictor's verbatim calc_supermarq_features vector (4 counts + 5 SupermarQ flow
# metrics). Defined HERE rather than read from the harness's FEATURE_SETS so this script is
# self-contained and does not require the (optional) `mqt9` key to be present in the harness.
MQT9_FEATS = ["num_qubits", "depth", "size", "num_2q_gates", "program_communication",
              "critical_depth", "entanglement_ratio", "parallelism", "liveness"]


def _rep_base_cols(rep):
    """Ordered feature list for a representation token ('spectral' -> harness spectral-17;
    anything else -> the local MQT-9 vector)."""
    return list(FEATURE_SETS["spectral"]) if rep == "spectral" else MQT9_FEATS


def _rep_shared_cols(rep, *dfs):
    """Columns of the representation present in EVERY frame (train/OOD alignment), order preserved."""
    return [c for c in _rep_base_cols(rep) if all(c in d.columns for d in dfs)]


# ===========================================================================
# COMPUTE 1 -- in-distribution-tuned validation table
#
# Tune by 5-fold GroupKFold CV on TRAIN only; test the 344-circuit validation corpus exactly once.
# Hyperparameters are selected on in-distribution CV, never on OOD performance, so the test set is
# never peeked at. Sanity: lasso here matches the harness's own tuned lasso
# (`supervised_analysis_run.py --ood-csv ... --model lasso`), since that is itself a GroupKFold
# GridSearchCV over the same alpha range.
# ===========================================================================
TUNE_DEVICE = "FakeBrisbane"


def _configs(m):
    """The grid searched per model."""
    if m == "ridge":  return [{"alpha": a} for a in np.logspace(-4, 3, 30)]
    if m == "lasso":  return [{"alpha": a} for a in np.logspace(-3, 1, 25)]
    if m == "rf":     return [dict(zip(("n_estimators", "max_depth", "min_samples_leaf"), v))
                              for v in itertools.product([300, 600], [None, 12], [1, 3])]
    if m == "histgb": return [dict(zip(("learning_rate", "max_depth", "l2_regularization"), v))
                              for v in itertools.product([0.03, 0.08], [None, 4], [0.0, 1.0])]
    if m == "knn":    return [dict(zip(("n_neighbors", "weights"), v))
                              for v in itertools.product([3, 5, 9, 15, 30], ["uniform", "distance"])]


def _make(m, p):
    if m == "ridge":  return Ridge(max_iter=20000, **p)
    if m == "lasso":  return Lasso(max_iter=20000, **p)
    if m == "rf":     return RandomForestRegressor(random_state=0, n_jobs=-1, **p)
    if m == "histgb": return HistGradientBoostingRegressor(random_state=0, early_stopping=True,
                                                           max_iter=600, **p)
    if m == "knn":    return KNeighborsRegressor(**p)


def _pipe(m, p):
    """StandardScaler + estimator -- exactly the harness build_pipe (scale every family; harmless
    for trees, required for ridge/lasso/knn)."""
    return Pipeline([("scale", StandardScaler()), ("model", _make(m, p))])


def _cv_r2(m, p, X, df, base, rel, n_splits=5):
    """Mean GroupKFold CV R2 on TRAIN, grouped by df.grp -- the SAME protocol as run_indist
    (fresh pipeline per fold; relative-target z-stats fit on the TRAIN fold only)."""
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for tr, te in gkf.split(np.arange(len(df)), groups=df.grp.values):
        train_mask = np.zeros(len(df), bool); train_mask[tr] = True
        yall = resolve_target(df, base, train_mask, rel)     # per-fold z-stats for pol_z
        est = _pipe(m, p).fit(X[tr], yall[tr])
        scores.append(metrics_full(yall[te], est.predict(X[te]))["R2"])
    return float(np.nanmean(scores))


def compute_indist_tuned(train):
    """For each model+target: select the config with the best TRAIN CV R2, then evaluate it once on
    the FakeBrisbane validation corpus. Writes cross_device_indist_tuned.csv."""
    print("== COMPUTE 1: in-distribution tuning, single validation touch ==")
    X_tr = train[shared_feats("spectral", train)].fillna(0).values
    swap_val = load_external(str(LABELED / f"val_swap_{TUNE_DEVICE}.csv"))
    pol_val = load_external(str(LABELED / f"val_pol_{TUNE_DEVICE}.csv"))

    rows = []
    for tok, base, rel in TARGETS3:
        val = swap_val if base == "route" else pol_val
        feats_v = shared_feats("spectral", train, val)       # columns present in BOTH
        X_tr_v = train[feats_v].fillna(0).values
        X_val = val[feats_v].fillna(0).values
        if rel:                                              # OOD z-stats fit on FULL train
            stats, gm, gs = fit_zstats(train[base].values, train.N.values)
            y_tr_full = apply_zstats(train[base].values, train.N.values, stats, gm, gs)
            y_val = apply_zstats(val[base].values, val.N.values, stats, gm, gs)
        else:
            y_tr_full = train[base].values.astype(float)
            y_val = val[base].values.astype(float)

        print(f"  target={tok} (n_train={len(train)}, n_val={len(val)})")
        for m in MODELS:
            cfgs = _configs(m)
            cv = np.array([_cv_r2(m, p, X_tr, train, base, rel) for p in cfgs])   # TUNE on train
            bi = int(np.nanargmax(cv)); best = cfgs[bi]
            est = _pipe(m, best).fit(X_tr_v, y_tr_full)                           # TEST once
            r2 = metrics_full(y_val, est.predict(X_val))["R2"]
            rows.append(dict(device=TUNE_DEVICE, target=tok, model=m, R2=r2, cv_r2=float(cv[bi]),
                             params=str(best), n_train=len(train), n_val=len(val)))
            print(f"    {m:7} CV-best {str(best):42} cv_r2={cv[bi]:+.3f} -> OOD R2={r2:+.3f}")
    pd.DataFrame(rows).to_csv(TUNED_CSV, index=False)
    print(f"  wrote {TUNED_CSV}\n")


# ===========================================================================
# COMPUTE 2 -- log-MQT-9 LASSO table
#
# route = log1p(bare_routed_2q); MQT-9 ships RAW counts, so a linear model predicting a log target
# from raw counts extrapolates explosively cross-device. log1p-scaling the four count columns is the
# fair fix. LASSO only (this is a lasso-section discussion point). Reuses cross_device.py's cached
# per-device labels and the harness eval primitives (run_indist / run_external).
# ===========================================================================
REPS = ["mqt9_raw", "mqt9_log", "spectral"]
# (token, base, relative, arm)
TARGETS_LL = [("route", "route", False, "swap"), ("pol", "pol", False, "pol"),
              ("pol_z", "pol", True, "pol")]


def _ll_feats(df_train, df_ood, rep):
    cols = _rep_shared_cols("spectral" if rep == "spectral" else "mqt9", df_train, df_ood)
    tr, od = df_train[cols].copy(), df_ood[cols].copy()
    if rep == "mqt9_log":
        for c in COUNTS:
            if c in cols:
                tr[c] = np.log1p(tr[c].clip(lower=0)); od[c] = np.log1p(od[c].clip(lower=0))
    return tr.fillna(0).values, od.fillna(0).values


def _ll_feats_indist(df, rep):
    cols = [c for c in _rep_base_cols("spectral" if rep == "spectral" else "mqt9") if c in df.columns]
    M = df[cols].copy()
    if rep == "mqt9_log":
        for c in COUNTS:
            if c in M.columns:
                M[c] = np.log1p(M[c].clip(lower=0))
    return M.fillna(0).values


def compute_log_lasso(train):
    """raw vs log1p MQT-9 vs spectral, LASSO, in-dist + cross-device. Writes mqt9_log_lasso.csv."""
    print("== COMPUTE 2: log-MQT-9 LASSO (raw vs log vs spectral) ==")
    rows = []
    for rep in REPS:                                         # in-distribution (run_indist)
        X = _ll_feats_indist(train, rep)
        for tok, base, rel, _arm in TARGETS_LL:
            r2 = run_indist(train, X, base, "lasso", relative=rel)["R2"]
            rows.append(dict(scheme="indist", task=tok, rep=rep, R2=r2))
    for dev in DEVICES:                                      # cross-device (run_external)
        for tok, base, rel, arm in TARGETS_LL:
            path = LABELED / f"val_{arm}_{dev}.csv"
            if not path.exists():
                print(f"  [skip] {dev}/{tok}: no cached labels ({path.name})")
                continue
            ood = load_external(str(path))
            if base == "pol" and not ood.attrs.get("has_pol"):
                continue
            if base == "route" and not ood.attrs.get("has_route"):
                continue
            for rep in REPS:
                X_tr, X_ood = _ll_feats(train, ood, rep)
                pt, _ci, _pred = run_external(train, X_tr, ood, X_ood, base, "lasso",
                                              relative=rel, B=200)
                rows.append(dict(scheme=dev.replace("Fake", ""), task=tok, rep=rep, R2=pt["R2"]))
    pd.DataFrame(rows).to_csv(LOG_LASSO_CSV, index=False)
    print(f"  wrote {LOG_LASSO_CSV}\n")


# ===========================================================================
# RENDER helpers
# ===========================================================================
def load_cdv():
    """Cross-device results (spectral-17) from cross_device.py -- the source for T2 and T4.
    Returns a tidy df (device, target, model, R2, rep='spectral') or None if the CSV is absent."""
    path = CDV["spectral"]
    if not os.path.exists(path):
        print(f"  [skip T2/T4] missing {path} -- run cross_device.py --phase eval")
        return None
    d = pd.read_csv(path)[["device", "target", "model", "R2"]].copy()
    d["rep"] = "spectral"
    return d


def build_X(df, rep):
    cols = [c for c in _rep_base_cols("spectral" if rep == "spectral" else "mqt9") if c in df.columns]
    return df[cols].fillna(0).values


def _neg_cells(matrix):
    return {(i, j) for i, row in enumerate(matrix) for j, v in enumerate(row)
            if np.isfinite(v) and v < 0}


def _fmt_params(model, p):
    if model == "rf":
        return f"{p['n_estimators']}t, leaf{p['min_samples_leaf']}"
    if model == "histgb":
        return f"lr{p['learning_rate']}, d{p['max_depth']}, l2={p['l2_regularization']:g}"
    if model in ("ridge", "lasso"):
        return f"α={p['alpha']:.2g}"
    if model == "knn":
        return f"k{p['n_neighbors']}, {p['weights']}"
    return str(p)


EXPECTED_QIR = "0.47.0"   # qiskit-ibm-runtime that produced the frozen device-noise pol labels (requirements.txt)


def assert_qir_version(expected=EXPECTED_QIR):
    """Soft provenance guard for side scripts that pull LIVE device params from backend.target.

    The frozen pol labels were simulated under qiskit-ibm-runtime EXPECTED_QIR; the fake-backend
    snapshots are version-bound, so a different installed version may describe DIFFERENT calibration
    snapshots than the ones that produced the labels -- silently desyncing any figure that mixes live
    params with the cached labels (per-edge correlation, gate-error rescaling, Option A device noise).
    On mismatch this prints a loud stderr banner (NON-fatal -- reruns still work) and returns the
    installed version string for stamping into outputs. qiskit is imported lazily so report_artifacts
    itself stays import-clean without qiskit installed."""
    import qiskit_ibm_runtime as _qir
    v = _qir.__version__
    if v != expected:
        print("\n" + "!" * 78 +
              f"\n  PROVENANCE WARNING: installed qiskit-ibm-runtime {v} != pinned {expected}."
              "\n  Live backend.target params pulled here may NOT match the fake-backend snapshots"
              f"\n  that produced the frozen pol labels (generated under {expected}). Either re-pin"
              "\n  qiskit-ibm-runtime or regenerate the labels before trusting device-physics figures."
              "\n" + "!" * 78 + "\n", file=sys.stderr)
    return v


def render_table(col_labels, row_labels, cell_text, path, title, note="",
                 best_cells=None, neg_cells=None, row_w=None, label_w=None, dim_cells=None,
                 fontsize=10, row_in=0.30):
    best_cells = best_cells or set()
    neg_cells = neg_cells or set()
    dim_cells = dim_cells or set()
    nrows = len(row_labels)
    label_in = label_w if label_w else 1.45
    data_ins = list(row_w) if row_w else [1.7] * len(col_labels)
    fig_w = label_in + sum(data_ins)
    title_in = 0.24 + 0.20 * (title.count("\n") + 1)
    note_in = (0.16 * (note.count("\n") + 1) + 0.08) if note else 0.05  # height proportional to note line count
    fig_h = row_in * (nrows + 1) + title_in + note_in
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    fig.subplots_adjust(left=0.004, right=0.996,
                        top=1 - title_in / fig_h, bottom=note_in / fig_h)
    full_cols = [""] + list(col_labels)
    full_rows = [[rl] + list(row) for rl, row in zip(row_labels, cell_text)]
    widths = [label_in / fig_w] + [d / fig_w for d in data_ins]
    tbl = ax.table(cellText=full_rows, colLabels=full_cols,
                   cellLoc="center", rowLoc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        cell.set_width(widths[c])
        if r == 0:
            cell.set_facecolor("#40466e")
            if c > 0:
                cell.set_text_props(color="w", fontweight="bold")
        elif c == 0:
            cell.set_facecolor("#d9d9d9"); cell.set_text_props(fontweight="bold")
        else:
            rc = (r - 1, c - 1)
            if rc in dim_cells:                      # diagnostic column: neutral grey, not a contender
                cell.set_facecolor("#ededed")
            if rc in best_cells:
                # green only for a genuine win; best-but-negative is "least-bad", not a win -> light red
                cell.set_facecolor("#f4cccc" if rc in neg_cells else "#c6efce")
                cell.set_text_props(fontweight="bold")
            if rc in neg_cells:
                cell.set_text_props(color="#b00020")
    fig.suptitle(title, fontweight="bold", fontsize=11.5, va="center",
                 y=1 - title_in / (2 * fig_h))
    if note:
        fig.text(0.5, note_in / (2 * fig_h), note, ha="center", va="center",
                 fontsize=8, style="italic")
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ===========================================================================
# TABLES
# ===========================================================================
def table1(train):
    X = build_X(train, "spectral")
    recs = []
    for tok, base, rel in TARGETS3:
        for m in MODELS:
            s = run_indist(train, X, base, m, relative=rel)
            recs.append(dict(task=tok, model=m, R2=s["R2"], R2_sd=s["R2_sd"], Spearman=s["Spearman"]))
    d = pd.DataFrame(recs)
    tasks = ["route", "pol", "pol_z"]
    cell, mat = [], []
    for m in MODELS:
        line, mrow = [], []
        for t in tasks:
            r = d[(d.task == t) & (d.model == m)].iloc[0]
            line += [f"{r.R2:.3f} ± {r.R2_sd:.3f}", f"{r.Spearman:.3f}"]
            mrow.append(r.R2)
        cell.append(line); mat.append(mrow)
    arr = np.array(mat)
    best_cells = {(int(np.nanargmax(arr[:, j])), j * 2) for j in range(arr.shape[1])}
    cols = ["route R² (±sd)", "route ρ", "pol R² (±sd)", "pol ρ", "pol_z R² (±sd)", "pol_z ρ"]
    d.to_csv(os.path.join(OUT, "table1_indist_brisbane.csv"), index=False)
    render_table(cols, MODELS, cell, os.path.join(OUT, "table1_indist_brisbane.png"),
                 "Table 1 — In-distribution FakeBrisbane (5-fold GroupKFold CV, spectral-17)",
                 note=f"{ROUTE_DEF};  pol_z = within-N z-scored pol (train-fit).  ρ = Spearman;  "
                      "green = best model per target.",
                 best_cells=best_cells, row_w=[2.1, 1.0, 2.1, 1.0, 2.1, 1.0])


def table2(cdv):
    sp = cdv[(cdv.rep == "spectral") & (cdv.device == "FakeBrisbane")]
    rows = ["route", "pol", "pol_z"]
    cell, mat = [], []
    for t in rows:
        vals = []
        for m in MODELS:
            s = sp[(sp.target == t) & (sp.model == m)]
            vals.append(float(s.R2.values[0]) if len(s) else np.nan)
        cell.append([(f"{v:.3f}" if np.isfinite(v) else "—") for v in vals]); mat.append(vals)
    best_cells = {(i, int(np.nanargmax(r))) for i, r in enumerate(mat)}
    render_table(MODELS, rows, cell, os.path.join(OUT, "table2_brisbane_unseen.png"),
                 "Table 2 — Device held constant, unseen circuits\n(train Brisbane set → Brisbane validation corpus, spectral-17, untuned)",
                 note=f"{ROUTE_DEF};  pol_z = within-N z-scored pol.  R²; green = best model per target.",
                 best_cells=best_cells, neg_cells=_neg_cells(mat))
    pd.DataFrame(cell, index=rows, columns=MODELS).to_csv(os.path.join(OUT, "table2_brisbane_unseen.csv"))


def table3():
    b = pd.read_csv(TUNED_CSV)
    b = b[b.device == "FakeBrisbane"]
    tgts = ["route", "pol", "pol_z"]
    cell, r2 = [], []
    for m in MODELS:
        line, mrow = [], []
        for t in tgts:
            r = b[(b.target == t) & (b.model == m)].iloc[0]
            line += [_fmt_params(m, ast.literal_eval(r.params)), f"{r.R2:.3f}"]
            mrow.append(r.R2)
        cell.append(line); r2.append(mrow)
    arr = np.array(r2)
    best_cells = {(int(np.argmax(arr[:, j])), 2 * j + 1) for j in range(len(tgts))}
    cols = ["route params", "route R²", "pol params", "pol R²", "pol_z params", "pol_z R²"]
    render_table(cols, MODELS, cell, os.path.join(OUT, "table3_validation_tuned.png"),
                 "Table 3 — Hyperparameters tuned by in-distribution CV, tested once on the\n"
                 "344-circuit validation corpus (FakeBrisbane)",
                 note=f"{ROUTE_DEF};  pol_z = within-N z-scored pol.  α/config selected by 5-fold "
                      "GroupKFold CV on TRAIN (Table-1 split).  green = best R².",
                 best_cells=best_cells, row_w=[2.3, 1.0, 2.3, 1.0, 2.3, 1.0])
    b.to_csv(os.path.join(OUT, "table3_validation_tuned.csv"), index=False)


def table5():
    """Table 5 -- LASSO feature representation: MQT-9 (log1p counts) vs spectral-17 on route, pol and
    pol_z, in-distribution and cross-device. All numbers from compute_log_lasso (harness run_indist /
    run_external) -> mqt9_log_lasso.csv. Raw unscaled counts are NOT shown -- they collapse on the log
    route target cross-device, but that motivation for log-scaling is carried in the prose, not as a
    column of catastrophic values. spectral-17 is the robust representation; pol_z exposes the same
    Berlin/Boston device-shift failure that T6 analyzes."""
    d = pd.read_csv(LOG_LASSO_CSV)
    reps = ["mqt9_log", "spectral"]
    order = [s for s in ["indist", "Brisbane", "Berlin", "Boston", "Sherbrooke", "Torino"]
             if s in d.scheme.unique()]
    rlbl = [("Brisbane*" if s == "Brisbane" else s) for s in order]
    tasks = ["route", "pol", "pol_z"]
    cols = ["route\nMQT-9", "route\nspec", "pol\nMQT-9", "pol\nspec", "pol_z\nMQT-9", "pol_z\nspec"]
    cell, mat = [], []
    for s in order:
        r = []
        for t in tasks:
            for rep in reps:                         # reps = [mqt9_log, spectral]
                sub = d[(d.scheme == s) & (d.task == t) & (d.rep == rep)]
                r.append(float(sub.R2.values[0]) if len(sub) else np.nan)
        cell.append([("—" if not np.isfinite(v) else (f"{v:.3f}" if abs(v) < 100 else f"{v:.0f}"))
                     for v in r]); mat.append(r)
    arr = np.array(mat)
    # best rep per task = the larger of {MQT-9 log1p, spectral} (the 2 cols of each target block) --
    # but only colour it when it WINS by more than TIE_EPS; within that, the two reps tie and neither
    # is coloured. TIE_EPS ≈ the route CV fold-sd (~0.002), so noise-level gaps don't read as wins
    # (e.g. in-dist route 0.940 vs 0.937 = Δ0.003 is a tie; the next real gap is 0.072).
    TIE_EPS = 0.005
    best_cells = set()
    for i in range(len(order)):
        for k in range(len(tasks)):
            pair = arr[i, 2 * k:2 * k + 2]
            if np.any(np.isfinite(pair)):
                bi = int(np.nanargmax(pair))
                other = pair[1 - bi]
                if (not np.isfinite(other)) or (pair[bi] - other > TIE_EPS):
                    best_cells.add((i, 2 * k + bi))
    render_table(cols, rlbl, cell, os.path.join(OUT, "table5_lasso_featurerep.png"),
                 "Table 5 — LASSO feature representation: MQT-9 (log1p) vs spectral-17",
                 note=f"{ROUTE_DEF};  pol_z = within-N z-scored pol.  LASSO;  MQT-9 counts log1p-scaled.  "
                      "green = best rep per task;  * = Brisbane self-transfer.",
                 best_cells=best_cells, neg_cells=_neg_cells(arr))
    d[d.task.isin(tasks)].to_csv(os.path.join(OUT, "table5_lasso_featurerep.csv"), index=False)


# ===========================================================================
# REPORT FIGURES (folded from sidetests/) -- each is compute+render in one function. The harness (H)
# does every fit/metric/CV/OOD; these functions only orchestrate and draw.
# ===========================================================================
def fig_headline_crossdevice(train):
    """Table 4 (bar chart) -- cross-device generalization by model family, 5-fold CV. Each (family,
    device) point is a 5-fold GroupKFold mean ± sd: in-dist = held-out fold; cross-device = fit the
    training fold, predict that device's full validation corpus (no tuning on the test device).
    Folded from sidetests/headline_crossdevice_cv.py. Writes headline_crossdevice_cv.png."""
    from matplotlib import cm
    OOD = ["FakeSherbrooke", "FakeTorino", "FakeBerlin", "FakeBoston"]
    FAMILIES = ["lasso", "ridge", "rf", "histgb", "knn"]
    XKEYS = ["Brisbane", "Sherbrooke", "Torino", "Berlin", "Boston"]
    XLABS = ["Brisbane\n(in-dist)", "Sherbrooke", "Torino", "Berlin", "Boston"]
    TARGETS = [("route", "route", False, "swap"), ("pol", "pol", False, "pol"),
               ("pol_z", "pol", True, "pol")]

    def eval_target(base, rel, arm, folds):
        hattr = "has_route" if arm == "swap" else "has_pol"
        oods = {}
        for d in OOD:
            p = LABELED / f"val_{arm}_{d}.csv"
            if p.exists():
                o = load_external(str(p))
                if o.attrs.get(hattr):
                    oods[d] = o
        feats = shared_feats("spectral", train, *oods.values())
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
                    stats, gm, gs = fit_zstats(yb[tr], N[tr])
                    ytr = apply_zstats(yb[tr], N[tr], stats, gm, gs)
                    yte = apply_zstats(yb[te], N[te], stats, gm, gs)
                else:
                    ytr, yte = yb[tr], yb[te]
                pipe = H._fit_full(fam, X[tr], ytr, grp[tr])
                per["Brisbane"].append(metrics_full(yte, pipe.predict(X[te]))["R2"])
                for d, o in oods.items():
                    yo = (apply_zstats(ybod[d], Nod[d], stats, gm, gs) if rel else ybod[d])
                    per[d.replace("Fake", "")].append(metrics_full(yo, pipe.predict(Xod[d]))["R2"])
            for k, v in per.items():
                res[fam][k] = (float(np.mean(v)) if v else np.nan, float(np.std(v)) if v else np.nan)
        return res

    folds = list(GroupKFold(5).split(np.arange(len(train)), groups=train.grp.values))
    all_res, rows = {}, []
    for tok, base, rel, arm in TARGETS:
        print(f"== headline cross-device CV: {tok} ==")
        res = eval_target(base, rel, arm, folds)
        all_res[tok] = res
        for fam in FAMILIES:
            for k in XKEYS:
                m, s = res[fam][k]
                rows.append(dict(target=tok, family=fam, device=k,
                                 scheme=("in-dist" if k == "Brisbane" else "cross-device"),
                                 R2_mean=m, R2_sd=s))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "headline_crossdevice_cv.csv"), index=False)

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
    p = os.path.join(OUT, "headline_crossdevice_cv.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"  wrote {p}")


def table_ood_importance(train, target="route"):
    """Table 6 -- LASSO feature importance & CROSS-DEVICE OOD ablation (spectral-17). Coefficients are
    the train-fit lasso (device-independent); the ablation columns are the drop in OOD R²/ρ (train
    Brisbane → test the 4 OOD devices, mean) when each feature is removed. Folded from
    sidetests/ood_importance.py. Writes table_lasso_importance_ood_<target>.png."""
    import diagnostics                                # harness leaf module (model_importances)
    OOD = ["FakeSherbrooke", "FakeTorino", "FakeBerlin", "FakeBoston"]
    arm = {"route": "swap", "pol": "pol", "pol_z": "pol"}[target]
    base = {"route": "route", "pol": "pol", "pol_z": "pol"}[target]
    rel = target == "pol_z"
    hasattr_ = {"swap": "has_route", "pol": "has_pol"}[arm]

    def ytr_yood(o):
        if rel:
            stats, gm, gs = fit_zstats(train[base].values, train.N.values)
            return (apply_zstats(train[base].values, train.N.values, stats, gm, gs),
                    apply_zstats(o[base].values, o.N.values, stats, gm, gs))
        return train[base].values.astype(float), o[base].values.astype(float)

    oods = {}
    for d in OOD:
        p = LABELED / f"val_{arm}_{d}.csv"
        if p.exists():
            o = load_external(str(p))
            if o.attrs.get(hasattr_):
                oods[d] = o
    feats = shared_feats("spectral", train, *oods.values())
    X = train[feats].fillna(0).values
    ytr = ytr_yood(next(iter(oods.values())))[0]
    yoods = {d: ytr_yood(o)[1] for d, o in oods.items()}

    def ood_mean(pipe, cols_idx):
        r2, rho = [], []
        for d, o in oods.items():
            Xo = o[[feats[i] for i in cols_idx]].fillna(0).values
            mm = metrics_full(yoods[d], pipe.predict(Xo))
            r2.append(mm["R2"]); rho.append(mm["Spearman"])
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
    df.to_csv(os.path.join(RESULTS, f"lasso_importance_ood_{target}.csv"), index=False)

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
    render_table(cols, rlbl, cell, os.path.join(OUT, f"table_lasso_importance_ood_{target}.png"),
                 f"Feature importance & ablation — LASSO ({target}, spectral-17), CROSS-DEVICE OOD",
                 note=note, dim_cells=dim, row_w=[1.7, 2.2, 2.1], label_w=2.9, fontsize=11, row_in=0.24)
    print(f"  [{target}] baseline OOD R²={base_r2:+.4f} ρ={base_rho:+.4f}; L1 keeps "
          f"{int(df.selected.sum())}/{len(df)} -> table_lasso_importance_ood_{target}.png")


def fig_ablation_by_device(train, target="route"):
    """Table 7 -- leave-one-feature-out OOD ablation per DEVICE (route, spectral-17), as a heatmap.
    The fitted coefficients are device-independent (one trained lasso); only how much each feature
    matters for predicting each device varies. Folded from sidetests/sensitivity_importance.py
    (the ablation-across-devices section only). Writes sensitivity_ablation_by_device_route.png."""
    OOD = ["FakeSherbrooke", "FakeTorino", "FakeBerlin", "FakeBoston"]
    oods = {}
    for d in OOD:
        p = LABELED / f"val_swap_{d}.csv"
        if p.exists():
            oods[d] = load_external(str(p))
    feats = shared_feats("spectral", train, *oods.values())
    Xtr = train[feats].fillna(0).values
    y = train["route"].values
    Xods = {d: o[feats].fillna(0).values for d, o in oods.items()}
    yods = {d: o["route"].values for d, o in oods.items()}

    tuned_pipe = H._fit_full("lasso", Xtr, y, train.grp.values)
    tuned_coef = np.ravel(tuned_pipe.named_steps["model"].best_estimator_.coef_)
    order = np.argsort(np.abs(tuned_coef))[::-1]              # sort by operating-point importance

    full_r2 = {}
    for d in OOD:
        full_r2[d] = metrics_full(yods[d], tuned_pipe.predict(Xods[d]))["R2"]
    A = np.zeros((len(feats), len(OOD)))
    for i in range(len(feats)):
        keep = [k for k in range(len(feats)) if k != i]
        pipe_i = H._fit_full("lasso", Xtr[:, keep], y, train.grp.values)
        for jd, d in enumerate(OOD):
            r2_i = metrics_full(yods[d], pipe_i.predict(Xods[d][:, keep]))["R2"]
            A[i, jd] = full_r2[d] - r2_i                       # ΔR² = drop in device-d OOD R²
    dev_short = [d.replace("Fake", "") for d in OOD]
    adf = pd.DataFrame(A, index=feats, columns=dev_short)
    adf["mean"] = adf.mean(axis=1)
    adf = adf.reindex([feats[i] for i in order])
    adf.to_csv(os.path.join(RESULTS, "sensitivity_ablation_by_device_route.csv"))

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
    p2 = os.path.join(OUT, "sensitivity_ablation_by_device_route.png")
    fig2.savefig(p2, dpi=150); plt.close(fig2)
    print(f"  wrote {p2}")


def fig_why_berlin_boston(train):
    """Table 8 (device_error_table.png) + Figure 5 (why_berlin_boston_fail.png) -- WHY Berlin/Boston
    have terrible pol OOD R²: they are cleaner devices, so a Brisbane-trained model under-predicts
    their (higher) polarization -> R² goes strongly negative though rank is preserved; the gate-error
    power-rescale P_D = P_B**(e_D/e_B) recovers the level. Folded from sidetests/why_berlin_boston_fail.py."""
    assert_qir_version()   # live backend.target params below must match the label-gen snapshots
    DEVICES_L = ["FakeBrisbane", "FakeBerlin", "FakeBoston"]
    SHORT = ["Brisbane\n(noisy, train)", "Berlin\n(3.5× cleaner)", "Boston\n(6.1× cleaner)"]

    def median_2q_error(name):
        from qiskit_ibm_runtime import fake_provider as fp
        bk = getattr(fp, name)()
        t = bk.target
        errs = [p.error for g in ("ecr", "cz") if g in t for p in t[g].values()
                if p is not None and p.error is not None]
        return float(np.median(errs))

    e = {d: median_2q_error(d) for d in DEVICES_L}
    eB = e["FakeBrisbane"]
    print("documented median 2-qubit gate error (live from backend.target):")
    for d in DEVICES_L:
        print(f"  {d:13s} e={e[d]:.5f}   ({eB/e[d]:.1f}× cleaner than Brisbane)" if d != "FakeBrisbane"
              else f"  {d:13s} e={e[d]:.5f}   (training device)")

    val = {d: load_external(str(LABELED / f"val_pol_{d}.csv")) for d in DEVICES_L}
    feats = shared_feats("spectral", train, *val.values())
    pipe = H._fit_full("lasso", train[feats].fillna(0).values, train["pol"].values, train.grp.values)

    base = val["FakeBrisbane"].reset_index(drop=True)
    pred_B = np.clip(pipe.predict(base[feats].fillna(0).values), 1e-6, 1.0)   # model's prediction
    actual = {d: val[d].set_index("file")["polarization"] for d in DEVICES_L}

    aB_all = actual["FakeBrisbane"]; aBe_all = actual["FakeBerlin"]; aBo_all = actual["FakeBoston"]
    mono = sum(1 for f in base["file"] if aB_all[f] < aBe_all[f] < aBo_all[f])
    ntot = len(base); mono_pct = 100.0 * mono / ntot
    print(f"\nCORPUS EVIDENCE: {mono}/{ntot} ({mono_pct:.0f}%) circuits rise monotonically "
          f"Brisbane<Berlin<Boston.")

    g2 = base["num_2q_gates"].values
    def is_mono(f): return aB_all[f] < aBe_all[f] < aBo_all[f]
    pick, used_algos = [], set()
    for target in (4, 10, 18):
        cand = sorted(range(len(base)), key=lambda i: (abs(g2[i] - target), base.loc[i, "file"]))
        for i in cand:
            f = base.loc[i, "file"]; a = base.loc[i, "algo"]
            if a not in used_algos and aB_all[f] > 0.05 and is_mono(f):
                pick.append(i); used_algos.add(a); break

    rows = []
    for idx in pick:
        f = base.loc[idx, "file"]
        aB, aBe, aBo = actual["FakeBrisbane"][f], actual["FakeBerlin"][f], actual["FakeBoston"][f]
        pB = pred_B[idx]
        fix = {d: float(np.clip(pB ** (e[d] / eB), 0, 1)) for d in DEVICES_L}
        rows.append(dict(file=f.split("/")[-1], algo=base.loc[idx, "algo"],
                         nq=int(base.loc[idx, "num_qubits"]), g2=int(base.loc[idx, "num_2q_gates"]),
                         actual_Brisbane=aB, actual_Berlin=aBe, actual_Boston=aBo,
                         pred_raw=pB, pred_fix_Berlin=fix["FakeBerlin"], pred_fix_Boston=fix["FakeBoston"]))
    df = pd.DataFrame(rows)

    def r2(y, p): return H.metrics_full(np.asarray(y), np.asarray(p))["R2"]
    def rho(y, p): return H.metrics_full(np.asarray(y), np.asarray(p))["Spearman"]
    print("\nfull-corpus pol R² (Brisbane-trained lasso), raw vs gate-error-rescaled:")
    yidx = base["file"].values
    for d in DEVICES_L:
        ad = actual[d].reindex(yidx).values
        praw = pred_B
        pfix = np.clip(pred_B ** (e[d] / eB), 0, 1)
        tag = "(train)" if d == "FakeBrisbane" else ""
        print(f"  {d:13s} raw R²={r2(ad, praw):+7.3f}  ->  rescaled R²={r2(ad, pfix):+7.3f}   "
              f"(rank ρ={rho(ad, praw):+.3f}, unchanged by rescale) {tag}")
    df.to_csv(os.path.join(RESULTS, "why_berlin_boston_fail.csv"), index=False)

    # ---- Table 8: small device-error table ----
    drows, dcell = [], []
    for d in DEVICES_L:
        pm = float(actual[d].mean())
        drows.append(d.replace("Fake", ""))
        dcell.append([f"{e[d]:.5f}", ("1.0× (train)" if d == "FakeBrisbane" else f"{eB/e[d]:.1f}× cleaner"),
                      f"{pm:.3f}"])
    render_table(["median 2q gate error", "vs Brisbane", "mean polarization"], drows, dcell,
                 os.path.join(OUT, "device_error_table.png"),
                 "Device gate-error & polarization — cleaner devices, higher polarization",
                 note="median 2-qubit gate error pulled live from backend.target;  "
                      "mean polarization over the 219-circuit validation corpus.",
                 row_w=[2.4, 1.9, 2.0], label_w=1.5, fontsize=12, row_in=0.34)

    # ---- Figure 5: 2-panel (3 circuits under-prediction + fix | full-corpus R² before/after) ----
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
    axL.annotate("model is device-blind: prediction (—) is flat across panels;\n"
                 "gap to each bar = noise error the rescaling (◇) corrects",
                 xy=(2 - bw, df.iloc[0].pred_raw), xycoords="data",
                 xytext=(1.30, 1.20), fontsize=7.3, color="#333", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#333", lw=1), zorder=6)

    praw = pred_B
    raw_r2 = [H.metrics_full(actual[d].reindex(yidx).values, praw)["R2"] for d in DEVICES_L]
    fix_r2 = [H.metrics_full(actual[d].reindex(yidx).values,
                             np.clip(praw ** (e[d] / eB), 0, 1))["R2"] for d in DEVICES_L]
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
    p = os.path.join(OUT, "why_berlin_boston_fail.png")
    fig.savefig(p, dpi=135); plt.close(fig)
    print(f"  wrote {p} + device_error_table.png")


def fig_sherbrooke_degeneration(train, model="knn"):
    """Figure 6 -- WHY the KNN-vs-Lasso comparison on Sherbrooke polarization is CONDITIONAL on the
    target. KNN emits only averages of training labels, so its predictions are clipped to the training
    range and never drop below 0; Lasso extrapolates and predicts pol<0 -- physically impossible.
    On RAW pol (hard physical floor at 0) that clipping HELPS: KNN avoids the impossible region and
    wins (R2 ~+.63 vs Lasso ~+.52). On z-scored pol_z there is NO floor (negative z is legitimate, no
    red band), so the clipping buys nothing and KNN's inability to reach the extremes HURTS -- KNN
    LOSES (R2 ~-.63 vs ~-.36; note the flat KNN band that under-predicts true z of 6-9). So the
    advantage is conditional on a bounded target with a hard floor, NOT a general property.
    Two DISTINCT ceilings on the raw-pol panel, kept separate on purpose: the training-LABEL max
    ~0.90, and KNN's AVERAGED prediction ceiling ~0.75 -- a k-neighbour mean can't reach the lone
    0.90 label, so KNN caps below the true training max. Two panels (raw pol, pol_z), KNN vs Lasso
    overlaid. `model` is accepted for call-compatibility; the figure always contrasts KNN vs Lasso."""
    dev = "FakeSherbrooke"
    o = load_external(str(LABELED / f"val_pol_{dev}.csv"))
    feats = shared_feats("spectral", train, o)
    Xtr = train[feats].fillna(0).values
    Xo = o[feats].fillna(0).values
    pol_max = float(train["pol"].values.astype(float).max())          # training-LABEL max ~0.90 (NOT KNN's ~0.75 pred cap)
    stats, gm, gs = fit_zstats(train["pol"].values, train.N.values)
    ytr_z = apply_zstats(train["pol"].values, train.N.values, stats, gm, gs)
    specs = [
        ("pol", False, o["pol"].values.astype(float), 0.0, pol_max, "polarization"),
        ("pol_z", True, apply_zstats(o["pol"].values, o.N.values, stats, gm, gs),
         float(ytr_z.min()), float(ytr_z.max()), "polarization z-score"),
    ]
    COL = {"knn": "#4C72B0", "lasso": "#DD8452"}
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8))
    for ax, (tok, rel, ytrue, tlo, thi, ylab) in zip(axes, specs):
        preds = {}
        for m in ("knn", "lasso"):
            _pt, _ci, pred = run_external(train, Xtr, o, Xo, "pol", m, relative=rel, B=0)
            preds[m] = (pred, _pt["R2"], _pt["Spearman"])
        axlo, axhi = (-0.5, 1.0) if tok == "pol" else (-4.0, 9.5)     # fixed axes for KNN-vs-Lasso comparison
        if tok == "pol":                                              # floor + two DISTINCT ceilings (label-max vs KNN cap)
            ax.axhspan(axlo, 0.0, color="#f4cccc", alpha=0.45, zorder=0)
            ax.axhline(0.0, color="#b00020", lw=1.0, zorder=1)
            ax.axhline(thi, color="#444", lw=1.0, ls=":", zorder=1)
            ax.text(axhi, axlo, " predicted pol < 0 : impossible ", color="#b00020", fontsize=8,
                    va="bottom", ha="right")
            ax.text(axhi, thi, f" training-label max {thi:.2f} ", color="#444", fontsize=8,
                    va="bottom", ha="right")
            knn_cap = float(preds["knn"][0].max())                    # KNN's averaged prediction ceiling (~0.75 != 0.90)
            ax.axhline(knn_cap, color=COL["knn"], lw=1.1, ls="--", alpha=.85, zorder=1)
            ax.text(axhi, knn_cap, f" KNN prediction ceiling {knn_cap:.2f} (averaging artifact) ",
                    color=COL["knn"], fontsize=8, va="top", ha="right")
        else:                                                        # KNN bounded to training z-range
            ax.axhline(tlo, color="#444", lw=1.0, ls=":", zorder=1)
            ax.axhline(thi, color="#444", lw=1.0, ls=":", zorder=1)
            ax.text(axhi, thi, " training range ", color="#444", fontsize=8, va="bottom", ha="right")
        ax.plot([axlo, axhi], [axlo, axhi], "k--", lw=1, alpha=.6, zorder=1, label="y = x")
        for m in ("knn", "lasso"):
            pred, r2, rho = preds[m]
            ax.scatter(ytrue, pred, s=22, c=COL[m], alpha=.6, edgecolor="white", linewidth=.3,
                       zorder=3, label=f"{m.upper()}   R2={r2:+.2f}  rho={rho:+.2f}")
        ax.set_xlim(axlo, axhi); ax.set_ylim(axlo, axhi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"true {ylab}"); ax.set_title(f"Sherbrooke -- {tok}", fontweight="bold", fontsize=11)
        ax.legend(fontsize=8, loc="upper left", framealpha=.92); ax.grid(alpha=.2)
    axes[0].set_ylabel("predicted")
    fig.suptitle("Figure 6 -- Sherbrooke: KNN vs Lasso", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "sherbrooke_degeneration.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", p)


def fig_per_edge_error(dead=0.99):
    """Appendix (Fig 6 companion) -- measuring numerically that Sherbrooke is a DIFFERENT device from
    Brisbane despite a matched MEDIAN 2q gate error. Both are Eagle r3 / 127q / ecr with the IDENTICAL
    144-edge coupling map, but for the SAME physical edge the errors are essentially uncorrelated
    (per-edge Pearson ~0, Spearman ~0.14): same wiring, same median, different geography of which edges
    are bad -> the device-level origin of Sherbrooke's within-N rank reordering on pol_z. Pulls live
    backend.target params (guarded against a version drift from the label-gen snapshots). Folded from
    sidetests/per_edge_error_correlation.py."""
    qir_version = assert_qir_version()
    from scipy.stats import pearsonr, spearmanr
    import qiskit_ibm_runtime.fake_provider as fp

    def dev(name):
        bk = getattr(fp, name)(); t = bk.target
        gate = "ecr" if "ecr" in t else "cz"
        edges = {tuple(sorted(k)): float(p.error) for k, p in t[gate].items()
                 if p is not None and p.error is not None}
        props = getattr(bk, "_props_dict", None) or {}
        return dict(edges=edges, gate=gate, nq=t.num_qubits,
                    real=props.get("backend_name", name), snap=str(props.get("last_update_date", "?"))[:10])

    B, S = dev("FakeBrisbane"), dev("FakeSherbrooke")
    assert B["gate"] == S["gate"] and B["nq"] == S["nq"] and set(B["edges"]) == set(S["edges"]), \
        "Brisbane/Sherbrooke must share architecture + coupling map for a per-edge comparison"
    shared = [e for e in B["edges"] if B["edges"][e] < dead and S["edges"][e] < dead]
    xb = np.array([B["edges"][e] for e in shared]); xs = np.array([S["edges"][e] for e in shared])
    pear, spear = float(pearsonr(xb, xs)[0]), float(spearmanr(xb, xs)[0])
    bmed = float(np.median([e for e in B["edges"].values() if e < dead]))
    smed = float(np.median([e for e in S["edges"].values() if e < dead]))

    pd.DataFrame([
        dict(device=B["real"], gate=B["gate"], n_qubits=B["nq"], n_edges=len(B["edges"]),
             median_2q=bmed, snapshot=B["snap"]),
        dict(device=S["real"], gate=S["gate"], n_qubits=S["nq"], n_edges=len(S["edges"]),
             median_2q=smed, snapshot=S["snap"]),
    ]).to_csv(os.path.join(OUT, "per_edge_error_correlation.csv"), index=False)

    col_labels = [f"{B['real']}\n(train)", f"{S['real']}\n(OOD)"]
    row_labels = ["native 2q gate", "qubits", "coupling-map edges",
                  "median 2q error", "per-edge error corr.\n(same physical edge)"]
    cells = [[B["gate"], S["gate"]], [str(B["nq"]), str(S["nq"])],
             [str(len(B["edges"])), str(len(S["edges"]))],
             [f"{bmed:.2e}", f"{smed:.2e}"], ["(reference)", f"r={pear:+.2f},  rho={spear:+.2f}"]]
    note = (f"qiskit-ibm-runtime {qir_version};  {B['real']} / {S['real']} calibration snapshots {B['snap']}.\n"
            f"Per-edge errors over the {len(shared)} shared live edges (dead edges err>={dead} excluded).")
    render_table(col_labels, row_labels, cells, os.path.join(OUT, "per_edge_error_correlation.png"),
                 "Appendix -- Brisbane vs Sherbrooke: 2q gate-error comparison",
                 note=note, best_cells={(4, 1)}, neg_cells={(4, 1)},
                 row_w=[2.1, 2.1], label_w=2.7)
    print("  wrote", os.path.join(OUT, "per_edge_error_correlation.png"))


def fig_mqt9_raw_vs_log():
    """Figure 7 -- raw (unscaled) vs log1p MQT-9, in- and out-of-distribution. Reads the
    already-computed mqt9_log_lasso.csv (LOG_LASSO_CSV); no recompute. Folded from
    sidetests/mqt9_raw_vs_log.py. Writes mqt9_raw_vs_log.png."""
    TASKS = ["route", "pol", "pol_z"]
    OOD = ["Berlin", "Boston", "Sherbrooke", "Torino"]
    REPS_L = [("mqt9_raw", "MQT-9 raw (pre-log)", "#C44E52"),
              ("mqt9_log", "MQT-9 log1p (post-log)", "#DD8452"),
              ("spectral", "spectral-17 (ref)", "#4C72B0")]
    d = pd.read_csv(LOG_LASSO_CSV)
    indist = {(t, r): float(d[(d.scheme == "indist") & (d.task == t) & (d.rep == r)].R2.values[0])
              for t in TASKS for r, *_ in REPS_L}
    ood = {(t, r): float(d[(d.scheme.isin(OOD)) & (d.task == t) & (d.rep == r)].R2.mean())
           for t in TASKS for r, *_ in REPS_L}

    xs = np.arange(len(TASKS)); nb = len(REPS_L); bw = 0.8 / nb
    fig, (axI, axO) = plt.subplots(1, 2, figsize=(13.4, 5.2))

    def grouped(ax, data, ylo, yhi, annotate_clipped):
        for k, (rep, lab, col) in enumerate(REPS_L):
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
    p = os.path.join(OUT, "mqt9_raw_vs_log.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"  wrote {p}")


def fig_feature_correlation(train):
    """Spearman rank-correlation heatmap among the spectral-17 features (training set) -- the
    intra-feature collinearity behind the importance story. Features reordered by hierarchical
    clustering on 1−|ρ|. Folded from sidetests/feature_correlation.py. Writes
    feature_correlation_spearman.png."""
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    feats = [c for c in FEATURE_SETS["spectral"] if c in train.columns]
    C = train[feats].corr(method="spearman")
    D = 1.0 - np.abs(C.values); np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    idx = list(leaves_list(Z))
    Cc = C.iloc[idx, idx]
    lab = list(Cc.columns)
    Cc.to_csv(os.path.join(RESULTS, "feature_correlation_spearman.csv"))

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
    p = os.path.join(OUT, "feature_correlation_spearman.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  wrote {p}")


def fig_ridge_vs_lasso_importance(train, target="route"):
    """APPENDIX -- is time_to_connected's large importance an L1 (Lasso) artifact, or a real signal?
    Compare per-SD standardized coefficients under LASSO (L1) and RIDGE (L2) on the SAME spectral-17
    fit (harness-tuned alpha each). A feature large under BOTH regularizers is not a regularization-
    method artifact; one large only under L1 is. Writes ridge_vs_lasso_importance.png + .csv."""
    import diagnostics                                       # harness leaf module (model_importances)
    feats = _rep_shared_cols("spectral", train)
    X = train[feats].fillna(0).values
    y = resolve_target(train, target, np.ones(len(train), bool), False)
    coef = {}
    for m in ("lasso", "ridge"):
        pipe = H._fit_full(m, X, y, train.grp.values)
        _l, _n, c = diagnostics.model_importances(pipe, feats)   # per-SD standardized coefficients
        coef[m] = np.asarray(c, float)
    order = np.argsort(np.maximum(np.abs(coef["lasso"]), np.abs(coef["ridge"])))[::-1]
    f = [feats[i] for i in order]
    cl = np.array([abs(coef["lasso"][i]) for i in order])
    cr = np.array([abs(coef["ridge"][i]) for i in order])
    pd.DataFrame({"feature": f, "lasso_abscoef": cl, "ridge_abscoef": cr}).to_csv(
        os.path.join(RESULTS, f"ridge_vs_lasso_importance_{target}.csv"), index=False)
    yy = np.arange(len(f))[::-1]; h = 0.40
    fig, ax = plt.subplots(figsize=(10.0, 7.2))
    ax.barh(yy + h / 2, cl, height=h, color="#4C72B0", label="LASSO (L1)")
    ax.barh(yy - h / 2, cr, height=h, color="#DD8452", label="Ridge (L2)")
    ylabels = [("» " + name if name == "time_to_connected" else name) for name in f]
    ax.set_yticks(yy); ax.set_yticklabels(ylabels, fontsize=8.5)
    if "time_to_connected" in f:
        ti = f.index("time_to_connected")
        ax.text(max(cl[ti], cr[ti]) + 0.004, yy[ti], f"L1={cl[ti]:.3f}  L2={cr[ti]:.3f}",
                va="center", fontsize=8, color="#222", fontweight="bold")
    ax.set_xlabel("|standardized coefficient|  (per-SD effect)")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=.25, axis="x")
    ax.set_title(f"Lasso vs Ridge — |standardized coefficient| ({target}, spectral-17, in-distribution)",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, "ridge_vs_lasso_importance.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  wrote {p}")


def preflight():
    """Verify the inputs the five tables need; print exactly what to run if anything's missing.
    Returns (ok, have_cdv). ok=False means a required input (training CSVs / validation labels)
    is absent -- the caller should stop and run the full analysis first."""
    print("preflight — checking the full analysis has been run:")
    miss_train = [str(f) for f in (DEFAULT_SWAP_CSV, DEFAULT_POL_CSV) if not os.path.exists(f)]
    all_dev = [TRAIN_DEV] + OOD_DEVS
    need = [LABELED / f"val_{arm}_{d}.csv" for d in all_dev for arm in ("swap", "pol")]
    miss_labels = [p.name for p in need if not p.exists()]
    have_cdv = os.path.exists(CDV["spectral"])
    print(f"  training CSVs        : {'OK' if not miss_train else 'MISSING ' + str(miss_train)}")
    print(f"  validation labels    : {'OK' if not miss_labels else f'MISSING {len(miss_labels)} ({miss_labels[:3]}...)'}")
    print(f"  cross_device results : {'OK' if have_cdv else 'MISSING -> T2 & T4 will be skipped'}")
    if miss_labels:
        print("  -> run:  python src/supervised/cross_device.py --phase label "
              "--devices " + " ".join(all_dev) + " --targets route pol")
    if not have_cdv:
        print("  -> run:  python src/supervised/cross_device.py --phase eval "
              "--devices " + " ".join(all_dev) + " --targets route pol pol_z")
    ok = not miss_train and not miss_labels
    if not ok:
        print("  FATAL: required inputs missing — run the full analysis first (commands above).")
    return ok, have_cdv


# ===========================================================================
def main():
    ok, have_cdv = preflight()
    if not ok:
        sys.exit(1)

    print("\nloading train set...")
    train = load_data(DEFAULT_SWAP_CSV, DEFAULT_POL_CSV)

    # --- compute (write the CSVs the report tables read; nothing hard-coded) ---
    compute_indist_tuned(train)                      # -> T3
    compute_log_lasso(train)                         # -> T5 (+ feeds Fig 7)

    cdv = load_cdv() if have_cdv else None

    # --- render the REPORT set ---
    print("rendering report tables + figures...")
    table1(train)                                    # T1  in-distribution baseline/context
    if cdv is not None:
        table2(cdv)                                  # T2  device held constant, unseen circuits
    else:
        print("  [skip T2: cross_device_results.csv absent]")
    table3()                                         # T3  hyperparameters tuned in-dist
    fig_headline_crossdevice(train)                  # T4  cross-device bar chart
    table5()                                         # T5  LASSO feature representation
    table_ood_importance(train, "route")             # T6  cross-device OOD importance & ablation
    fig_ablation_by_device(train, "route")           # T7  per-device OOD ablation heatmap
    fig_why_berlin_boston(train)                     # T8 (device_error_table) + Fig 5
    fig_sherbrooke_degeneration(train, "knn")        # Fig 6  pol_z degeneration
    fig_per_edge_error()                             # Appendix  Brisbane-vs-Sherbrooke per-edge stats
    fig_mqt9_raw_vs_log()                            # Fig 7  unscaled vs log1p
    fig_feature_correlation(train)                   # spectral-17 correlation heatmap
    fig_ridge_vs_lasso_importance(train, "route")    # Appendix  ridge-vs-lasso (L1-artifact check)

    print("\nDONE. Report artifacts in", OUT)
    for f in ("table1_indist_brisbane.png", "table2_brisbane_unseen.png",
              "table3_validation_tuned.png", "headline_crossdevice_cv.png",
              "table5_lasso_featurerep.png", "table_lasso_importance_ood_route.png",
              "sensitivity_ablation_by_device_route.png", "device_error_table.png",
              "why_berlin_boston_fail.png", "sherbrooke_degeneration.png",
              "per_edge_error_correlation.png", "mqt9_raw_vs_log.png",
              "feature_correlation_spearman.png", "ridge_vs_lasso_importance.png"):
        mark = "  " if os.path.exists(os.path.join(OUT, f)) else "  [MISSING] "
        print(f"{mark}{f}")


if __name__ == "__main__":
    main()
