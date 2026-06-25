#!/usr/bin/env python3
"""
report_artifacts.py -- ONE script that reproduces every table and figure in the supervised
section of the report. It is the single committable consolidation of three former side scripts
(ood_indist_tuned.py + mqt_feature_eval.py + build_report_artifacts.py).

It does NO modeling of its own beyond hyperparameter SELECTION: every fit / metric / CV / OOD
evaluation is delegated to the canonical harness `supervised_analysis_run.py` (imported as H).
Models: kNN (instance/locality), Ridge + Lasso (regularized linear), HistGB + RF (tree ensembles)
-- five diverse families. Metrics: R2 (level), Spearman (rank), and within-N z-scored polarization
(relative resilience). Targets: route = log1p(bare_routed_2q), pol = mirror polarization,
pol_z = within-N z-scored pol (train-fit, no normalization leakage).

Pipeline (main): a preflight check verifies the full analysis has been run (training CSVs,
validation labels, cross_device results), then COMPUTE writes the CSVs each table reads and
RENDER draws the headline summary, the five tables, and the In-Depth-Evaluation importance table.

  RENDER (data/results/figures/, each table as .png + .csv)
    Overall summary  best model per family on the pooled OOD test, route R² + bootstrap CI + spread
    Table 1  in-distribution FakeBrisbane 5-fold CV (spectral-17) -- baseline/context
    Table 2  device held constant, unseen circuits (spectral-17)                 [cross_device.py CSV]
    Table 3  hyperparameters tuned in-distribution, tested once on validation
    Table 4  every sklearn model on the OOD cross-device test, 3 targets x devices [cross_device.py CSV]
    Table 5  LASSO feature representation: MQT-9 (log1p) vs spectral-17, route/pol/pol_z
    + In-Depth Eval: LASSO feature importance & ablation (best model) -> table_lasso_importance

Cross-device Tables 2 & 4 read cross_device.py's spectral results CSV; if absent they are skipped
with a note. Compute steps read the cached validation labels under data/xdev_out/labeled/ (produced
by cross_device.py --phase label). Reproduce the importance numbers per-feature with the harness:
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

OUT = str(REPO / "data" / "results" / "figures")
os.makedirs(OUT, exist_ok=True)
LABELED = REPO / "data" / "xdev_out" / "labeled"
TUNED_CSV = str(REPO / "data" / "results" / "cross_device_indist_tuned.csv")
LOG_LASSO_CSV = os.path.join(OUT, "mqt9_log_lasso.csv")
CDV = {"spectral": str(REPO / "data/xdev_out/cross_device_results.csv"),
       "mqt9_raw": str(REPO / "data/xdev_out/mqt9_run/cross_device_results.csv")}

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
# COMPUTE 1 -- in-distribution-tuned validation table  (was ood_indist_tuned.py)
#
# Tune by 5-fold GroupKFold CV on TRAIN only; test the 344-circuit validation corpus exactly once.
# This REPLACES the removed transfer-tuned (leave-one-device-out) grid, which selected
# hyperparameters using OOD performance -- i.e. peeked at the test set. The grid is unchanged; only
# the selection OBJECTIVE is honest now. Sanity: lasso here matches the harness's own tuned lasso
# (`supervised_analysis_run.py --ood-csv ... --model lasso`), since that is itself a GroupKFold
# GridSearchCV over the same alpha range.
# ===========================================================================
TUNE_DEVICE = "FakeBrisbane"


def _configs(m):
    """The grid searched per model (same grid the transfer-tuned version used)."""
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
# COMPUTE 2 -- log-MQT-9 LASSO table  (was mqt_feature_eval.py)
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


def render_table(col_labels, row_labels, cell_text, path, title, note="",
                 best_cells=None, neg_cells=None, row_w=None, label_w=None, dim_cells=None):
    best_cells = best_cells or set()
    neg_cells = neg_cells or set()
    dim_cells = dim_cells or set()
    nrows = len(row_labels)
    label_in = label_w if label_w else 1.45
    data_ins = list(row_w) if row_w else [1.7] * len(col_labels)
    fig_w = label_in + sum(data_ins)
    row_in = 0.30
    title_in = 0.24 + 0.20 * (title.count("\n") + 1)
    note_in = 0.24 if note else 0.05
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
    tbl.set_fontsize(10)
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


def table4(cdv):
    """Table 4 -- every sklearn model on the OOD cross-device test: 3 targets x devices x 5 models,
    spectral-17, read from cross_device.py's results CSV (the canonical cross-device engine).
    Rows are (target, device); columns are the five sklearn families (GNN excluded -- it is
    benchmarked separately, not part of the sklearn grid)."""
    sp = cdv[cdv.rep == "spectral"]
    tgts = ["route", "pol", "pol_z"]
    rows_lbl, cell, mat = [], [], []
    for t in tgts:
        for dev, short in zip(DEVICES, DEV_SHORT):
            vals = []
            for m in MODELS:
                s = sp[(sp.target == t) & (sp.device == dev) & (sp.model == m)]
                vals.append(float(s.R2.values[0]) if len(s) else np.nan)
            rows_lbl.append(f"{t} · {short}")
            cell.append([(f"{v:.2f}" if np.isfinite(v) else "—") for v in vals])
            mat.append(vals)
    best_cells = {(i, int(np.nanargmax(r))) for i, r in enumerate(mat) if np.any(np.isfinite(r))}
    render_table(MODELS, rows_lbl, cell, os.path.join(OUT, "table4_ood_grid.png"),
                 "Table 4 — Every sklearn model on the OOD cross-device test (spectral-17)",
                 note=f"{ROUTE_DEF};  pol_z = within-N z-scored pol.  R²: train FakeBrisbane → test device.  "
                      "green = best (R²>0);  light-red = best-but-R²<0 (least-bad, not a win);  * = Brisbane self-transfer.",
                 best_cells=best_cells, neg_cells=_neg_cells(mat), label_w=1.9)
    pd.DataFrame(mat, index=rows_lbl, columns=MODELS).to_csv(os.path.join(OUT, "table4_ood_grid.csv"))


# ===========================================================================
# OVERALL SUMMARY (the headline result) -- per-family route R² on the POOLED OOD test:
# train FakeBrisbane, test the four non-Brisbane validation corpora concatenated into one OOD set.
# Point estimate + 95% CI come from the harness run_external (cluster bootstrap over circuit
# groups); the spread column is the sd of per-device route R² across the four devices. Everything
# is computed from the harness -- nothing hard-coded.
# ===========================================================================
TRAIN_DEV = "FakeBrisbane"
OOD_DEVS = ["FakeBerlin", "FakeBoston", "FakeSherbrooke", "FakeTorino"]
SUMMARY_CSV = str(REPO / "data" / "results" / "ood_summary_route.csv")


def _pooled_ood_swap(train):
    """Concatenate the non-Brisbane swap validation corpora into one OOD test set, aligned to the
    spectral columns shared with TRAIN. Group keys are made device-unique (@device) so the cluster
    bootstrap treats each device's circuit instance as its own independence unit."""
    frames = []
    for dev in OOD_DEVS:
        od = load_external(str(LABELED / f"val_swap_{dev}.csv")).copy()
        od["grp"] = od["grp"].astype(str) + "@" + dev
        od["_device"] = dev
        frames.append(od)
    pooled = pd.concat(frames, ignore_index=True)
    return pooled, shared_feats("spectral", train, pooled)


def compute_ood_summary(train, B=2000):
    print("== OVERALL SUMMARY: pooled-OOD route R² + 95% bootstrap CI per family ==")
    pooled, feats = _pooled_ood_swap(train)
    X_tr = train[feats].fillna(0).values
    X_pool = pooled[feats].fillna(0).values
    y_pool = pooled["route"].values.astype(float)
    dev_of = pooled["_device"].values
    rows = []
    for m in MODELS:
        point, cis, pred = run_external(train, X_tr, pooled, X_pool, "route", m, B=B)
        per = [metrics_full(y_pool[dev_of == d], pred[dev_of == d])["R2"] for d in OOD_DEVS]
        rows.append(dict(family=m, R2=point["R2"], lo=cis["R2"][0], hi=cis["R2"][1],
                         spread_sd=float(np.std(per)),
                         **{f"R2_{d.replace('Fake', '')}": v for d, v in zip(OOD_DEVS, per)}))
        print(f"  {m:7} R2={point['R2']:+.3f}  95%CI[{cis['R2'][0]:+.3f},{cis['R2'][1]:+.3f}]  "
              f"dev-sd={np.std(per):.3f}")
    df = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"  wrote {SUMMARY_CSV}")
    return df


def table_overall_summary(df):
    """Headline per-family OOD summary (route R²), sorted best-first."""
    cols = ["route R² (OOD)", "95% CI", "device spread (sd)"]
    cell = [[f"{r.R2:.3f}", f"[{r.lo:+.2f}, {r.hi:+.2f}]", f"±{r.spread_sd:.3f}"]
            for _, r in df.iterrows()]
    best_cells = {(int(np.argmax(df.R2.values)), 0)}
    render_table(cols, list(df.family), cell, os.path.join(OUT, "table_overall_ood_summary.png"),
                 "Overall summary — best model per family on the OOD cross-device test\n"
                 "(train FakeBrisbane → test Berlin / Boston / Sherbrooke / Torino, pooled)",
                 note=f"{ROUTE_DEF}.  95% CI = cluster bootstrap over groups;  "
                      "spread = sd across the 4 OOD devices.  green = best.",
                 best_cells=best_cells, row_w=[1.9, 2.0, 2.2], label_w=1.6)


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
        cell.append([("—" if not np.isfinite(v) else (f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"))
                     for v in r]); mat.append(r)
    arr = np.array(mat)
    # best rep per task = argmax over {MQT-9 log1p, spectral} -- the 2 columns of each target block.
    best_cells = set()
    for i in range(len(order)):
        for k in range(len(tasks)):
            pair = arr[i, 2 * k:2 * k + 2]
            if np.any(np.isfinite(pair)):
                best_cells.add((i, 2 * k + int(np.nanargmax(pair))))
    render_table(cols, rlbl, cell, os.path.join(OUT, "table5_lasso_featurerep.png"),
                 "Table 5 — LASSO feature representation: MQT-9 (log1p) vs spectral-17",
                 note=f"{ROUTE_DEF};  pol_z = within-N z-scored pol.  LASSO;  MQT-9 counts are log1p-scaled.  "
                      "green = best rep per task;  light-red = least-bad but R²<0;  * = Brisbane self-transfer.",
                 best_cells=best_cells, neg_cells=_neg_cells(arr))
    d[d.task.isin(tasks)].to_csv(os.path.join(OUT, "table5_lasso_featurerep.csv"), index=False)


# ===========================================================================
# IN-DEPTH EVALUATION (start) -- feature importance + ablation on the BEST model (LASSO).
# Reproduce per-feature directly with the harness CLI:
#   python src/supervised/supervised_analysis_run.py --model lasso --features spectral --target route --coef
#   python src/supervised/supervised_analysis_run.py --model lasso --features spectral --target route --ablate
#   python src/supervised/supervised_analysis_run.py --model lasso --features spectral --target route --importance
# ===========================================================================
REP_NICE = {"spectral": "spectral-17", "mqt9": "MQT-9"}


def compute_lasso_importance(train, target="route", features="spectral"):
    """LASSO feature importance + ablation on a feature set (default spectral-17 = the best model;
    features='mqt9' produces the appendix MQT-9 version).
    importance = standardized lasso coefficients (per-SD effect; L1 zeros the unhelpful features,
                 via the harness diagnostics.model_importances);
    ablation   = CV R² lost when each feature is dropped (leave-one-feature-out, harness run_indist).
    Returns (df sorted by |coef|, full CV R²). Nothing hard-coded."""
    import diagnostics                                # harness leaf module
    feats = _rep_shared_cols(features, train)        # spectral-17 or local MQT-9 columns
    X = train[feats].fillna(0).values
    y = resolve_target(train, target, np.ones(len(train), bool), False)
    pipe = H._fit_full("lasso", X, y, train.grp.values)
    _label, _names, coef = diagnostics.model_importances(pipe, feats)   # standardized coefs
    full = run_indist(train, X, target, "lasso")["R2"]
    rows = []
    for i, f in enumerate(feats):
        r2_drop = run_indist(train, np.delete(X, i, axis=1), target, "lasso")["R2"]
        rows.append(dict(feature=f, coef=float(coef[i]), abl_dR2=float(full - r2_drop),
                         selected=bool(abs(coef[i]) > 1e-6)))
    df = (pd.DataFrame(rows)
          .reindex(pd.DataFrame(rows).coef.abs().sort_values(ascending=False).index)
          .reset_index(drop=True))
    out = str(REPO / "data" / "results" / f"lasso_importance_{features}_{target}.csv")
    df.to_csv(out, index=False)
    print(f"== LASSO importance/ablation ({REP_NICE.get(features, features)}, {target}): "
          f"{int(df.selected.sum())}/{len(df)} features selected (full CV R²={full:.3f}) -> {out}")
    return df, full


def table_lasso_importance(df, full_r2, target="route", features="spectral"):
    """Render the feature importance & ablation table for LASSO on the given feature set."""
    nice = REP_NICE.get(features, features)
    cols = ["std. coef", "ablation ΔR²"]
    rlbl, cell, dim = [], [], set()
    for i, (_, r) in enumerate(df.iterrows()):
        rlbl.append(r.feature)
        cell.append([f"{r.coef:+.3f}", f"{r.abl_dR2:+.4f}"])
        if not r.selected:                           # L1-zeroed: grey it out
            dim |= {(i, 0), (i, 1)}
    suffix = "" if features == "spectral" else f"_{features}"
    render_table(cols, rlbl, cell, os.path.join(OUT, f"table_lasso_importance{suffix}.png"),
                 f"Feature importance & ablation — LASSO ({target}, {nice})",
                 note=f"std. coef = per-SD lasso weight (grey = zeroed by L1);  ablation ΔR² = CV R² lost "
                      f"dropping the feature;  full CV R² = {full_r2:.3f}.",
                 dim_cells=dim, row_w=[2.2, 2.4], label_w=3.0)


# ===========================================================================
# OPTION A -- a-priori (leakage-free) device-noise feature added to spectral-17  (was device_noise_legit.py)
#
# Per circuit/device, from backend.target calibration ONLY (no routed_2q, no label transform):
#   exp_2qerr   = sum of the device's best-k 2q-gate errors, k = #interaction edges
#   meanbk_2qerr= exp_2qerr / k        med_2qerr = device median 2q error    mean_readout = device mean readout
# Train on Brisbane (Brisbane calibration), test each device's val_pol with ITS calibration. The
# question: does a cheap a-priori noise scalar let LASSO recover the polarization level cross-device?
# ===========================================================================
OPTIONA_CSV = str(REPO / "data" / "results" / "option_a.csv")


def _device_cal(name):
    import qiskit_ibm_runtime.fake_provider as fp
    t = getattr(fp, name)().target
    g2 = next(g for g in ("ecr", "cz", "cx") if g in t.operation_names)
    errs2 = np.array(sorted(p.error for _, p in t[g2].items() if p and p.error is not None))
    ro = ([p.error for _, p in t["measure"].items() if p and p.error is not None]
          if "measure" in t.operation_names else [])
    return dict(cum=np.cumsum(errs2), ne=len(errs2), med2=float(np.median(errs2)),
                meanro=float(np.mean(ro)) if ro else 0.0)


def _expected_feats(df, cal):
    """[exp_2qerr, meanbk_2qerr, med_2qerr, mean_readout] per row -- a-priori, no routed_2q."""
    cum, ne = cal["cum"], cal["ne"]
    out = []
    for s in df["interaction_edges"].values:
        k = min(len(json.loads(s)), ne)
        e = float(cum[k - 1]) if k > 0 else 0.0
        out.append([e, e / k if k > 0 else 0.0, cal["med2"], cal["meanro"]])
    return np.array(out)


def compute_optionA(train, model="lasso"):
    """Blind (spectral-17) vs +A (spectral-17 + a-priori device noise) for pol and pol_z, per device.
    Returns a tidy df or None if qiskit's fake_provider isn't importable (Option A then skips)."""
    try:
        import qiskit_ibm_runtime.fake_provider  # noqa: F401
    except Exception as e:
        print(f"  [skip Option A] qiskit fake_provider unavailable ({type(e).__name__}); "
              "pip install qiskit-ibm-runtime to enable.")
        return None
    feats = shared_feats("spectral", train)
    cal = {d: _device_cal(d) for d in DEVICES}
    frames = {d: load_external(str(LABELED / f"val_pol_{d}.csv")) for d in DEVICES}
    Xs_tr = train[feats].fillna(0).values
    Xa_tr = _expected_feats(train, cal[TRAIN_DEV])          # train rows use the TRAIN device's calibration
    grp = train.grp.values
    stats, gm, gs = fit_zstats(train["pol"].values, train.N.values)
    rows = []
    for tgt in ["pol", "pol_z"]:
        if tgt == "pol":
            y_tr = train["pol"].values.astype(float)
            y_of = {d: frames[d]["pol"].values.astype(float) for d in DEVICES}
        else:                                               # within-N z, train-fit (run_external recipe)
            y_tr = apply_zstats(train["pol"].values, train.N.values, stats, gm, gs)
            y_of = {d: apply_zstats(frames[d]["pol"].values, frames[d].N.values, stats, gm, gs)
                    for d in DEVICES}
        for use_A in (False, True):
            Xtr = np.column_stack([Xs_tr, Xa_tr]) if use_A else Xs_tr
            pipe = H._fit_full(model, Xtr, y_tr, grp)
            for d in DEVICES:
                Xs = frames[d][feats].fillna(0).values
                X = np.column_stack([Xs, _expected_feats(frames[d], cal[d])]) if use_A else Xs
                r2 = metrics_full(y_of[d], pipe.predict(X))["R2"]
                rows.append(dict(device=d, target=tgt, scheme=("+A" if use_A else "blind"), R2=r2))
    df = pd.DataFrame(rows)
    df.to_csv(OPTIONA_CSV, index=False)
    print(f"== Option A ({model}): a-priori device noise added to spectral-17 -> {OPTIONA_CSV}")
    return df


def table_optionA(df, model="lasso"):
    """Render the Option A table: per device, pol & pol_z, blind vs +A. Green where +A improves over
    blind; light-red where the +A best-in-pair is still R²<0."""
    cols = ["pol\nblind", "pol\n+A", "pol_z\nblind", "pol_z\n+A"]
    cell, mat = [], []
    for d in DEVICES:
        r = []
        for tgt in ("pol", "pol_z"):
            for sch in ("blind", "+A"):
                v = df[(df.device == d) & (df.target == tgt) & (df.scheme == sch)].R2.values
                r.append(float(v[0]) if len(v) else np.nan)
        cell.append([f"{x:+.2f}" for x in r]); mat.append(r)
    arr = np.array(mat)
    best_cells = set()
    for i in range(len(DEVICES)):
        if np.isfinite(arr[i, 1]) and arr[i, 1] > arr[i, 0]:   # pol: +A beat blind
            best_cells.add((i, 1))
        if np.isfinite(arr[i, 3]) and arr[i, 3] > arr[i, 2]:   # pol_z: +A beat blind
            best_cells.add((i, 3))
    render_table(cols, DEV_SHORT, cell, os.path.join(OUT, "table_option_a.png"),
                 f"Option A — a-priori device-noise feature added to spectral-17 ({model.upper()})",
                 note="blind = spectral-17;  +A = + 4 a-priori device-noise cal features.  "
                      "green = +A beats blind;  light-red = +A best but R²<0;  * = self-transfer.",
                 best_cells=best_cells, neg_cells=_neg_cells(arr))
    df.to_csv(os.path.join(OUT, "table_option_a.csv"), index=False)


# ===========================================================================
# PREDICTION-vs-TRUTH CHARTS (optional, --charts) -- per device, scatter of predicted vs true with
# a fitted regression line + the y=x identity. Train Brisbane (spectral-17), test each device's
# val_pol. target='pol' (raw) or 'pol_z' (within-N z, train-fit, the run_external recipe).
# ===========================================================================
_DEV_COLOR = {"FakeBrisbane": "#4C72B0", "FakeBerlin": "#DD8452", "FakeBoston": "#C44E52",
              "FakeSherbrooke": "#55A868", "FakeTorino": "#8172B3"}


def fig_pred_vs_truth(train, model="lasso", target="pol_z"):
    """Per-device predicted-vs-true scatter + regression line + y=x. Writes pred_vs_truth_<target>.png."""
    short = dict(zip(DEVICES, DEV_SHORT))
    feats = shared_feats("spectral", train)
    if target == "pol_z":
        stats, gm, gs = fit_zstats(train["pol"].values, train.N.values)
        ytr = apply_zstats(train["pol"].values, train.N.values, stats, gm, gs)
        ylab = "polarization z-score (within-N)"
    else:
        ytr = train["pol"].values.astype(float); ylab = "polarization"
    pipe = H._fit_full(model, train[feats].fillna(0).values, ytr, train.grp.values)
    panels = {}
    for d in DEVICES:
        ood = load_external(str(LABELED / f"val_pol_{d}.csv"))
        y = (apply_zstats(ood["pol"].values, ood.N.values, stats, gm, gs) if target == "pol_z"
             else ood["pol"].values.astype(float))
        panels[d] = (y, pipe.predict(ood[feats].fillna(0).values))
    lo = min(min(y.min(), p.min()) for y, p in panels.values())
    hi = max(max(y.max(), p.max()) for y, p in panels.values())
    pad = 0.05 * (hi - lo); ax_lo, ax_hi = lo - pad, hi + pad
    fig, axes = plt.subplots(1, len(DEVICES), figsize=(3.5 * len(DEVICES), 3.9),
                             sharex=True, sharey=True, constrained_layout=True)
    for ax, d in zip(axes, DEVICES):
        y, pred = panels[d]; mm = metrics_full(y, pred)
        ax.plot([ax_lo, ax_hi], [ax_lo, ax_hi], "k--", lw=1, alpha=.6, zorder=1, label="y = x")
        ax.scatter(y, pred, s=14, c=_DEV_COLOR[d], alpha=.5, edgecolor="none", zorder=2)
        m, b = np.polyfit(y, pred, 1)                    # least-squares regression line pred ~ y
        xs = np.array([ax_lo, ax_hi])
        ax.plot(xs, m * xs + b, c=_DEV_COLOR[d], lw=2, zorder=3, label=f"fit (slope {m:.2f})")
        ax.set_title(f"{short[d]}   R²={mm['R2']:.2f}  ρ={mm['Spearman']:.2f}", fontsize=11)
        ax.set_xlabel("true"); ax.set_xlim(ax_lo, ax_hi); ax.set_ylim(ax_lo, ax_hi)
        ax.grid(alpha=.25); ax.set_aspect("equal", adjustable="box")
        if d == DEVICES[0]:
            ax.legend(fontsize=8, loc="upper left", framealpha=.9)
    axes[0].set_ylabel(f"predicted ({ylab})")
    fig.suptitle(f"{model.upper()} prediction vs truth — {target}  (train Brisbane → test device; "
                 "regression line + y=x identity)", fontsize=12.5)
    fp = os.path.join(OUT, f"pred_vs_truth_{target}.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {fp}")


# ===========================================================================
# FEW-SHOT AFFINE CALIBRATION  (was few_shot_calibration.py; extended to pol_z)  -- FEW-SHOT, NOT zero-shot
#
# The headline summary / T4 are LABEL-FREE zero-shot (the model never sees the target device). This is
# a DIFFERENT epistemic category: fit a 2-param affine map a*ŷ+b on K LABELED target circuits, evaluate
# on the held-out remainder. It tests whether the cross-device collapse is an affine offset (recoverable
# with a handful of labels) rather than the model learning the wrong structure. Keep that boundary
# explicit -- few-shot recovery does NOT undercut the zero-shot failure; it localizes it.
#
# Discipline: the K calibration circuits and the eval remainder are GROUP-DISJOINT (no interaction-graph
# sibling spans fit/eval -- same GroupKFold rule used everywhere), and results are mean +- sd over draws.
# pol clips a*ŷ+b to [0,1] and also fits the physics model exp(alpha*routed_2q) for contrast; pol_z is a
# z-score, so it is NOT clipped and physics is not applied (a multiplicative gate-decay is meaningless on
# a standardized target).
# ===========================================================================
KS_FEWSHOT = [5, 10, 20, 40]
FEWSHOT_SEEDS = 25
FEWSHOT_CSV = str(REPO / "data" / "results" / "few_shot_calibration.csv")


def _fit_affine(pred, y):
    a, b = np.linalg.lstsq(np.column_stack([pred, np.ones_like(pred)]), y, rcond=None)[0]
    return float(a), float(b)


def _apply_affine(pred, a, b, clip):
    v = a * pred + b
    return np.clip(v, clip[0], clip[1]) if clip is not None else v


def _fit_physics(pred, m, y):
    from scipy.optimize import minimize_scalar
    def loss(al):
        return float(np.mean((np.clip(pred * np.exp(al * m), 0.0, 1.0) - y) ** 2))
    return float(minimize_scalar(loss, bounds=(-0.05, 0.05), method="bounded").x)


def _kshot_split(groups, K, rng):
    """K calibration circuits drawn at random; eval = circuits whose GROUP is untouched by the
    calibration set, so no interaction-graph sibling is split across fit/eval (group-disjoint)."""
    n = len(groups)
    cal = rng.choice(n, size=K, replace=False)
    cal_groups = set(groups[cal])
    ev = np.array([i for i in range(n) if groups[i] not in cal_groups])
    return cal, ev


def compute_fewshot(train, model="lasso", targets=("pol", "pol_z")):
    """Few-shot affine (and, for pol, physics) recalibration per device and K. Group-disjoint
    fit/eval; mean +- sd over FEWSHOT_SEEDS draws. Writes few_shot_calibration.csv."""
    feats = shared_feats("spectral", train)
    frames = {d: load_external(str(LABELED / f"val_pol_{d}.csv")) for d in DEVICES}
    stats, gm, gs = fit_zstats(train["pol"].values, train.N.values)
    rows = []
    for tgt in targets:
        if tgt == "pol":
            ytr = train["pol"].values.astype(float); clip = (0.0, 1.0); do_phys = True
            y_of = {d: frames[d]["pol"].values.astype(float) for d in DEVICES}
        else:                                            # pol_z: z-score, no clip, no physics
            ytr = apply_zstats(train["pol"].values, train.N.values, stats, gm, gs)
            clip = None; do_phys = False
            y_of = {d: apply_zstats(frames[d]["pol"].values, frames[d].N.values, stats, gm, gs)
                    for d in DEVICES}
        pipe = H._fit_full(model, train[feats].fillna(0).values, ytr, train.grp.values)
        for d in DEVICES:
            ood = frames[d]; y = y_of[d]; groups = ood.grp.values
            pred = pipe.predict(ood[feats].fillna(0).values)
            m = ood["routed_2q"].values.astype(float) if (do_phys and "routed_2q" in ood.columns) else None
            r2_raw = metrics_full(y, pred)["R2"]
            for K in KS_FEWSHOT:
                af, ph = [], []
                for s in range(FEWSHOT_SEEDS):
                    rng = np.random.default_rng(1000 * s + K + (0 if tgt == "pol" else 7))
                    cal, ev = _kshot_split(groups, K, rng)
                    if len(ev) < 5 or len(np.unique(pred[cal])) < 2:
                        continue
                    a, b = _fit_affine(pred[cal], y[cal])
                    af.append(metrics_full(y[ev], _apply_affine(pred[ev], a, b, clip))["R2"])
                    if do_phys and m is not None:
                        al = _fit_physics(pred[cal], m[cal], y[cal])
                        ph.append(metrics_full(y[ev], np.clip(pred[ev] * np.exp(al * m[ev]), 0, 1))["R2"])
                rec = dict(target=tgt, device=d, K=K, R2_raw=r2_raw,
                           R2_affine=float(np.nanmean(af)), R2_affine_sd=float(np.nanstd(af)))
                if ph:
                    rec["R2_phys"] = float(np.nanmean(ph)); rec["R2_phys_sd"] = float(np.nanstd(ph))
                rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(FEWSHOT_CSV, index=False)
    print(f"== Few-shot affine calibration ({model}, group-disjoint, {FEWSHOT_SEEDS} draws) -> {FEWSHOT_CSV}")
    return df


def table_fewshot(df):
    """Compact few-shot summary: per (target, device), zero-shot R² vs affine at K=5/20/40 (mean ± sd)."""
    tgts = ["pol", "pol_z"]; Kshow = [5, 20, 40]
    cols = ["zero-shot"] + [f"affine\nK={k}" for k in Kshow]
    rlbl, cell, mat = [], [], []
    for tgt in tgts:
        for d, short in zip(DEVICES, DEV_SHORT):
            sub = df[(df.target == tgt) & (df.device == d)]
            raw = float(sub.R2_raw.iloc[0]) if len(sub) else np.nan
            row, texts = [raw], [f"{raw:+.2f}"]
            for k in Kshow:
                r = sub[sub.K == k]
                if len(r):
                    mny, sdy = float(r.R2_affine.iloc[0]), float(r.R2_affine_sd.iloc[0])
                    row.append(mny); texts.append(f"{mny:+.2f} ±{sdy:.2f}")
                else:
                    row.append(np.nan); texts.append("—")
            rlbl.append(f"{tgt} · {short}"); cell.append(texts); mat.append(row)
    arr = np.array(mat)
    best = {(i, j) for i in range(len(rlbl)) for j in (1, 2, 3)
            if np.isfinite(arr[i, j]) and arr[i, j] > 0}             # affine recovered to R²>0
    render_table(cols, rlbl, cell, os.path.join(OUT, "table_fewshot_calibration.png"),
                 "Few-shot affine calibration (LASSO) — K labeled target circuits recover the level",
                 note=f"FEW-SHOT: a·ŷ+b fit on K target circuits (group-disjoint fit/eval), mean±sd over "
                      f"{FEWSHOT_SEEDS} draws — NOT zero-shot (summary/T4 are label-free).  green = recovered R²>0.",
                 best_cells=best, neg_cells=_neg_cells(arr), label_w=2.2, row_w=[1.5, 2.0, 2.0, 2.0])
    df.to_csv(os.path.join(OUT, "table_fewshot_calibration.csv"), index=False)


def fig_fewshot(df):
    """Learning curves: held-out R² vs K, affine (+ physics for pol), per device, pol & pol_z."""
    tgts = ["pol", "pol_z"]
    fig, axes = plt.subplots(2, len(DEVICES), figsize=(2.6 * len(DEVICES), 6.0),
                             sharex=True, constrained_layout=True)
    for r, tgt in enumerate(tgts):
        for c, (d, short) in enumerate(zip(DEVICES, DEV_SHORT)):
            ax = axes[r, c]; s = df[(df.target == tgt) & (df.device == d)].sort_values("K")
            K = s.K.values
            ax.axhline(0, color="#888", lw=1)
            if len(s):
                ax.axhline(s.R2_raw.iloc[0], color="#C44E52", ls="--", lw=1.3,
                           label=f"zero-shot ({s.R2_raw.iloc[0]:.2f})")
            ax.errorbar(K, s.R2_affine, yerr=s.R2_affine_sd, marker="o", color="#55A868",
                        capsize=3, label="affine (2-param)")
            if "R2_phys" in s.columns and s.R2_phys.notna().any():
                ax.errorbar(K, s.R2_phys, yerr=s.R2_phys_sd, marker="s", color="#4C72B0",
                            capsize=3, label="physics exp(α·m)")
            ax.set_title(f"{short} — {tgt}", fontsize=10); ax.set_xticks(K); ax.grid(alpha=.25)
            if r == 1:
                ax.set_xlabel("K target labels")
            if c == 0:
                ax.set_ylabel("held-out R²")
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Few-shot affine calibration — K labeled target circuits recover pol / pol_z level   "
                 "(FEW-SHOT, not zero-shot; group-disjoint fit/eval; mean ± sd over "
                 f"{FEWSHOT_SEEDS} draws)", fontsize=12)
    fp = os.path.join(OUT, "few_shot_calibration.png")
    fig.savefig(fp, dpi=130); plt.close(fig)
    print(f"  wrote {fp}")


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
    import argparse
    ap = argparse.ArgumentParser(description="Reproduce the supervised-report tables (and optional "
                                             "prediction-vs-truth charts) from the harness.")
    ap.add_argument("--charts", action="store_true",
                    help="also render the per-device prediction-vs-truth regression charts (pol, pol_z)")
    ap.add_argument("--no-optiona", action="store_true",
                    help="skip the Option A device-noise table (its qiskit dependency)")
    args = ap.parse_args()

    ok, have_cdv = preflight()
    if not ok:
        sys.exit(1)

    print("\nloading train set...")
    train = load_data(DEFAULT_SWAP_CSV, DEFAULT_POL_CSV)

    # --- compute (write the CSVs each table reads; nothing hard-coded) ---
    compute_indist_tuned(train)                      # -> T3
    compute_log_lasso(train)                         # -> T5
    summary = compute_ood_summary(train)             # -> overall summary
    imp_df, full_r2 = compute_lasso_importance(train, "route")          # -> importance (spectral-17)
    imp_m, full_m = compute_lasso_importance(train, "route", "mqt9")    # -> importance appendix (MQT-9)
    optA = None if args.no_optiona else compute_optionA(train, "lasso")  # -> Option A (skips w/o qiskit)
    fewshot = compute_fewshot(train, "lasso")        # -> few-shot affine calibration (pol, pol_z)

    cdv = load_cdv() if have_cdv else None

    # --- render: overall summary + five tables + In-Depth-Eval importance + Option A ---
    print("rendering tables...")
    table_overall_summary(summary)                   # headline result
    table1(train)                                    # T1  in-distribution (baseline/context)
    if cdv is not None:
        table2(cdv)                                  # T2  device held constant
        table4(cdv)                                  # T4  15x5 OOD grid
    else:
        print("  [skip T2 + T4: cross_device_results.csv absent]")
    table3()                                         # T3  hyperparameter tuning
    table5()                                         # T5  LASSO feature representation
    table_lasso_importance(imp_df, full_r2, "route")            # importance & ablation (spectral-17)
    table_lasso_importance(imp_m, full_m, "route", "mqt9")      # importance appendix (MQT-9)
    if optA is not None:
        table_optionA(optA, "lasso")                 # Option A device-noise recovery (zero-shot)
    table_fewshot(fewshot)                           # few-shot affine calibration (NOT zero-shot)

    if args.charts:
        print("rendering charts...")
        fig_pred_vs_truth(train, "lasso", "pol")
        fig_pred_vs_truth(train, "lasso", "pol_z")
        fig_fewshot(fewshot)

    print("\nDONE. Artifacts in", OUT)
    for f in sorted(os.listdir(OUT)):
        if f.startswith(("table_overall", "table1", "table2", "table3", "table4", "table5",
                         "table_lasso", "table_option", "table_fewshot", "pred_vs_truth",
                         "few_shot")):
            print("  ", f)


if __name__ == "__main__":
    main()
