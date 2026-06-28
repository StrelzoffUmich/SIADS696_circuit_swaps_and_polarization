#!/usr/bin/env python3
"""exploratory_artifacts.py -- the NON-REPORT supervised figures/tables.

These artifacts were factored OUT of report_artifacts.py: they are exploratory / diagnostic and do
NOT appear in the final report. report_artifacts.py remains the single producer of the report set;
this module reuses its shared infrastructure (imported as R) so nothing is duplicated.

Contents (each delegates fit/metric/CV/OOD to the canonical harness via R / R.H):
  * overall pooled-OOD route summary (table_overall_ood_summary)
  * in-distribution LASSO feature importance & ablation (route spectral-17, MQT-9, pol, pol_z) + combined
  * Table-4 OOD grid (the 15x5 coloured table; the report uses bars instead)
  * Option A -- a-priori device-noise feature added to spectral-17
  * few-shot affine calibration table + learning curves
  * per-device prediction-vs-truth scatters (pol, pol_z)

Run from src/supervised:  python viz_tests/exploratory_artifacts.py
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# report_artifacts lives in the parent dir (src/supervised); make it importable from any cwd
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent.parent))
import report_artifacts as R   # noqa: E402  -- shared infrastructure (render_table, OUT, harness, ...)


# ===========================================================================
# OVERALL SUMMARY (the headline result) -- per-family route R² on the POOLED OOD test:
# train FakeBrisbane, test the four non-Brisbane validation corpora concatenated into one OOD set.
# Point estimate + 95% CI come from the harness run_external (cluster bootstrap over circuit
# groups); the spread column is the sd of per-device route R² across the four devices. Everything
# is computed from the harness -- nothing hard-coded.
# ===========================================================================
SUMMARY_CSV = str(R.REPO / "data" / "results" / "ood_summary_route.csv")


def _pooled_ood_swap(train):
    """Concatenate the non-Brisbane swap validation corpora into one OOD test set, aligned to the
    spectral columns shared with TRAIN. Group keys are made device-unique (@device) so the cluster
    bootstrap treats each device's circuit instance as its own independence unit."""
    frames = []
    for dev in R.OOD_DEVS:
        od = R.load_external(str(R.LABELED / f"val_swap_{dev}.csv")).copy()
        od["grp"] = od["grp"].astype(str) + "@" + dev
        od["_device"] = dev
        frames.append(od)
    pooled = pd.concat(frames, ignore_index=True)
    return pooled, R.shared_feats("spectral", train, pooled)


def compute_ood_summary(train, B=2000):
    print("== OVERALL SUMMARY: pooled-OOD route R² + 95% bootstrap CI per family ==")
    pooled, feats = _pooled_ood_swap(train)
    X_tr = train[feats].fillna(0).values
    X_pool = pooled[feats].fillna(0).values
    y_pool = pooled["route"].values.astype(float)
    dev_of = pooled["_device"].values
    rows = []
    for m in R.MODELS:
        point, cis, pred = R.run_external(train, X_tr, pooled, X_pool, "route", m, B=B)
        per = [R.metrics_full(y_pool[dev_of == d], pred[dev_of == d])["R2"] for d in R.OOD_DEVS]
        rows.append(dict(family=m, R2=point["R2"], lo=cis["R2"][0], hi=cis["R2"][1],
                         spread_sd=float(np.std(per)),
                         **{f"R2_{d.replace('Fake', '')}": v for d, v in zip(R.OOD_DEVS, per)}))
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
    R.render_table(cols, list(df.family), cell, os.path.join(R.OUT_EXPLORATORY, "table_overall_ood_summary.png"),
                   "Overall summary — best model per family on the OOD cross-device test\n"
                   "(train FakeBrisbane → test Berlin / Boston / Sherbrooke / Torino, pooled)",
                   note=f"{R.ROUTE_DEF}.  95% CI = cluster bootstrap over groups;  "
                        "spread = sd across the 4 OOD devices.  green = best.",
                   best_cells=best_cells, row_w=[1.9, 2.0, 2.2], label_w=1.6)


# ===========================================================================
# TABLE 4 -- every sklearn model on the OOD cross-device test (the coloured 15x5 grid). The report
# uses the headline_crossdevice_cv bar chart instead, so this grid is now exploratory.
# ===========================================================================
def table4(cdv):
    """Table 4 -- every sklearn model on the OOD cross-device test: 3 targets x devices x 5 models,
    spectral-17, read from cross_device.py's results CSV (the canonical cross-device engine).
    Rows are (target, device); columns are the five sklearn families (GNN excluded -- it is
    benchmarked separately, not part of the sklearn grid)."""
    sp = cdv[cdv.rep == "spectral"]
    tgts = ["route", "pol", "pol_z"]
    rows_lbl, cell, mat = [], [], []
    for t in tgts:
        for dev, short in zip(R.DEVICES, R.DEV_SHORT):
            vals = []
            for m in R.MODELS:
                s = sp[(sp.target == t) & (sp.device == dev) & (sp.model == m)]
                vals.append(float(s.R2.values[0]) if len(s) else np.nan)
            rows_lbl.append(f"{t} · {short}")
            cell.append([(f"{v:.2f}" if np.isfinite(v) else "—") for v in vals])
            mat.append(vals)
    best_cells = {(i, int(np.nanargmax(r))) for i, r in enumerate(mat) if np.any(np.isfinite(r))}
    R.render_table(R.MODELS, rows_lbl, cell, os.path.join(R.OUT_EXPLORATORY, "table4_ood_grid.png"),
                   "Table 4 — Every sklearn model on the OOD cross-device test (spectral-17)",
                   note=f"{R.ROUTE_DEF};  pol_z = within-N z-scored pol.  R²: train FakeBrisbane → test device.  "
                        "green = best (R²>0);  light-red = best-but-R²<0 (least-bad, not a win);  * = Brisbane self-transfer.",
                   best_cells=best_cells, neg_cells=R._neg_cells(mat), label_w=1.9)
    pd.DataFrame(mat, index=rows_lbl, columns=R.MODELS).to_csv(os.path.join(R.OUT_EXPLORATORY, "table4_ood_grid.csv"))


# ===========================================================================
# IN-DEPTH EVALUATION -- in-distribution feature importance + ablation on the BEST model (LASSO).
# Reproduce per-feature directly with the harness CLI:
#   python supervised_analysis_run.py --model lasso --features spectral --target route --coef / --ablate
# ===========================================================================
REP_NICE = {"spectral": "spectral-17", "mqt9": "MQT-9"}


def compute_lasso_importance(train, target="route", features="spectral", base=None, relative=False):
    """LASSO feature importance + ablation on a feature set (default spectral-17 = the best model;
    features='mqt9' produces the appendix MQT-9 version).
    importance = standardized lasso coefficients (per-SD effect; L1 zeros the unhelpful features,
                 via the harness diagnostics.model_importances);
    ablation   = CV R² lost when each feature is dropped (leave-one-feature-out, harness run_indist).
    `base`/`relative` follow the harness target encoding: pol_z is (base='pol', relative=True), so
    the z-stats are train-fold-fit (no normalization leakage) -- pass them, not the 'pol_z' token,
    to the harness primitives. `target` is kept only for labelling/filenames.
    Returns (df sorted by |coef|, full CV R²). Nothing hard-coded."""
    import diagnostics                                # harness leaf module
    base = base or target                            # route/pol: base == token; pol_z: base='pol'
    feats = R._rep_shared_cols(features, train)       # spectral-17 or local MQT-9 columns
    X = train[feats].fillna(0).values
    y = R.resolve_target(train, base, np.ones(len(train), bool), relative)
    pipe = R.H._fit_full("lasso", X, y, train.grp.values)
    _label, _names, coef = diagnostics.model_importances(pipe, feats)   # standardized coefs
    full = R.run_indist(train, X, base, "lasso", relative=relative)["R2"]
    rows = []
    for i, f in enumerate(feats):
        r2_drop = R.run_indist(train, np.delete(X, i, axis=1), base, "lasso", relative=relative)["R2"]
        rows.append(dict(feature=f, coef=float(coef[i]), abl_dR2=float(full - r2_drop),
                         selected=bool(abs(coef[i]) > 1e-6)))
    df = (pd.DataFrame(rows)
          .reindex(pd.DataFrame(rows).coef.abs().sort_values(ascending=False).index)
          .reset_index(drop=True))
    out = str(R.REPO / "data" / "results" / f"lasso_importance_{features}_{target}.csv")
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
    # target in the name so pol/pol_z don't clobber route; route keeps its historical filenames
    suffix = ("" if target == "route" else f"_{target}") + ("" if features == "spectral" else f"_{features}")
    R.render_table(cols, rlbl, cell, os.path.join(R.OUT_EXPLORATORY, f"table_lasso_importance{suffix}.png"),
                   f"Feature importance & ablation — LASSO ({target}, {nice})",
                   note=f"std. coef = per-SD lasso weight (grey = zeroed by L1);\n"
                        f"ablation ΔR² = CV R² lost dropping the feature;  full CV R² = {full_r2:.3f}.",
                   dim_cells=dim, row_w=[1.45, 1.65], label_w=2.9, fontsize=13, row_in=0.36)


def table_lasso_importance_combined(specs, path=None, fontsize=12, row_in=0.34):
    """Pack several LASSO importance/ablation tables (e.g. route spectral-17 | MQT-9) side-by-side
    in ONE figure: shared title + note, top-aligned, identical per-row height so the two feature
    reps line up. `specs` = list of (df, full_r2, target, features). Each sub-table keeps its own
    header and a per-table full-CV-R² caption. L1-zeroed rows are greyed, same as the singles."""
    label_in, data_ins = 2.8, [1.4, 1.55]
    table_w = label_in + sum(data_ins)
    gap_in, side_in = 0.5, 0.3
    n = len(specs)
    nmax = max(len(df) for df, *_ in specs)
    title_in, cap_in, note_in = 0.46, 0.30, 0.50
    fig_w = 2 * side_in + n * table_w + (n - 1) * gap_in
    fig_h = title_in + cap_in + (nmax + 1) * row_in + note_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    tgt = specs[0][2]
    reps = " vs ".join(REP_NICE.get(f, f) for *_, f in specs)
    fig.suptitle(f"Feature importance & ablation — LASSO ({tgt}): {reps}",
                 fontweight="bold", fontsize=13, y=1 - title_in / (2 * fig_h))
    body_top_in = fig_h - title_in - cap_in              # top edge of every table (top-aligned)
    for j, (df, full_r2, _target, features) in enumerate(specs):
        nrows = len(df)
        h_in = (nrows + 1) * row_in
        x0_in = side_in + j * (table_w + gap_in)
        y0_in = body_top_in - h_in
        ax = fig.add_axes([x0_in / fig_w, y0_in / fig_h, table_w / fig_w, h_in / fig_h])
        ax.axis("off")
        nice = REP_NICE.get(features, features)
        fig.text((x0_in + table_w / 2) / fig_w, (body_top_in + cap_in / 2) / fig_h,
                 f"{nice}  —  full CV R² = {full_r2:.3f}", ha="center", va="center",
                 fontweight="bold", fontsize=fontsize)
        full_cols = ["", "std. coef", "ablation ΔR²"]
        rows, dim = [], set()
        for i, (_, r) in enumerate(df.iterrows()):
            rows.append([r.feature, f"{r.coef:+.3f}", f"{r.abl_dR2:+.4f}"])
            if not r.selected:
                dim |= {(i, 0), (i, 1)}
        widths = [label_in / table_w] + [d / table_w for d in data_ins]
        tbl = ax.table(cellText=rows, colLabels=full_cols, cellLoc="center",
                       rowLoc="center", bbox=[0, 0, 1, 1])
        tbl.auto_set_font_size(False); tbl.set_fontsize(fontsize)
        for (rr, cc), cell in tbl.get_celld().items():
            cell.set_edgecolor("#bbbbbb"); cell.set_width(widths[cc])
            if rr == 0:
                cell.set_facecolor("#40466e")
                if cc > 0:
                    cell.set_text_props(color="w", fontweight="bold")
            elif cc == 0:
                cell.set_facecolor("#d9d9d9"); cell.set_text_props(fontweight="bold")
            elif (rr - 1, cc - 1) in dim:                # L1-zeroed feature: neutral grey
                cell.set_facecolor("#ededed")
    fig.text(0.5, note_in / (2 * fig_h),
             "std. coef = per-SD lasso weight (grey = zeroed by L1);   "
             "ablation ΔR² = CV R² lost dropping the feature.",
             ha="center", va="center", fontsize=9, style="italic")
    path = path or os.path.join(R.OUT_EXPLORATORY, "table_lasso_importance_combined.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"== combined LASSO importance/ablation ({tgt}: {reps}) -> {path}")


# ===========================================================================
# OPTION A -- a-priori (leakage-free) device-noise feature added to spectral-17
#
# Per circuit/device, from backend.target calibration ONLY (no routed_2q, no label transform):
#   exp_2qerr   = sum of the device's best-k 2q-gate errors, k = #interaction edges
#   meanbk_2qerr= exp_2qerr / k        med_2qerr = device median 2q error    mean_readout = device mean readout
# Train on Brisbane (Brisbane calibration), test each device's val_pol with ITS calibration.
# ===========================================================================
OPTIONA_CSV = str(R.REPO / "data" / "results" / "option_a.csv")


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
    R.assert_qir_version()                          # soft guard: live device params must match the label-gen snapshots
    feats = R.shared_feats("spectral", train)
    cal = {d: _device_cal(d) for d in R.DEVICES}
    frames = {d: R.load_external(str(R.LABELED / f"val_pol_{d}.csv")) for d in R.DEVICES}
    Xs_tr = train[feats].fillna(0).values
    Xa_tr = _expected_feats(train, cal[R.TRAIN_DEV])         # train rows use the TRAIN device's calibration
    grp = train.grp.values
    stats, gm, gs = R.fit_zstats(train["pol"].values, train.N.values)
    rows = []
    for tgt in ["pol", "pol_z"]:
        if tgt == "pol":
            y_tr = train["pol"].values.astype(float)
            y_of = {d: frames[d]["pol"].values.astype(float) for d in R.DEVICES}
        else:                                               # within-N z, train-fit (run_external recipe)
            y_tr = R.apply_zstats(train["pol"].values, train.N.values, stats, gm, gs)
            y_of = {d: R.apply_zstats(frames[d]["pol"].values, frames[d].N.values, stats, gm, gs)
                    for d in R.DEVICES}
        for use_A in (False, True):
            Xtr = np.column_stack([Xs_tr, Xa_tr]) if use_A else Xs_tr
            pipe = R.H._fit_full(model, Xtr, y_tr, grp)
            for d in R.DEVICES:
                Xs = frames[d][feats].fillna(0).values
                X = np.column_stack([Xs, _expected_feats(frames[d], cal[d])]) if use_A else Xs
                r2 = R.metrics_full(y_of[d], pipe.predict(X))["R2"]
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
    for d in R.DEVICES:
        r = []
        for tgt in ("pol", "pol_z"):
            for sch in ("blind", "+A"):
                v = df[(df.device == d) & (df.target == tgt) & (df.scheme == sch)].R2.values
                r.append(float(v[0]) if len(v) else np.nan)
        cell.append([f"{x:+.2f}" for x in r]); mat.append(r)
    arr = np.array(mat)
    best_cells = set()
    for i in range(len(R.DEVICES)):
        if np.isfinite(arr[i, 1]) and arr[i, 1] > arr[i, 0]:   # pol: +A beat blind
            best_cells.add((i, 1))
        if np.isfinite(arr[i, 3]) and arr[i, 3] > arr[i, 2]:   # pol_z: +A beat blind
            best_cells.add((i, 3))
    R.render_table(cols, R.DEV_SHORT, cell, os.path.join(R.OUT_EXPLORATORY, "table_option_a.png"),
                   f"Option A — a-priori device-noise feature added to spectral-17 ({model.upper()})",
                   note="blind = spectral-17;  +A = + 4 a-priori device-noise cal features.  "
                        "green = +A beats blind;  light-red = +A best but R²<0;  * = self-transfer.",
                   best_cells=best_cells, neg_cells=R._neg_cells(arr))
    df.to_csv(os.path.join(R.OUT_EXPLORATORY, "table_option_a.csv"), index=False)


# ===========================================================================
# PREDICTION-vs-TRUTH CHARTS -- per device, scatter of predicted vs true with a fitted regression
# line + the y=x identity. Train Brisbane (spectral-17), test each device's val_pol. target='pol'
# (raw) or 'pol_z' (within-N z, train-fit, the run_external recipe).
# ===========================================================================
_DEV_COLOR = {"FakeBrisbane": "#4C72B0", "FakeBerlin": "#DD8452", "FakeBoston": "#C44E52",
              "FakeSherbrooke": "#55A868", "FakeTorino": "#8172B3"}


def fig_pred_vs_truth(train, model="lasso", target="pol_z"):
    """Per-device predicted-vs-true scatter + regression line + y=x. Writes pred_vs_truth_<target>.png."""
    short = dict(zip(R.DEVICES, R.DEV_SHORT))
    feats = R.shared_feats("spectral", train)
    if target == "pol_z":
        stats, gm, gs = R.fit_zstats(train["pol"].values, train.N.values)
        ytr = R.apply_zstats(train["pol"].values, train.N.values, stats, gm, gs)
        ylab = "polarization z-score (within-N)"
    else:
        ytr = train["pol"].values.astype(float); ylab = "polarization"
    pipe = R.H._fit_full(model, train[feats].fillna(0).values, ytr, train.grp.values)
    panels = {}
    for d in R.DEVICES:
        ood = R.load_external(str(R.LABELED / f"val_pol_{d}.csv"))
        y = (R.apply_zstats(ood["pol"].values, ood.N.values, stats, gm, gs) if target == "pol_z"
             else ood["pol"].values.astype(float))
        panels[d] = (y, pipe.predict(ood[feats].fillna(0).values))
    lo = min(min(y.min(), p.min()) for y, p in panels.values())
    hi = max(max(y.max(), p.max()) for y, p in panels.values())
    pad = 0.05 * (hi - lo); ax_lo, ax_hi = lo - pad, hi + pad
    fig, axes = plt.subplots(1, len(R.DEVICES), figsize=(3.5 * len(R.DEVICES), 3.9),
                             sharex=True, sharey=True, constrained_layout=True)
    for ax, d in zip(axes, R.DEVICES):
        y, pred = panels[d]; mm = R.metrics_full(y, pred)
        ax.plot([ax_lo, ax_hi], [ax_lo, ax_hi], "k--", lw=1, alpha=.6, zorder=1, label="y = x")
        ax.scatter(y, pred, s=14, c=_DEV_COLOR[d], alpha=.5, edgecolor="none", zorder=2)
        m, b = np.polyfit(y, pred, 1)                    # least-squares regression line pred ~ y
        xs = np.array([ax_lo, ax_hi])
        ax.plot(xs, m * xs + b, c=_DEV_COLOR[d], lw=2, zorder=3, label=f"fit (slope {m:.2f})")
        ax.set_title(f"{short[d]}   R²={mm['R2']:.2f}  ρ={mm['Spearman']:.2f}", fontsize=11)
        ax.set_xlabel("true"); ax.set_xlim(ax_lo, ax_hi); ax.set_ylim(ax_lo, ax_hi)
        ax.grid(alpha=.25); ax.set_aspect("equal", adjustable="box")
        if d == R.DEVICES[0]:
            ax.legend(fontsize=8, loc="upper left", framealpha=.9)
    axes[0].set_ylabel(f"predicted ({ylab})")
    fig.suptitle(f"{model.upper()} prediction vs truth — {target}  (train Brisbane → test device; "
                 "regression line + y=x identity)", fontsize=12.5)
    fp = os.path.join(R.OUT_EXPLORATORY, f"pred_vs_truth_{target}.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {fp}")


# ===========================================================================
# FEW-SHOT AFFINE CALIBRATION  -- FEW-SHOT, NOT zero-shot
#
# Fit a 2-param affine map a*ŷ+b on K LABELED target circuits, evaluate on the held-out remainder.
# The K calibration circuits and the eval remainder are GROUP-DISJOINT; results are mean +- sd over
# draws. pol clips a*ŷ+b to [0,1] and also fits the physics model exp(alpha*routed_2q) for contrast;
# pol_z is a z-score, so it is NOT clipped and physics is not applied.
# ===========================================================================
KS_FEWSHOT = [5, 10, 20, 40]
FEWSHOT_SEEDS = 25
FEWSHOT_CSV = str(R.REPO / "data" / "results" / "few_shot_calibration.csv")


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
    feats = R.shared_feats("spectral", train)
    frames = {d: R.load_external(str(R.LABELED / f"val_pol_{d}.csv")) for d in R.DEVICES}
    stats, gm, gs = R.fit_zstats(train["pol"].values, train.N.values)
    rows = []
    for tgt in targets:
        if tgt == "pol":
            ytr = train["pol"].values.astype(float); clip = (0.0, 1.0); do_phys = True
            y_of = {d: frames[d]["pol"].values.astype(float) for d in R.DEVICES}
        else:                                            # pol_z: z-score, no clip, no physics
            ytr = R.apply_zstats(train["pol"].values, train.N.values, stats, gm, gs)
            clip = None; do_phys = False
            y_of = {d: R.apply_zstats(frames[d]["pol"].values, frames[d].N.values, stats, gm, gs)
                    for d in R.DEVICES}
        pipe = R.H._fit_full(model, train[feats].fillna(0).values, ytr, train.grp.values)
        for d in R.DEVICES:
            ood = frames[d]; y = y_of[d]; groups = ood.grp.values
            pred = pipe.predict(ood[feats].fillna(0).values)
            m = ood["routed_2q"].values.astype(float) if (do_phys and "routed_2q" in ood.columns) else None
            r2_raw = R.metrics_full(y, pred)["R2"]
            for K in KS_FEWSHOT:
                af, ph = [], []
                for s in range(FEWSHOT_SEEDS):
                    rng = np.random.default_rng(1000 * s + K + (0 if tgt == "pol" else 7))
                    cal, ev = _kshot_split(groups, K, rng)
                    if len(ev) < 5 or len(np.unique(pred[cal])) < 2:
                        continue
                    a, b = _fit_affine(pred[cal], y[cal])
                    af.append(R.metrics_full(y[ev], _apply_affine(pred[ev], a, b, clip))["R2"])
                    if do_phys and m is not None:
                        al = _fit_physics(pred[cal], m[cal], y[cal])
                        ph.append(R.metrics_full(y[ev], np.clip(pred[ev] * np.exp(al * m[ev]), 0, 1))["R2"])
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
        for d, short in zip(R.DEVICES, R.DEV_SHORT):
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
    R.render_table(cols, rlbl, cell, os.path.join(R.OUT_EXPLORATORY, "table_fewshot_calibration.png"),
                   "Few-shot affine calibration (LASSO) — K labeled target circuits recover the level",
                   note=f"FEW-SHOT: a·ŷ+b fit on K target circuits (group-disjoint fit/eval), mean±sd over "
                        f"{FEWSHOT_SEEDS} draws — NOT zero-shot (summary/T4 are label-free).  green = recovered R²>0.",
                   best_cells=best, neg_cells=R._neg_cells(arr), label_w=2.2, row_w=[1.5, 2.0, 2.0, 2.0])
    df.to_csv(os.path.join(R.OUT_EXPLORATORY, "table_fewshot_calibration.csv"), index=False)


def fig_fewshot(df):
    """Learning curves: held-out R² vs K, affine (+ physics for pol), per device, pol & pol_z."""
    tgts = ["pol", "pol_z"]
    fig, axes = plt.subplots(2, len(R.DEVICES), figsize=(2.6 * len(R.DEVICES), 6.0),
                             sharex=True, constrained_layout=True)
    for r, tgt in enumerate(tgts):
        for c, (d, short) in enumerate(zip(R.DEVICES, R.DEV_SHORT)):
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
    fp = os.path.join(R.OUT_EXPLORATORY, "few_shot_calibration.png")
    fig.savefig(fp, dpi=130); plt.close(fig)
    print(f"  wrote {fp}")


# ===========================================================================
def main():
    ok, have_cdv = R.preflight()
    if not ok:
        sys.exit(1)
    print("\nloading train set...")
    train = R.load_data(R.DEFAULT_SWAP_CSV, R.DEFAULT_POL_CSV)

    # --- overall pooled-OOD summary ---
    summary = compute_ood_summary(train)
    table_overall_summary(summary)

    # --- in-distribution LASSO feature importance & ablation ---
    imp_df, full_r2 = compute_lasso_importance(train, "route")           # spectral-17
    imp_m, full_m = compute_lasso_importance(train, "route", "mqt9")     # MQT-9 appendix
    imp_pol, full_pol = compute_lasso_importance(train, "pol")           # spectral-17, pol
    imp_pz, full_pz = compute_lasso_importance(train, "pol_z", base="pol", relative=True)  # pol_z
    table_lasso_importance(imp_df, full_r2, "route")
    table_lasso_importance(imp_m, full_m, "route", "mqt9")
    table_lasso_importance(imp_pol, full_pol, "pol")
    table_lasso_importance(imp_pz, full_pz, "pol_z")
    table_lasso_importance_combined(
        [(imp_df, full_r2, "route", "spectral"), (imp_m, full_m, "route", "mqt9")])

    # --- Table-4 OOD grid (coloured 15x5; report uses bars instead) ---
    cdv = R.load_cdv() if have_cdv else None
    if cdv is not None:
        table4(cdv)
    else:
        print("  [skip table4 grid: cross_device_results.csv absent]")

    # --- Option A device-noise recovery ---
    optA = compute_optionA(train, "lasso")
    if optA is not None:
        table_optionA(optA, "lasso")

    # --- few-shot affine calibration (table + curves) ---
    fewshot = compute_fewshot(train, "lasso")
    table_fewshot(fewshot)
    fig_fewshot(fewshot)

    # --- prediction-vs-truth scatters ---
    fig_pred_vs_truth(train, "lasso", "pol")
    fig_pred_vs_truth(train, "lasso", "pol_z")

    print("\nDONE. Exploratory artifacts in", R.OUT_EXPLORATORY)


if __name__ == "__main__":
    main()
