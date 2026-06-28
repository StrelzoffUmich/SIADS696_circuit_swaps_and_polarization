#!/usr/bin/env python3
"""
Auxiliary diagnostics for the harness's sklearn models, split out of supervised_analysis_run.py for clarity:

  * feature importance  -- native (--coef) and permutation (--importance)
  * ablation            -- leave-one-feature-out CV dR2 (--ablate)
  * runtime             -- the total-run timer and per-call timing helper

These are reporting/analysis side-channels, not part of the core eval protocol, so they live
here instead of cluttering the harness. This module is a LEAF: it imports nothing from harness.
The reporters take the harness's eval primitives (`resolve_target`, `_fit_full`, `run_indist`)
as keyword arguments, so there is no circular import (harness imports diagnostics, never the
reverse). The GNN is unsupported throughout -- it consumes graphs, not a feature matrix.
"""
from __future__ import annotations
import sys, time
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance


# ---- runtime ---------------------------------------------------------------
def register_total_runtime(stream=sys.stderr):
    """Print '[runtime] total {s}s' at interpreter exit. Call once at the top of main()."""
    import atexit
    t0 = time.perf_counter()
    atexit.register(lambda: print(f"[runtime] total {time.perf_counter() - t0:.2f}s", file=stream))


def timed(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) and return (result, elapsed_seconds)."""
    t = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t


# ---- native importances ----------------------------------------------------
def model_importances(pipe, feats):
    """Native per-feature importances for the best (non-GNN) model: standardized coefficients
    for the linear models (ridge/lasso/linear/polyreg -- features are StandardScaler'd, so |coef|
    is a per-SD effect), or feature_importances_ for the trees (rf/histgb). Returns
    (label, names, values) or None if the model exposes neither (kNN -- use ablation instead).
    GridSearchCV (lasso) is unwrapped to best_estimator_; polyreg names are the degree-2 expansion."""
    est = pipe.named_steps["model"]
    inner = getattr(est, "best_estimator_", est)          # unwrap GridSearchCV (lasso)
    names = (list(pipe.named_steps["poly"].get_feature_names_out(feats))
             if "poly" in pipe.named_steps else list(feats))
    if hasattr(inner, "coef_"):
        return "Standardized coefficients (per-SD effect)", names, np.ravel(inner.coef_)
    if hasattr(inner, "feature_importances_"):
        return "Tree feature importances", names, np.asarray(inner.feature_importances_, float)
    return None


# ---- reporters (compute + print; mirror the --coef/--ablate/--importance branches) ----
def report_coef(df, X, feats, target, model_key, relative, *, resolve_target, fit_full):
    """--coef: native coefficients (linear) or feature_importances_ (trees) for one model."""
    if model_key == "gnn":
        print("--coef: not available for the GNN (it consumes graphs, not a feature matrix).")
        return
    tm = np.ones(len(df), bool); y = resolve_target(df, target, tm, relative)
    pipe = fit_full(model_key, X, y, df.grp.values)
    imp = model_importances(pipe, feats)
    if imp is None:
        print(f"--coef: {model_key} exposes no native coefficients/importances "
              f"(e.g. kNN); use --ablate for a model-agnostic importance view.")
        return
    label, names, vals = imp
    order = np.argsort(np.abs(vals))[::-1]
    print(f"\n{label} ({model_key}, target={target}):")
    for i in order:
        if abs(vals[i]) > 1e-6:
            print(f"  {names[i]:<30}{vals[i]:+.4f}")
    print(f"\n{int((np.abs(vals) > 1e-6).sum())}/{len(vals)} features nonzero")


def report_ablation(df, X, feats, target, model_key, relative, *, run_indist):
    """--ablate: leave-one-feature-out CV R2 loss for one model (any sklearn family)."""
    if model_key == "gnn":
        print("--ablate operates on the feature matrix; the GNN consumes graphs. "
              "Use a sklearn model.")
        return
    full = run_indist(df, X, target, model_key, relative=relative)["R2"]
    rows = []
    for i, f in enumerate(feats):
        r2_i = run_indist(df, np.delete(X, i, axis=1), target, model_key,
                          relative=relative)["R2"]
        rows.append((f, r2_i, full - r2_i))
    print(f"\nLeave-one-feature-out ablation ({model_key}, target={target}, "
          f"full CV R2={full:.3f}):")
    print(f"  {'dropped feature':<30}{'R2_without':>11}{'dR2':>9}")
    for f, r2_i, d in sorted(rows, key=lambda t: t[2], reverse=True):
        print(f"  {f:<30}{r2_i:>11.3f}{d:>+9.3f}")
    print("  dR2 = CV R2 lost by removing the feature (positive = the feature helps).")


def report_permutation(df, X, feats, target, model_key, relative, *,
                       resolve_target, fit_full, n_repeats=20, seed=0):
    """--importance: model-agnostic permutation importance (drop in held-out R2 when each INPUT
    feature is shuffled). Fit on a group-fold's train, scored on its test; mean +/- sd over
    n_repeats. Works for every sklearn family (incl. kNN, and polyreg -- permuting a raw input
    column captures its total effect through the degree-2 expansion)."""
    if model_key == "gnn":
        print("--importance: permutation is over the feature matrix; the GNN consumes graphs.")
        return
    gkf = GroupKFold(n_splits=5)
    tr, te = next(gkf.split(np.arange(len(df)), groups=df.grp.values))
    tm = np.zeros(len(df), bool); tm[tr] = True
    y = resolve_target(df, target, tm, relative)
    pipe = fit_full(model_key, X[tr], y[tr], df.grp.values[tr])    # handles lasso's group CV
    r = permutation_importance(pipe, X[te], y[te], n_repeats=n_repeats,
                               random_state=seed, scoring="r2")
    order = np.argsort(r.importances_mean)[::-1]
    print(f"\nPermutation importance ({model_key}, target={target}; "
          f"held-out R2 lost when a feature is shuffled):")
    print(f"  {'feature':<30}{'perm_dR2':>10}{'+-sd':>9}")
    for i in order:
        print(f"  {feats[i]:<30}{float(r.importances_mean[i]):>+10.4f}"
              f"{float(r.importances_std[i]):>9.4f}")
