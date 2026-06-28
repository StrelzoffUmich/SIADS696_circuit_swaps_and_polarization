#!/usr/bin/env python3
"""
Integrated supervised harness: structure -> routing / polarization / relative-resilience.

One evaluation protocol shared across every model family, so all results are directly
comparable. Families: RF, HistGradientBoosting (boosted trees), RidgeCV, LassoCV, kNN
(all via a StandardScaler pipeline), plus Steven's GraphSAGE GNN via an adapter that
runs through the same splits and metrics (it consumes graphs, not a feature matrix, so
it is dispatched separately rather than through build_pipe).

Targets
-------
  route : log(bare_routed_2q)                -- routing overhead (the prediction task)
  pol   : mirror polarization                 -- absolute noise resilience
  z     : within-N z-scored polarization      -- relative resilience (Krystian's target),
          group mean/std fit on the TRAIN fold only (no normalization leakage)

  --relative wraps ANY base target (route/pol) in the same within-N, train-fit z-score,
  so `--target route --relative` is relative routing overhead and `--target pol
  --relative` is identical to `--target z`. The z-score statistics are always fit on the
  training rows and applied to the test rows (per-N mean/std, global-train fallback for
  unseen N) -- no normalization leakage in-distribution OR out-of-distribution.

Scaling
-------
StandardScaler for every scale-sensitive model (Ridge, Lasso, kNN) -- one scaler so model
differences are not confounded by preprocessing and Ridge/Lasso coefficients are comparable
(per-SD effects). Harmless for the trees (scale-invariant). The GNN self-standardizes.

Metrics
-------
R^2 (level), Pearson (linear assoc.), Spearman (rank), MAE (robust level), RMSE (level).
Report rank AND a level metric: they dissociate under distribution shift. MAE is the
recommended OOD level metric (RMSE/R^2 are outlier-sensitive and a few structurally-alien
generators dominate them); RMSE/R^2 are reported alongside so the outlier sensitivity is
visible rather than hidden.

External OOD evaluation (train on ID, test on a held-out corpus)
---------------------------------------------------------------
`--ood-csv PATH` trains each model on the FULL in-distribution set (no CV) and evaluates on
an external, already-feature-extracted corpus (e.g. the mqt/nwq/qasmbench validation suite
built by build_validation_corpus_sv.py). This is the true leave-the-distribution test, as
opposed to `--mode ood` which holds out one family from the training data. The external
file is expected to be a single pre-merged dataset (features + bare_routed_2q [+ optional
polarization]); the z-score statistics are fit on training and applied to the OOD rows.
Polarization targets are skipped automatically if the corpus has no polarization column
(routing-only corpora are still fully evaluable for the routing task).

Protocol notes (learned the hard way; do not relax)
---------------------------------------------------
  * GROUPING: vqe_real_amp/vqe_two_local are byte-identical; the count-matched VQE trio
    shares interaction graphs and bare_routed_2q at each (N, seed). Splits group on
    (vqe3-collapsed, N, seed). ~0 effect in-distribution, ~0.11 on leave-one-family-out.
  * Never standardize by routed_2q (a mediator). Feature/target scaling is train-fold only.
  * Linear models are in-distribution instruments; they collapse OOD. Report them in-dist;
    let RF and the GNN carry the OOD/transfer table.

Usage
-----
  python supervised_analysis_run.py --model rf    --features spectral --target route
  python supervised_analysis_run.py --model gnn   --features spectral --target pol --mode ood
  python supervised_analysis_run.py --model ridge --target z --ci          # bootstrap 95% CIs
  python supervised_analysis_run.py --all --target route                   # every model + per-fold mean+-sd, time
  python supervised_analysis_run.py --all --target pol

  # model insight (run on your best model):
  python supervised_analysis_run.py --model ridge --coef       --target route   # native coefficients / tree importances
  python supervised_analysis_run.py --model ridge --importance --target route   # permutation importance (any sklearn model)
  python supervised_analysis_run.py --model ridge --ablate     --target route   # leave-one-feature-out dR2
  python supervised_analysis_run.py --model ridge --dump-predictions oof.csv --target route   # per-record residuals (failure analysis)

  # train on ID, evaluate on an external out-of-distribution corpus (--all = every model for --target):
  python supervised_analysis_run.py --ood-csv swap_run/full_dataset.csv --all --target route
  python supervised_analysis_run.py --ood-csv pol_run/full_dataset.csv  --all --target pol
"""
from __future__ import annotations
import argparse
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LassoCV, Lasso, RidgeCV, Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from pathlib import Path

# Repo layout:  repo/src/supervised/supervised_analysis_run.py , repo/data/datasets/...
# Defaults anchor to the script location (not the working directory) so the harness
# runs identically from any cwd. Override with --swap-csv / --pol-csv.
_REPO = Path(__file__).resolve().parent.parent.parent       # src/supervised/supervised_analysis_run.py -> repo/
_DEFAULT_DATA = _REPO / "data" / "datasets"
DEFAULT_SWAP_CSV = _DEFAULT_DATA / "train_swap_FakeBrisbane.csv"
DEFAULT_POL_CSV = _DEFAULT_DATA / "train_pol_FakeBrisbane.csv"

VQE_TRIO = {"vqe_real_amp", "vqe_two_local", "vqe_su2"}

# ----------------------------------------------------------------------------
FEATURE_SETS = {
    "size_only": ["num_qubits", "depth", "size", "num_2q_gates"],
    "basic": ["num_qubits", "depth", "size", "num_2q_gates", "graph_density",
              "gate_entropy", "num_unique_gates", "degree_variance",
              "avg_clustering", "assortativity"],
    "spectral": ["fiedler_topology", "laplacian_energy", "log_spanning_trees",
                 "log_estrada_index", "effective_resistance", "von_neumann_entropy",
                 "graph_density", "graph_diameter", "avg_clustering", "num_triangles",
                 "degree_variance", "assortativity", "critical_depth", "parallelism",
                 "program_communication", "twoq_temporal_locality", "time_to_connected"],
    "keep4": ["log_estrada_index", "fiedler_at_half_depth",
              "spectral_entropy_topology", "depth_per_qubit"],
    # Krystian's full set: every numeric feature minus IDs, targets, and post-routing
    # columns (routed_2q is dropped -- it is the mediator/leakage for polarization).
    "krystian": ["depth", "size", "num_2q_gates", "program_communication", "critical_depth",
                 "entanglement_ratio", "parallelism", "liveness", "fiedler_topology",
                 "spectral_entropy_topology", "laplacian_max_eig_topology",
                 "spectral_gap_ratio_topology", "effective_resistance", "log_spanning_trees",
                 "laplacian_energy", "von_neumann_entropy", "fiedler_2q_weighted",
                 "spectral_entropy_2q_weighted", "gini_2q_multiplicity", "edge_weight_mean_2q",
                 "log_estrada_index", "graph_density", "graph_diameter", "avg_clustering",
                 "max_degree", "degree_variance", "assortativity", "num_triangles",
                 "twoq_temporal_locality", "gate_entropy", "num_unique_gates", "depth_per_qubit",
                 "fiedler_at_half_depth", "time_to_connected", "has_2q_interactions",
                 "n_rotation_gates", "mean_abs_angle", "std_angle", "sum_sin_sq_half",
                 "mean_angle_squared", "angle_position_weighted"],
}

# sklearn families. GNN is handled separately (graph input). All share StandardScaler.
MODELS = {
    "rf":     lambda: RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=0),
    "histgb": lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                                    early_stopping=True, random_state=0),
    "ridge":  lambda: RidgeCV(alphas=np.logspace(-4, 3, 50), cv=5),
    "lasso":  lambda: make_lasso(),
    "knn":    lambda: KNeighborsRegressor(n_neighbors=7),
    "linear": lambda: LinearRegression(),
    "polyreg":lambda: LinearRegression()
}
SKLEARN = set(MODELS)
ALL_MODELS = list(MODELS) + ["gnn"]


def build_pipe(model_key):
    steps = []
   
    # for the polynomial regression model transform the original data into polynomial features first
    if model_key == "polyreg":
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))

    steps.append(("scale", StandardScaler()))
    steps.append(("model", MODELS[model_key]()))
    return Pipeline(steps)


def make_lasso():
    return GridSearchCV(
        estimator=Lasso(max_iter=20000),
        param_grid={"alpha": np.logspace(-3, 1, 50)},
        cv=GroupKFold(n_splits=5)
    )


def _fit_full(model_key, X, y, groups):
    """Fit a sklearn pipeline on ALL rows. lasso's inner GridSearchCV uses GroupKFold, so it
    needs the group labels threaded through (same as _fit_predict does per fold)."""
    pipe = build_pipe(model_key)
    if model_key == "lasso":
        pipe.fit(X, y, model__groups=groups)
    else:
        pipe.fit(X, y)
    return pipe


def feats_in(df, name):
    cols = [c for c in FEATURE_SETS[name] if c in df.columns]
    missing = [c for c in FEATURE_SETS[name] if c not in df.columns]
    if missing:
        print(f"  [warn] {name}: {len(missing)} columns absent, using {len(cols)}: {missing[:5]}")
    return cols


def shared_feats(name, *dfs):
    """Feature columns from set `name` present in EVERY dataframe (for train/OOD alignment),
    preserving FEATURE_SETS order so the train and test design matrices match column-for-column."""
    cols = [c for c in FEATURE_SETS[name] if all(c in d.columns for d in dfs)]
    missing = [c for c in FEATURE_SETS[name] if c not in cols]
    if missing:
        print(f"  [warn] {name}: {len(missing)} cols absent in train or OOD, using {len(cols)}: {missing[:5]}")
    return cols


def _log_route(x):
    """Routing target transform: log1p(x) = log(1 + bare_routed_2q).

    The data dictionary specifies log(bare_routed_2q), but that is undefined for the
    zero-routed circuits that appear in the corpus (a circuit with no 2-qubit gates has
    nothing to route -> bare_routed_2q == 0), and a cross-device test corpus on a different
    coupling map can produce them too. log1p is rank-IDENTICAL to log on the positive rows
    (Spearman unchanged) and corr(log1p, log) = 0.999 there, maps 0 -> 0 cleanly, and never
    raises -- so the same transform serves train and every test device. Counts are never
    negative; guard only against that genuinely-invalid case."""
    x = np.asarray(x, dtype=float)
    if np.any(x < 0):
        n = int((x < 0).sum())
        raise ValueError(f"routing target has {n} negative value(s); a routed 2q count "
                         "cannot be < 0 -- check the routing column.")
    return np.log1p(x)


def load_data(swap_csv, pol_csv):
    swap = pd.read_csv(swap_csv)
    pol = pd.read_csv(pol_csv)
    df = swap.merge(pol[["file", "polarization"]], on="file", how="inner")
    df["N"] = df.num_qubits.astype(int)
    df["seed"] = df.file.str.extract(r"_s(\d+)\.qasm").astype(int)
    df["unit"] = np.where(df.algo.isin({"vqe_real_amp", "vqe_two_local"}), "vqe_pair", df.algo)
    df["grp"] = (np.where(df.algo.isin(VQE_TRIO), "vqe3", df.algo)
                 + "_" + df.N.astype(str) + "_" + df.seed.astype(str))
    df["route"] = _log_route(df.bare_routed_2q)
    df["pol"] = df.polarization
    return df.reset_index(drop=True)


def load_external(csv, pol_col="polarization", route_col=None):
    """Load a single pre-merged external corpus (features + a routing column [+ polarization]).

    Mirrors load_data's derived columns (N, seed, unit, grp, route, pol) so the external
    rows flow through the SAME target/metric machinery. The build_validation_corpus_sv.py
    naming convention is {algo}_{level}_{target}_{N}_s{seed}.qasm, so `algo` is the canonical
    family, `target` is the source suite, and the within-group index comes from seed_idx
    (regex fallback). Note: a pre-existing `seed` column in the labeled corpus is the global
    labeling RNG seed (constant), NOT the instance index -- the group key is rebuilt from
    seed_idx so each circuit instance is its own bootstrap unit.

    Routing column: the model trains on `bare_routed_2q` (bare routing overhead). The
    polarization-labeling output instead carries `routed_2q` -- the SIMULATED mirror circuit's
    2q count (forward+inverse, ~2x bare). It is a valid routing-overhead target on its own but
    is NOT on the same scale as a bare-routed-trained model, so we prefer bare_routed_2q and
    warn loudly on the routed_2q fallback. has_route / has_pol gate which targets are runnable.
    """
    df = pd.read_csv(csv)
    if "num_qubits" not in df.columns:
        raise KeyError("external corpus missing 'num_qubits'")
    df["N"] = df.num_qubits.astype(int)
    if "seed_idx" in df.columns:                       # instance index (overrides any RNG-seed col)
        df["seed"] = df.seed_idx.astype(int)
    else:
        df["seed"] = df.file.str.extract(r"_s(\d+)\.qasm").astype(int)
    # source suite ('target' column from the build script) joins the group key so two suites
    # contributing the same (algo, N, seed) index are not collapsed into one bootstrap unit.
    # NB: build the key with pandas Series (str concat works); numpy <U arrays have no add ufunc.
    suite = df["target"].astype(str) if "target" in df.columns else pd.Series("", index=df.index)
    df["unit"] = np.where(df.algo.isin({"vqe_real_amp", "vqe_two_local"}), "vqe_pair", df.algo)
    algo_grp = pd.Series(np.where(df.algo.isin(VQE_TRIO), "vqe3", df.algo), index=df.index).astype(str)
    df["grp"] = algo_grp + "_" + suite + "_" + df.N.astype(str) + "_" + df.seed.astype(str)

    if route_col is None:                              # auto-detect, prefer the bare count
        if "bare_routed_2q" in df.columns:
            route_col = "bare_routed_2q"
        elif "routed_2q" in df.columns:
            route_col = "routed_2q"                     # mirror count; warn ONLY if route is used
    df.attrs["route_col"] = route_col
    df.attrs["route_fallback"] = (route_col is not None and route_col != "bare_routed_2q")
    df.attrs["has_route"] = route_col is not None
    df["route"] = _log_route(df[route_col].values) if route_col else np.nan

    df.attrs["has_pol"] = pol_col in df.columns
    df["pol"] = df[pol_col].values if pol_col in df.columns else np.nan
    return df.reset_index(drop=True)


# ---- targets --------------------------------------------------------------
def fit_zstats(y, N):
    """Per-N mean/std plus global fallback, fit on whatever rows are passed (the TRAIN rows)."""
    t = pd.DataFrame({"y": np.asarray(y, float), "N": np.asarray(N)})
    stats = t.groupby("N")["y"].agg(["mean", "std"])
    return stats, float(np.mean(y)), float(np.std(y))


def apply_zstats(y, N, stats, gmean, gstd):
    """Apply fitted per-N z-stats to arbitrary rows (TRAIN or TEST); global fallback for unseen N."""
    y = np.asarray(y, float); N = np.asarray(N)
    z = np.empty(len(y))
    for i in range(len(y)):
        n = N[i]
        if n in stats.index and stats.loc[n, "std"] > 1e-9:
            z[i] = (y[i] - stats.loc[n, "mean"]) / stats.loc[n, "std"]
        else:
            z[i] = (y[i] - gmean) / gstd if gstd > 1e-9 else 0.0
    return z


def zscore_train_stats(pol, N, train_mask):
    """Within-N z-score using TRAIN-fold group mean/std only. Test rows are normalized
    with the train statistics for their N (global train fallback if N unseen)."""
    stats, gmean, gstd = fit_zstats(pol[train_mask], N[train_mask])
    return apply_zstats(pol, N, stats, gmean, gstd)


def resolve_base_relative(target, relative):
    """Map the (target, --relative) pair onto (base_quantity, is_relative).
    'z' is the back-compat alias for relative polarization."""
    if target == "z":
        return "pol", True
    return target, bool(relative)


def resolve_target(df, target, train_mask, relative=False):
    """Return the target vector. Relative targets use train-fold stats (no leakage)."""
    base, rel = resolve_base_relative(target, relative)
    vals = df[base].values
    if rel:
        return zscore_train_stats(vals, df.N.values, train_mask)
    return vals


# ---- metrics --------------------------------------------------------------
def scores(y, p):
    """Back-compat 3-tuple (R2, Spearman, MAE) used by the in-distribution paths."""
    m = metrics_full(y, p)
    return m["R2"], m["Spearman"], m["MAE"]


def metrics_full(y, p):
    """Full metric set: R2, Pearson, Spearman, MAE, RMSE (all NaN-safe on degenerate folds)."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    ss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - p) ** 2).sum() / ss if ss > 1e-12 else float("nan")
    if len(y) > 2 and np.std(p) > 1e-12 and np.std(y) > 1e-12:
        pear = float(pearsonr(y, p)[0])
        rho = float(spearmanr(y, p).correlation)
    else:
        pear = rho = float("nan")
    mae = float(np.abs(y - p).mean())
    rmse = float(np.sqrt(((y - p) ** 2).mean()))
    return dict(R2=r2, Pearson=pear, Spearman=rho, MAE=mae, RMSE=rmse)


_METRIC_ORDER = ["R2", "Pearson", "Spearman", "MAE", "RMSE"]


# ---- GNN dispatch ---------------------------------------------------------
# Interaction-graph GraphSAGE (gnn_interaction): graphs built from the dataset's
# interaction_edges column -- no QASM, no external script, reproducible from full_dataset.
# Serves any target (route/pol/z); the per-fold target is written into graph.y here.


def _gnn_graphs(df):
    """Build (and cache) the interaction-graph list for df.

    Cache lives on df.attrs so its lifetime is tied to the DataFrame. The previous version
    keyed a module-global dict on id(df); across the cross-device loop a freed OOD frame's
    address gets reused by the next device's frame, so the cache could hand back the PREVIOUS
    device's graphs (silent wrong numbers). The graphs are device-INDEPENDENT (pre-routing
    topology + structural features); only graph.y changes per fold/device and the caller
    overwrites it, so caching the topology per-frame is both correct and cheap."""
    cached = df.attrs.get("_gnn_graphs")
    if cached is None:
        from gnn_interaction import build_graphs, GRAPH_FEATS
        missing = [c for c in (*GRAPH_FEATS, "interaction_edges") if c not in df.columns]
        if missing:
            print(f"WARNING: GNN graph inputs missing {missing} -- build_graphs zero-fills "
                  "these (getattr default 0.0), so the net silently sees degraded features. "
                  "Check the corpus feature columns match the training set.", file=sys.stderr)
        cached = build_graphs(df, "route")            # topology only; per-fold y set by caller
        df.attrs["_gnn_graphs"] = cached
    return cached


def _gnn_fit_predict(df, yall, target, train_idx, test_idx, seed=0):
    import torch
    from gnn_interaction import GNNRegressor
    graphs = _gnn_graphs(df)
    for i, g in enumerate(graphs):                    # set this fold's target
        g.y = torch.tensor([[float(yall[i])]], dtype=torch.float)
    tri, vai = next(GroupShuffleSplit(1, test_size=0.15, random_state=0)
                    .split(train_idx, groups=df.grp.values[train_idx]))
    torch.manual_seed(seed)                           # determinism: init/dropout/loader shuffle
    m = GNNRegressor().fit(graphs, train_idx[tri], train_idx[vai])
    return m.predict(graphs, test_idx)


def _gnn_fit_predict_external(train_df, ytr, ood_df, yood, seed=0):
    """GNN train-on-ID / predict-on-OOD adapter. Builds two graph lists (train + OOD), trains
    on an internal grouped train/val split, predicts on the OOD graphs.

    torch is seeded here so the whole training (weight init, dropout, DataLoader shuffle) is
    deterministic. This matters for cross-device: train_df is IDENTICAL across the per-device
    eval calls, so a fixed seed yields the SAME trained model for every test device -- i.e.
    'train once on Brisbane-ID, evaluate on each device's corpus', matching how the deterministic
    sklearn models already behave. Without it each device re-draws a random net and the
    cross-device delta is confounded with ~0.15-R2 init noise."""
    import torch
    from gnn_interaction import GNNRegressor
    g_tr = _gnn_graphs(train_df)
    g_ood = _gnn_graphs(ood_df)
    for i, g in enumerate(g_tr):
        g.y = torch.tensor([[float(ytr[i])]], dtype=torch.float)
    for i, g in enumerate(g_ood):
        g.y = torch.tensor([[float(yood[i])]], dtype=torch.float)
    idx = np.arange(len(g_tr))
    tri, vai = next(GroupShuffleSplit(1, test_size=0.15, random_state=0)
                    .split(idx, groups=train_df.grp.values))
    torch.manual_seed(seed)                           # see docstring: makes device rows comparable
    m = GNNRegressor().fit(g_tr, idx[tri], idx[vai])
    return m.predict(g_ood, np.arange(len(g_ood)))


# ---- evaluation -----------------------------------------------------------

def _fit_predict(df, X, target, model_key, train_idx, test_idx, relative=False):
    train_mask = np.zeros(len(df), bool)
    train_mask[train_idx] = True
    yall = resolve_target(df, target, train_mask, relative)

    if model_key == "gnn":
        pred = _gnn_fit_predict(df, yall, target, train_idx, test_idx)
    else:
        pipe = build_pipe(model_key)

        if model_key == "lasso":
            pipe.fit(
                X[train_idx],
                yall[train_idx],
                model__groups=df.grp.values[train_idx]
            )
        else:
            pipe.fit(X[train_idx], yall[train_idx])

        pred = pipe.predict(X[test_idx])

    return yall[test_idx], pred


def run_indist(df, X, target, model_key, n_splits=5, return_oof=False, relative=False):
    idx = np.arange(len(df))
    gkf = GroupKFold(n_splits=n_splits)
    per_fold = {k: [] for k in _METRIC_ORDER}             # collect EVERY metric per fold
    oof = np.full(len(df), np.nan)
    for tr, te in gkf.split(idx, groups=df.grp.values):
        yte, pred = _fit_predict(df, X, target, model_key, tr, te, relative)
        oof[te] = pred
        mm = metrics_full(yte, pred)
        for k in _METRIC_ORDER:
            per_fold[k].append(mm[k])
    # mean + across-fold sd for each metric; keys include R2/R2_sd/Spearman/MAE.
    summ = {}
    for k in _METRIC_ORDER:
        summ[k] = float(np.nanmean(per_fold[k]))
        summ[f"{k}_sd"] = float(np.nanstd(per_fold[k]))
    return (summ, oof) if return_oof else summ


def run_ood(df, X, target, model_key, relative=False):
    rows = []
    for unit in sorted(df.unit.unique()):
        te = np.where(df.unit.values == unit)[0]
        tr = np.where(df.unit.values != unit)[0]
        yte, pred = _fit_predict(df, X, target, model_key, tr, te, relative)
        r2, rho, mae = scores(yte, pred)
        rows.append(dict(held_out=unit, n=len(te), R2=r2, Spearman=rho, MAE=mae))
    return pd.DataFrame(rows)


def cluster_bootstrap(df, oof, target, B=3000, seed=0, relative=False):
    """95% CIs on in-distribution OOF metrics, resampling the group key (the right unit
    of independence -- not individual circuits)."""
    train_mask = np.ones(len(df), bool)
    y = resolve_target(df, target, train_mask, relative)
    groups = df.grp.values
    uniq = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in _METRIC_ORDER}
    for _ in range(B):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([gidx[g] for g in gs])
        m = ~np.isnan(oof[idx])
        if m.sum() < 5: continue
        mm = metrics_full(y[idx][m], oof[idx][m])
        for k in _METRIC_ORDER: acc[k].append(mm[k])
    def ci(a): return (np.nanpercentile(a, 2.5), np.nanpercentile(a, 97.5))
    return {k: ci(v) for k, v in acc.items()}


# ---- external OOD evaluation (train on ID, test on a held-out corpus) ------
def _bootstrap_groups(y, p, groups, B, seed):
    """Cluster bootstrap over OOD groups: resample groups with replacement, recompute every
    metric. Groups (not circuits) are the independence unit -- byte-identical structures that
    share an (algo, suite, N, seed) key move together."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    uniq = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in _METRIC_ORDER}
    for _ in range(B):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([gidx[g] for g in gs])
        if len(idx) < 5: continue
        mm = metrics_full(y[idx], p[idx])
        for k in _METRIC_ORDER: acc[k].append(mm[k])
    def ci(a): return (np.nanpercentile(a, 2.5), np.nanpercentile(a, 97.5))
    return {k: ci(v) for k, v in acc.items()}


def run_external(train_df, X_tr, ood_df, X_ood, target, model_key,
                 relative=False, B=2000, seed=0):
    """Train each model on the FULL in-distribution set, predict on the external corpus.
    Relative (within-N z) targets fit their stats on TRAIN and apply them to OOD rows, with a
    global-train fallback for any N not seen in training. Returns (point_metrics, cis, pred)."""
    base, rel = resolve_base_relative(target, relative)
    ytr_base = train_df[base].values
    yood_base = ood_df[base].values
    if rel:
        stats, gm, gs = fit_zstats(ytr_base, train_df.N.values)
        ytr = apply_zstats(ytr_base, train_df.N.values, stats, gm, gs)
        yood = apply_zstats(yood_base, ood_df.N.values, stats, gm, gs)
    else:
        ytr, yood = np.asarray(ytr_base, float), np.asarray(yood_base, float)

    if model_key == "gnn":
        pred = _gnn_fit_predict_external(train_df, ytr, ood_df, yood, seed=seed)
    else:
        pipe = _fit_full(model_key, X_tr, ytr, train_df.grp.values)
        pred = pipe.predict(X_ood)

    point = metrics_full(yood, pred)
    cis = _bootstrap_groups(yood, pred, ood_df.grp.values, B, seed)
    return point, cis, pred


def _ext_target_label(base, rel):
    return f"{base}{'·z' if rel else ''}"


def external_table(train_df, ood_df, features, models, targets, B=2000, seed=0):
    """Full OOD comparison: models x targets, point metrics + bracketed 95% CIs."""
    feats = shared_feats(features, train_df, ood_df)
    X_tr = train_df[feats].fillna(0).values
    X_ood = ood_df[feats].fillna(0).values
    has_pol = bool(ood_df.attrs.get("has_pol", False))
    has_route = bool(ood_df.attrs.get("has_route", False))

    hdr = f"{'model':<8}{'target':<10}" + "".join(f"{m:>22}" for m in _METRIC_ORDER)
    print(hdr); print("-" * len(hdr))
    for base, rel in targets:
        if base == "pol" and not has_pol:
            print(f"{'--':<8}{_ext_target_label(base, rel):<10}  [skipped: no polarization label in OOD corpus]")
            continue
        if base == "route" and not has_route:
            print(f"{'--':<8}{_ext_target_label(base, rel):<10}  [skipped: no routing column in OOD corpus]")
            continue
        for mk in models:
            try:
                pt, ci, _ = run_external(train_df, X_tr, ood_df, X_ood,
                                         base, mk, relative=rel, B=B, seed=seed)
                cells = "".join(f"{pt[m]:>8.3f} [{ci[m][0]:+.2f},{ci[m][1]:+.2f}]" for m in _METRIC_ORDER)
                print(f"{mk:<8}{_ext_target_label(base, rel):<10}{cells}")
            except Exception as e:
                print(f"{mk:<8}{_ext_target_label(base, rel):<10}  skipped ({type(e).__name__}: {e})")
        print()
    bases = {b for b, _ in targets}
    z_used = any(r for _, r in targets)
    print(f"n_train={len(train_df)}  n_ood={len(ood_df)}  features={features}({len(feats)}d)  "
          f"groups_ood={ood_df.grp.nunique()}  bootstrap={B}")
    print("CIs are cluster-bootstrap 95% over OOD groups."
          + ("  '·z' = within-N z-score, fit on TRAIN." if z_used else ""))
    if "route" in bases and has_route and ood_df.attrs.get("route_col") != "bare_routed_2q":
        print(f"NOTE: route target uses '{ood_df.attrs.get('route_col')}' (not bare_routed_2q) -> "
              "routing metrics are not comparable to a bare-routed-trained model.")
    if "pol" in bases and not has_pol:
        print("NOTE: OOD corpus has no polarization label -> polarization targets skipped. "
              "Run the noisy-sim labeling stage to populate a 'polarization' column.")


def run_external_single(train_df, ood_df, features, model_key, target, relative, B, seed):
    base, rel = resolve_base_relative(target, relative)
    has_pol = bool(ood_df.attrs.get("has_pol", False))
    has_route = bool(ood_df.attrs.get("has_route", False))
    if base == "pol" and not has_pol:
        print("OOD corpus has no polarization label; cannot evaluate the polarization target.")
        print("Run the noisy-sim labeling stage (populates 'polarization') or pass --target route.")
        return
    if base == "route" and not has_route:
        print("OOD corpus has no routing column; cannot evaluate the routing target.")
        return
    feats = shared_feats(features, train_df, ood_df)
    X_tr = train_df[feats].fillna(0).values
    X_ood = ood_df[feats].fillna(0).values
    pt, ci, _ = run_external(train_df, X_tr, ood_df, X_ood, base, model_key,
                             relative=rel, B=B, seed=seed)
    print(f"model={model_key} features={features}({len(feats)}d) target={_ext_target_label(base, rel)} "
          f"  TRAIN-on-ID / TEST-on-OOD")
    print(f"n_train={len(train_df)}  n_ood={len(ood_df)}  groups_ood={ood_df.grp.nunique()}  bootstrap={B}")
    for m in _METRIC_ORDER:
        lo, hi = ci[m]
        print(f"  {m:<9}{pt[m]:>9.4f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
    if base == "route" and ood_df.attrs.get("route_fallback"):
        print(f"NOTE: route target uses '{ood_df.attrs.get('route_col')}' (not bare_routed_2q) -> "
              "not comparable to a bare-routed-trained model.")


# ---- CLI ------------------------------------------------------------------
def main():
    import diagnostics                       # leaf helper: importance / ablation / runtime
    diagnostics.register_total_runtime()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--swap-csv", default=str(DEFAULT_SWAP_CSV),
                    help="SWAP-arm training set (default: data/datasets/train_swap_FakeBrisbane.csv)")
    ap.add_argument("--pol-csv", default=str(DEFAULT_POL_CSV),
                    help="fidelity-arm training set (default: data/datasets/train_pol_FakeBrisbane.csv)")
    ap.add_argument("--model", default="rf", choices=ALL_MODELS)
    ap.add_argument("--features", default="spectral", choices=list(FEATURE_SETS))
    ap.add_argument("--target", default="route", choices=["route", "pol", "z"])
    ap.add_argument("--relative", action="store_true",
                    help="within-N z-score the base target (train-fit). 'pol --relative' == 'z'.")
    ap.add_argument("--mode", default="indist", choices=["indist", "ood"])
    ap.add_argument("--ci", action="store_true", help="cluster-bootstrap 95%% CIs (in-distribution)")
    ap.add_argument("--coef", action="store_true",
                    help="native feature importances for --model (coefficients for linear models, "
                         "feature_importances_ for trees; not the GNN)")
    ap.add_argument("--ablate", action="store_true",
                    help="leave-one-feature-out ablation for --model: CV R2 lost when each "
                         "feature is dropped (any sklearn model)")
    ap.add_argument("--importance", action="store_true",
                    help="permutation importance for --model: held-out R2 lost when each feature "
                         "is shuffled, mean+-sd over shuffles (model-agnostic; not the GNN)")
    ap.add_argument("--dump-predictions", default=None, metavar="CSV",
                    help="write per-record out-of-fold predictions + residuals (sorted by "
                         "|residual|) to this CSV, for failure analysis (in-distribution mode)")
    ap.add_argument("--all", action="store_true",
                    help="run ALL models for the chosen --target (model-comparison table)")
    # external OOD evaluation
    ap.add_argument("--ood-csv", default=None,
                    help="external corpus (pre-merged features + bare_routed_2q [+ polarization]); "
                         "train on ID, evaluate here")
    ap.add_argument("--ood-pol-col", default="polarization",
                    help="name of the polarization column in the OOD corpus (if present)")
    ap.add_argument("--ood-route-col", default=None,
                    help="routing column in the OOD corpus (default: auto -- bare_routed_2q, "
                         "else routed_2q with a comparability warning)")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap reps for OOD CIs")
    args = ap.parse_args()

    df = load_data(args.swap_csv, args.pol_csv)

    # ---- external OOD evaluation branch -----------------------------------
    if args.ood_csv:
        ood = load_external(args.ood_csv, args.ood_pol_col, args.ood_route_col)
        if args.all:                                   # --all == every MODEL, for --target
            base, rel = resolve_base_relative(args.target, args.relative)
            external_table(df, ood, args.features, list(MODELS), [(base, rel)], B=args.boot)
            print("\nGNN: add `--model gnn` (single-model; slow; needs gnn_interaction).")
        else:
            run_external_single(df, ood, args.features, args.model,
                                args.target, args.relative, args.boot, seed=0)
        return

    if args.all:                                       # --all == every MODEL, for --target
        base, rel = resolve_base_relative(args.target, args.relative)
        feats = feats_in(df, args.features); X = df[feats].fillna(0).values
        tlabel = _ext_target_label(base, rel)
        print(f"{'model':<8}{'target':<9}{'R2(mean+-sd)':>18}{'Spearman(+-sd)':>20}{'MAE':>8}{'time_s':>9}  "
              f"(in-distribution 5-fold, features={args.features})")
        for mk in MODELS:                              # sklearn families; GNN on demand
            try:
                s, dt = diagnostics.timed(run_indist, df, X, base, mk, relative=rel)
                print(f"{mk:<8}{tlabel:<9}"
                      f"{s['R2']:>10.3f}+-{s['R2_sd']:<5.3f}"
                      f"{s['Spearman']:>12.3f}+-{s['Spearman_sd']:<5.3f}"
                      f"{s['MAE']:>8.3f}{dt:>9.2f}")
            except Exception as e:
                print(f"{mk:<8}{tlabel:<9}  skipped ({type(e).__name__})")
        print(f"\nGNN: add `--model gnn --target {args.target}` (slow but self-contained).")
        return

    feats = feats_in(df, args.features); X = df[feats].fillna(0).values
    print(f"model={args.model} features={args.features}({len(feats)}d) "
          f"target={args.target} relative={args.relative} mode={args.mode} n={len(df)}")

    if args.coef:
        diagnostics.report_coef(df, X, feats, args.target, args.model, args.relative,
                                resolve_target=resolve_target, fit_full=_fit_full)
        return

    if args.ablate:
        diagnostics.report_ablation(df, X, feats, args.target, args.model, args.relative,
                                    run_indist=run_indist)
        return

    if args.importance:
        diagnostics.report_permutation(df, X, feats, args.target, args.model, args.relative,
                                       resolve_target=resolve_target, fit_full=_fit_full)
        return

    if args.mode == "ood":
        out = run_ood(df, X, args.target, args.model, args.relative)
        print(out.to_string(index=False))
        print(f"\nmean: R2={out.R2.mean():.3f}  Spearman={out.Spearman.mean():.3f}  MAE={out.MAE.mean():.3f}")
        return

    (summ, oof), dt = diagnostics.timed(run_indist, df, X, args.target, args.model,
                                        return_oof=True, relative=args.relative)
    print(f"in-distribution 5-fold (mean+-sd over folds): "
          f"R2={summ['R2']:.4f}+-{summ['R2_sd']:.4f}  "
          f"Spearman={summ['Spearman']:.4f}+-{summ['Spearman_sd']:.4f}  "
          f"MAE={summ['MAE']:.4f}+-{summ['MAE_sd']:.4f}  ({dt:.2f}s)")
    if args.ci:
        c = cluster_bootstrap(df, oof, args.target, relative=args.relative)
        print(f"  95% CI (cluster bootstrap over groups):")
        for m in _METRIC_ORDER:
            print(f"    {m:<9}[{c[m][0]:+.3f}, {c[m][1]:+.3f}]")
    if args.dump_predictions:
        tm = np.ones(len(df), bool)
        ytrue = resolve_target(df, args.target, tm, args.relative)
        keep = [c for c in ("file", "algo", "N", "n_qubits", "grp") if c in df.columns]
        dump = df[keep].copy()
        dump["y_true"] = ytrue
        dump["y_pred"] = oof
        dump["residual"] = oof - ytrue
        dump["abs_residual"] = np.abs(dump["residual"])
        dump = dump.sort_values("abs_residual", ascending=False, na_position="last")
        dump.to_csv(args.dump_predictions, index=False)
        print(f"  wrote {int(np.isfinite(oof).sum())} per-record predictions "
              f"(sorted by |residual|) -> {args.dump_predictions}")


if __name__ == "__main__":
    main()
