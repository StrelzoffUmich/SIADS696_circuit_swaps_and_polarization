#!/usr/bin/env python3
"""
cross_device.py -- automate routing/polarization performance polling across X arbitrary devices.

The validation corpus is device-AGNOSTIC QASM (built once). The device enters only when
MQT_Loader routes/labels that corpus. So a cross-device study is:

    build_validation_corpus_sv.py  ->  one QASM corpus  (device-independent)
        |
        |  for each device X:
        v
    run_pipeline.py --skip-gen --qasm-dir <corpus> --device X [--no-naiveMP]
        ->  per-device labels:  swap arm  -> bare_routed_2q   (routing)
                                fidelity  -> polarization     (resilience)
        |
        v
    supervised_analysis_run.py  train on the (Brisbane) in-distribution set, TEST on device X's labels
        ->  R2 / Pearson / Spearman / MAE / RMSE  + cluster-bootstrap 95% CIs, per model

This script drives all three and aggregates into one cross-device table. It does NOT modify
the harness -- it imports and reuses it, so the evaluation protocol is identical to a manual
`supervised_analysis_run.py --ood-csv <device file> --all --target {route,pol}` run.

Phases (separable -- labeling is heavy, evaluation is cheap):
    --phase label   run MQT_Loader for each (device, arm). Cached: existing outputs skipped.
    --phase eval    train-on-ID / test-on-device with the harness; build the results table.
    --phase all     both (default).

Training is the in-distribution corpus (Brisbane by default via --swap-train/--pol-train);
only the OOD *test* device varies. Point --swap-train/--pol-train at device-specific training
data if you want a full train-device x test-device matrix instead of Brisbane-transfer.

Examples
--------
  # defaults resolve to the repo layout: --mqt <repo>/src/mqtloader,
  # --corpus <repo>/data/corpora/validation_qasm, train sets in <repo>/data/datasets,
  # --out <repo>/data/xdev_out. Override any of them; the examples below use defaults.

  # dry-run: print the MQT_Loader commands without executing the heavy sim
  python cross_device.py --devices FakeBrisbane FakeTorino FakeSherbrooke --dry-run

  # label three devices, then evaluate, writing cross_device_results.csv
  python cross_device.py --devices FakeBrisbane FakeTorino FakeSherbrooke

  # re-evaluate only (labels already exist), add the within-N z (relative) targets
  python cross_device.py --devices FakeBrisbane FakeTorino --phase eval \
      --targets route pol pol_z
"""
from __future__ import annotations
import argparse, glob, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

# Layout-aware defaults, resolved from THIS script's location so they hold regardless of cwd.
# Expected repo layout:  repo/src/supervised/cross_device.py , repo/src/mqtloader , repo/data
_SCRIPT = Path(__file__).resolve()
_REPO = _SCRIPT.parent.parent.parent                       # src/supervised/cross_device.py -> repo/
DEFAULT_MQT = _REPO / "src" / "mqtloader"
DEFAULT_DATA = _REPO / "data"
DEFAULT_CORPUS = DEFAULT_DATA / "corpora" / "validation_qasm"
DEFAULT_SWAP_TRAIN = DEFAULT_DATA / "datasets" / "train_swap_FakeBrisbane.csv"
DEFAULT_POL_TRAIN = DEFAULT_DATA / "datasets" / "train_pol_FakeBrisbane.csv"
DEFAULT_XDEV_OUT = DEFAULT_DATA / "xdev_out"

# target token -> (harness base target, within-N z?)
TARGET_MAP = {
    "route":   ("route", False),
    "route_z": ("route", True),
    "pol":     ("pol",   False),
    "pol_z":   ("pol",   True),
}
ARM_OF_BASE = {"route": "swap", "pol": "pol"}   # which MQT_Loader arm produces each target


def import_harness(harness_path):
    import importlib.util
    p = Path(harness_path).resolve()
    if not p.exists():
        sys.exit(f"harness not found at {p}")
    spec = importlib.util.spec_from_file_location("harness", str(p))
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    return h


# ---- labeling (drives MQT_Loader/run_pipeline.py) -------------------------
def build_corpus(builder, corpus_dir, sources, n_lo, n_hi, dedupe, dry_run):
    """Build the device-agnostic QASM corpus once (skipped if it already exists)."""
    corpus = Path(corpus_dir)
    if corpus.exists() and any(corpus.glob("*.qasm")):
        print(f"[corpus] reusing existing {corpus} ({len(list(corpus.glob('*.qasm')))} qasm files)")
        return
    cmd = [sys.executable, str(builder), "--out-dir", str(corpus),
           "--n-lo", str(n_lo), "--n-hi", str(n_hi), "--sources", *sources]
    if dedupe:
        cmd.append("--dedupe-graphs")
    print(f"[corpus] {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)


def resolve_corpus(corpus):
    """Return the dir to hand MQT_Loader as --qasm-dir. NO staging, NO copying.

    All three stages read circuits straight from the dir you pass them:
      - extract_features.py:            <dir>/*.qasm  (falls back to flat when there's no qasm/ subdir)
      - swap_features.py:               <dir>/*.qasm
      - mirror_polarization_label.py:   <dir>/*.qasm
    So a FLAT corpus (exactly what build_validation_corpus_sv.py writes) already satisfies all
    three. We just resolve to an absolute path and sanity-check it holds circuits. If the
    circuits happen to live in a <corpus>/qasm subdir instead, we point at that subdir (which
    is itself flat) — same contract. Either way the returned dir contains *.qasm directly.

    (The old stage_corpus copied the flat files into <work>/corpus_staged/qasm/, which ONLY
    extract_features descends into — the labeler/swapper read the empty parent and saw 0
    circuits. That self-inflicted layout split was the entire cross-device bug.)"""
    corpus = Path(corpus).resolve()
    if any(corpus.glob("*.qasm")):
        chosen = corpus
    elif (corpus / "qasm").is_dir() and any((corpus / "qasm").glob("*.qasm")):
        chosen = corpus / "qasm"
    else:
        sys.exit(f"no .qasm files in {corpus} (or {corpus}/qasm) — build the corpus first "
                 f"(build_validation_corpus_sv.py --out-dir {corpus})")
    n = len(list(chosen.glob("*.qasm")))
    print(f"[corpus] {n} circuits at {chosen} — passed straight to MQT_Loader (no staging)")
    return chosen


def label_device(mqt_root, corpus_dir, device, arm, n_lo, n_hi, work_dir, dry_run, force):
    """Run one MQT_Loader pass (one device, one arm) on the validation corpus; return the CSV.

    Tags the run as role=val so run_pipeline names its output val_<arm>_<device>.csv inside a
    val__<arm>__<device>__n*__<ts>/ dir (with a run_manifest.json). We copy that to
    <work>/labeled/val_<arm>_<device>.csv and cache it."""
    ds_name = f"val_{arm}_{device}.csv"
    out_csv = Path(work_dir) / "labeled" / ds_name
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists() and not force:
        print(f"[label] {device}/{arm}: cached -> {out_csv}")
        return out_csv

    run_pipeline = Path(mqt_root).resolve() / "run_pipeline.py"
    if not run_pipeline.exists():
        sys.exit(f"run_pipeline.py not found under --mqt ({run_pipeline})")
    results_dir = (Path(work_dir).resolve() / "_runs" / f"{device}_{arm}")
    cmd = [sys.executable, str(run_pipeline),
           "--skip-gen", "--qasm-dir", str(Path(corpus_dir).resolve()),
           "--role", "val", "--device", device, "--n-lo", str(n_lo), "--n-hi", str(n_hi),
           "--results-dir", str(results_dir)]
    if arm == "swap":
        cmd.append("--no-naiveMP")
    print(f"[label] {device}/{arm}: {' '.join(cmd)}")
    if dry_run:
        return out_csv  # path is where it WOULD land

    if results_dir.exists():
        shutil.rmtree(results_dir)
    subprocess.run(cmd, cwd=str(Path(mqt_root).resolve()), check=True)
    hits = sorted(glob.glob(str(results_dir / f"val__{arm}__{device}__*" / ds_name)))
    if not hits:                                   # naming/results-dir override -> search under it
        hits = sorted(glob.glob(str(results_dir / "**" / ds_name), recursive=True))
    if not hits:                                   # legacy fallback (old full_dataset.csv naming)
        hits = sorted(glob.glob(str(results_dir / "**" / "full_dataset.csv"), recursive=True))
    if not hits:
        print(f"[label] {device}/{arm}: WARNING no dataset produced; skipping")
        return None
    shutil.copy(hits[-1], out_csv)
    print(f"[label] {device}/{arm}: -> {out_csv}")
    return out_csv


# ---- evaluation (reuses the frozen harness) -------------------------------
STOCHASTIC_MODELS = {"gnn"}   # re-seeding only changes these; sklearn fits are deterministic


def eval_device(h, train_df, ood_csv, base, relative, features, models, B, pol_col, route_col,
                seeds=(0,)):
    """One device, one target: train on ID, test on the device's OOD labels. Returns metric rows.

    seeds: for STOCHASTIC models (the GNN) the model is retrained at each seed and every metric
    is reported as mean +/- sd across seeds (the {m}_sd columns), isolating init/training variance
    from the bootstrap sampling CI ({m}_lo/{m}_hi, taken from the first seed as a representative
    fit). Deterministic sklearn models ignore the sweep -- re-seeding can't move them -- so they
    run once (n_seeds=1, every _sd=0)."""
    ood = h.load_external(str(ood_csv), pol_col, route_col)
    if base == "pol" and not ood.attrs.get("has_pol"):
        return [{"note": "no polarization label in OOD file"}]
    if base == "route" and not ood.attrs.get("has_route"):
        return [{"note": "no routing column in OOD file"}]
    feats = h.shared_feats(features, train_df, ood)
    X_tr = train_df[feats].fillna(0).values
    X_ood = ood[feats].fillna(0).values
    rows = []
    for mk in models:
        try:
            use_seeds = list(seeds) if mk in STOCHASTIC_MODELS else [seeds[0]]
            pts, ci0 = [], None
            for sd in use_seeds:
                pt, ci, _ = h.run_external(train_df, X_tr, ood, X_ood, base, mk,
                                           relative=relative, B=B, seed=sd)
                pts.append(pt)
                if ci0 is None:
                    ci0 = ci                                  # representative sampling CI (1st seed)
            row = {"model": mk, "n_ood": len(ood), "groups_ood": int(ood.grp.nunique()),
                   "route_col": ood.attrs.get("route_col"), "n_seeds": len(use_seeds)}
            for m in h._METRIC_ORDER:
                vals = np.array([p[m] for p in pts], dtype=float)
                row[m] = float(vals.mean())
                row[f"{m}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                row[f"{m}_lo"], row[f"{m}_hi"] = ci0[m]
            rows.append(row)
        except Exception as e:
            rows.append({"model": mk, "note": f"{type(e).__name__}: {e}"})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", nargs="+", required=True, help="Fake* backend names")
    ap.add_argument("--targets", nargs="+", default=["route", "pol"],
                    choices=list(TARGET_MAP), help="which analyses to poll")
    ap.add_argument("--phase", default="all", choices=["all", "label", "eval"])
    # paths (defaults resolve to the repo layout: src/mqtloader, data/...; override freely)
    ap.add_argument("--mqt", default=str(DEFAULT_MQT),
                    help="dir containing run_pipeline.py (default: <repo>/src/mqtloader)")
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS),
                    help="device-agnostic QASM corpus dir (default: <repo>/data/corpora/validation_qasm)")
    ap.add_argument("--harness", default=str(_SCRIPT.parent / "supervised_analysis_run.py"))
    ap.add_argument("--swap-train", default=str(DEFAULT_SWAP_TRAIN),
                    help="ID training: SWAP arm (default: data/datasets/train_swap_FakeBrisbane.csv)")
    ap.add_argument("--pol-train", default=str(DEFAULT_POL_TRAIN),
                    help="ID training: fidelity arm (default: data/datasets/train_pol_FakeBrisbane.csv)")
    ap.add_argument("--out", default=str(DEFAULT_XDEV_OUT),
                    help="output dir for labeled CSVs + results table (default: data/xdev_out)")
    # corpus build (only if --corpus is missing)
    ap.add_argument("--build-corpus", action="store_true")
    ap.add_argument("--builder", default=str(_SCRIPT.parent / "build_validation_corpus_sv.py"))
    ap.add_argument("--corpus-sources", nargs="+", default=["mqtbench", "nwqbench", "qasmbench"])
    ap.add_argument("--corpus-n-lo", type=int, default=3)
    ap.add_argument("--corpus-n-hi", type=int, default=15)
    ap.add_argument("--dedupe-graphs", action="store_true")
    # labeling N ranges (swap arm runs full N; fidelity arm is sim-bound to low N)
    ap.add_argument("--swap-n-lo", type=int, default=3)
    ap.add_argument("--swap-n-hi", type=int, default=15)
    ap.add_argument("--pol-n-lo", type=int, default=3)
    ap.add_argument("--pol-n-hi", type=int, default=10)
    # evaluation knobs
    ap.add_argument("--features", default="spectral")
    ap.add_argument("--models", nargs="+", default=["rf", "histgb", "ridge", "lasso", "knn"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0],
                    help="seed sweep for STOCHASTIC models (gnn): trains once per seed and reports "
                         "each metric as mean +/- sd across seeds. Deterministic sklearn models run "
                         "once regardless. e.g. --seeds 0 1 2 3 4")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--ood-pol-col", default="polarization")
    ap.add_argument("--ood-route-col", default=None)
    ap.add_argument("--force", action="store_true", help="re-label even if cached output exists")
    ap.add_argument("--dry-run", action="store_true", help="print MQT_Loader commands; don't run")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    bases_needed = {TARGET_MAP[t][0] for t in args.targets}     # route -> swap arm, pol -> fidelity arm
    arms_needed = {ARM_OF_BASE[b] for b in bases_needed}

    # ---- LABEL phase ------------------------------------------------------
    labeled = {}   # (device, arm) -> csv path
    if args.phase in ("all", "label"):
        if args.build_corpus:
            build_corpus(args.builder, args.corpus, args.corpus_sources,
                         args.corpus_n_lo, args.corpus_n_hi, args.dedupe_graphs, args.dry_run)
        qasm_dir = resolve_corpus(args.corpus)      # pass the flat corpus straight through
        for device in args.devices:
            for arm in sorted(arms_needed):
                n_lo = args.swap_n_lo if arm == "swap" else args.pol_n_lo
                n_hi = args.swap_n_hi if arm == "swap" else args.pol_n_hi
                p = label_device(args.mqt, qasm_dir, device, arm, n_lo, n_hi,
                                 out, args.dry_run, args.force)
                if p:
                    labeled[(device, arm)] = p
        if args.phase == "label":
            print(f"\n[done] labeling complete. Outputs under {out/'labeled'}/")
            return
        if args.dry_run:
            print("\n[dry-run] no labels produced; stopping before eval.")
            return

    # ---- EVAL phase -------------------------------------------------------
    h = import_harness(args.harness)
    for f in (args.swap_train, args.pol_train):
        if not Path(f).exists():
            sys.exit(f"training file not found: {f}")
    train_df = h.load_data(args.swap_train, args.pol_train)
    print(f"[train] ID set: n={len(train_df)} (swap={args.swap_train}, pol={args.pol_train})")

    all_rows = []
    for device in args.devices:
        for tok in args.targets:
            base, rel = TARGET_MAP[tok]
            arm = ARM_OF_BASE[base]
            ood_csv = labeled.get((device, arm)) or (out / "labeled" / f"val_{arm}_{device}.csv")
            if not Path(ood_csv).exists():
                print(f"[eval] {device}/{tok}: missing labels ({ood_csv}); run --phase label first")
                continue
            try:
                nrows = len(pd.read_csv(ood_csv))
            except Exception:
                nrows = 0
            if nrows == 0:
                print(f"[eval] {device}/{tok}: labeled file has 0 rows ({ood_csv}) -- "
                      "labeling produced no circuits; skipping (check the label stage)")
                continue
            rows = eval_device(h, train_df, ood_csv, base, rel, args.features,
                               args.models, args.boot, args.ood_pol_col, args.ood_route_col,
                               seeds=args.seeds)
            for r in rows:
                r0 = {"device": device, "target": tok}
                r0.update(r)
                all_rows.append(r0)

    if not all_rows:
        print("[eval] nothing evaluated. Did you label the devices? (--phase label)")
        return
    res = pd.DataFrame(all_rows)
    res_path = out / "cross_device_results.csv"
    res.to_csv(res_path, index=False)

    # compact printed summary: one line per device x target x model with point + R2/Spearman CI
    print(f"\n{'device':<16}{'target':<8}{'model':<8}"
          f"{'R2':>8}{'Pearson':>9}{'Spearman':>10}{'MAE':>8}{'RMSE':>8}")
    print("-" * 73)
    for _, r in res.iterrows():
        if "note" in r and isinstance(r.get("note"), str):
            print(f"{r['device']:<16}{r['target']:<8}{r.get('model',''):<8}  ({r['note']})")
            continue
        suffix = ""
        if int(r.get("n_seeds", 1) or 1) > 1:
            suffix = (f"   [{int(r['n_seeds'])} seeds; "
                      f"R2±{r.get('R2_sd', 0):.3f}, Sp±{r.get('Spearman_sd', 0):.3f}]")
        print(f"{r['device']:<16}{r['target']:<8}{r['model']:<8}"
              f"{r['R2']:>8.3f}{r['Pearson']:>9.3f}{r['Spearman']:>10.3f}{r['MAE']:>8.3f}{r['RMSE']:>8.3f}"
              f"{suffix}")
    print(f"\nfull table (per-metric mean, seed sd, and 95% bootstrap CI) -> {res_path}")


if __name__ == "__main__":
    main()
