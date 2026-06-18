#!/usr/bin/env python3
"""
run_pipeline.py - one-command runner for the mirror-polarization / SWAP dataset.

The development of this pipeline was assisted with Claude Opus 4.7/4.8. 
All code was smoke tested by the developer and manually documented

Runs the four stages in order, from the reorganized folder layout:
    load_qasm/loader_v2.py                    (0) generate QASM corpus
    helpers/extract_features.py               (1) pre-routing features  -> features.csv
    post_transpile/mirror_..._label  (2) post-routing labels   -> labels.csv
    build_dataset.py                          (3) join on 'file'        -> full_dataset.csv

The scripts live in different subfolders and import each other (loader_v2, frf,
features); this runner puts every code folder on PYTHONPATH so those imports resolve
regardless of the reorganized layout, then invokes each step as a subprocess.

EXAMPLE (smoke test: qaoa, N=3-5, clean from scratch):
    python run_pipeline.py --algorithms qaoa --qubits 3 5 --k 5 \
        --n-lo 3 --n-hi 5 --clean

EXAMPLE (full fidelity-arm run):
    python run_pipeline.py --qubits 3 20 --k 25 --n-lo 3 --n-hi 5

Skip stages you've already done with --skip-gen / --skip-features / --skip-labels.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# every folder that contains an importable module (loader_v2, frf, features, ...)
CODE_DIRS = [
    ROOT,
    ROOT / "load_qasm",
    ROOT / "helpers",
    ROOT / "post_transpile",
]

# canonical script locations within the reorganized layout
LOADER   = ROOT / "load_qasm" / "loader_v2.py"
FEATURES = ROOT / "helpers" / "extract_features.py"
LABELER  = ROOT / "post_transpile" / "mirror_polarization_label.py"
SWAPPER  = ROOT / "post_transpile" / "swap_features.py"
JOINER   = ROOT / "build_dataset.py"


def _env():
    """PYTHONPATH = all code dirs, so cross-folder imports (loader_v2, frf,
    features) resolve no matter which subfolder a script sits in."""
    env = os.environ.copy()
    extra = os.pathsep.join(str(d) for d in CODE_DIRS)
    env["PYTHONPATH"] = extra + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def run(step_name, cmd):
    print(f"\n{'='*70}\n[{step_name}] {' '.join(str(c) for c in cmd)}\n{'='*70}", flush=True)
    r = subprocess.run([sys.executable, *map(str, cmd)], env=_env(), cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(f"[{step_name}] FAILED (exit {r.returncode}) - stopping pipeline.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # generation
    ap.add_argument("--algorithms", nargs="+", default=None,
                    help="families to generate (default: all)")
    ap.add_argument("--qubits", nargs=2, type=int, default=[3, 20],
                    metavar=("LO", "HI"), help="qubit range for generation")
    ap.add_argument("--k", type=int, default=25, help="seeds per (family,N) cell")
    # paths
    ap.add_argument("--qasm-dir", default="qasm", help="corpus dir (created by gen)")
    ap.add_argument("--features-csv", default="features.csv")
    ap.add_argument("--labels-csv", default="polarization_labels.csv")
    ap.add_argument("--out", default="full_dataset.csv")
    # labeling window / knobs
    ap.add_argument("--n-lo", type=int, default=3, help="min N to LABEL")
    ap.add_argument("--n-hi", type=int, default=6, help="max N to LABEL")
    ap.add_argument("--families", default="", help="comma list to label (default all present)")
    ap.add_argument("--device", default="FakeBrisbane",
                    help="fake backend(s) for routing+noise; comma-separated for a "
                         "multi-device run (e.g. FakeBrisbane,FakeBoston). Each device "
                         "labels every circuit; rows are stamped with a 'device' column.")
    ap.add_argument("--no-mp", "--swap-only", "--no-naiveMP", dest="no_mp",
                    action="store_true",
                    help="SWAP-ARM mode: skip the naive mirror-polarization SIMULATION "
                         "stage and run the transpile-only SWAP pass (swap_features.py) "
                         "instead. Records routing overhead (bare/mirror routed_2q, "
                         "depth) with NO noisy simulation, so it runs fast over the FULL "
                         "N range. For analysis that needs routing cost (swap insertion) "
                         "but not noise resilience. Omit the flag for the fidelity arm "
                         "(the mirror-polarization label, which IS the naive U.U^-1 "
                         "mirror scored against |0...0>).")
    # stage control
    ap.add_argument("--clean", action="store_true",
                    help="delete qasm/ and generated CSVs first (full fresh run)")
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--skip-features", action="store_true")
    ap.add_argument("--skip-labels", action="store_true")
    ap.add_argument("--results-dir", default="results",
                    help="parent folder for run outputs (default: results/)")
    ap.add_argument("--role", default="train", choices=["train", "val"],
                    help="run role: 'train' (in-distribution training set) or 'val' "
                         "(out-of-distribution validation corpus). Drives the run-dir / "
                         "dataset naming and the manifest.")
    ap.add_argument("--no-timestamp", action="store_true",
                    help="write outputs to ROOT (legacy behavior) instead of a "
                         "context-named results/<role>__<arm>__<device>__n*__<ts>/ subfolder")
    args = ap.parse_args()

    import datetime, json
    # CONTEXT-AWARE NAMING: every run is tagged by role x arm x device x N-range, so a run
    # dir (and the dataset it produces) is self-describing on disk. arm comes from --no-mp
    # (swap = routing-only, pol = fidelity/sim); device from --device; role from --role.
    arm = "swap" if args.no_mp else "pol"
    devs = [d.strip() for d in args.device.split(",") if d.strip()]
    devslug = devs[0] if len(devs) == 1 else "+".join(devs)
    nlo, nhi = args.n_lo, args.n_hi
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.no_timestamp:
        run_dir = ROOT
    else:
        slug = f"{args.role}__{arm}__{devslug}__n{nlo}-{nhi}__{stamp}"
        run_dir = ROOT / args.results_dir / slug
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[results] writing this run's outputs to {run_dir}")

    qasm = ROOT / args.qasm_dir                  # corpus stays at ROOT (shared/reused)
    feats = run_dir / args.features_csv          # per-run outputs
    # stage-2 output name reflects WHAT it contains: routing counts (swap arm) vs
    # mirror-polarization labels (fidelity arm). Honor an explicit --labels-csv; otherwise
    # use the arm-tagged default instead of the misleading "polarization_labels.csv".
    if args.labels_csv == "polarization_labels.csv":
        labels = run_dir / f"{arm}_labels.csv"
    else:
        labels = run_dir / args.labels_csv
    # joined dataset: honor an explicit --out, else name it {role}_{arm}_{device}.csv so it
    # can be promoted to data/datasets/ as-is and read unambiguously by the harness.
    dataset_name = args.out if args.out != "full_dataset.csv" else f"{args.role}_{arm}_{devslug}.csv"
    out = run_dir / dataset_name

    if args.clean:
        print("[clean] removing qasm/ and generated CSVs")
        if qasm.exists():
            shutil.rmtree(qasm)
        for p in (feats, labels, out):
            if p.exists():
                p.unlink()

    # (0) generate
    if not args.skip_gen:
        cmd = [LOADER, "--qubits", args.qubits[0], args.qubits[1],
               "--k", args.k, "--out", qasm]
        if args.algorithms:
            cmd += ["--algorithms", *args.algorithms]
        run("0/gen", cmd)
    else:
        print("[0/gen] skipped")

    # (1) pre-routing features — write DIRECTLY into the run dir via --out
    # (no more write-to-root-then-copy, which left an orphan features.csv at root)
    if not args.skip_features:
        run("1/features", [FEATURES, "--corpus", qasm, "--out", feats])
    else:
        print("[1/features] skipped")

    # (2) post-routing stage — fidelity labels (sim) OR swap-arm routing counts (no sim)
    if not args.skip_labels:
        devices = [d.strip() for d in args.device.split(",") if d.strip()]
        if args.no_mp:
            # SWAP ARM: transpile-only routing counts, full N range, no simulation
            for dev_name in devices:
                cmd = [SWAPPER, qasm, labels, "--n-lo", args.n_lo, "--n-hi", args.n_hi,
                       "--device", dev_name]
                if args.families:
                    cmd += ["--families", args.families]
                run(f"2/swap [{dev_name}]", cmd)
        else:
            # FIDELITY ARM: mirror-polarization labels via noisy simulation
            for dev_name in devices:
                cmd = [LABELER, qasm, labels, "--n-lo", args.n_lo, "--n-hi", args.n_hi,
                       "--device", dev_name]
                if args.families:
                    cmd += ["--families", args.families]
                run(f"2/labels [{dev_name}]", cmd)
    else:
        print("[2/labels] skipped")

    # (3) stitch
    cmd = [JOINER, feats, labels, out]
    run("3/join", cmd)

    # ---- run_manifest.json: self-documenting context for outsiders --------------
    # so anyone opening the dataset knows role/arm/device/N, the target column, and exactly
    # how the harness target is derived from it -- without reading any code.
    try:
        n_rows = max(0, sum(1 for _ in open(out)) - 1)        # minus header
    except OSError:
        n_rows = None
    try:
        git_commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                    cwd=str(ROOT), capture_output=True, text=True).stdout.strip() or None
    except Exception:
        git_commit = None
    if arm == "swap":
        target_col = "bare_routed_2q"
        target_note = ("routing overhead. harness target 'route' = log1p(bare_routed_2q); "
                       "deterministic transpile, no simulation, full N range.")
    else:
        target_col = "polarization"
        target_note = ("mirror-circuit polarization in [0,1] (1 = perfect, 0 = noise). harness "
                       "target 'pol' = raw; 'z'/'pol_z' = within-N z-score (train-fit). sim-bound, low N.")
    manifest = {
        "generated_by": "run_pipeline.py",
        "role": args.role,                       # train | val
        "arm": arm,                              # swap | pol
        "device": devs if len(devs) > 1 else devslug,
        "n_lo": nlo, "n_hi": nhi,
        "qubits_generated": list(args.qubits),
        "k_per_cell": args.k,
        "families": (args.algorithms if args.algorithms else "all_structural"),
        "target_column": target_col,
        "target_note": target_note,
        "dataset": out.name,
        "features_csv": feats.name,
        "labels_csv": labels.name,
        "n_rows": n_rows,
        "timestamp": stamp,
        "git_commit": git_commit,
    }
    man_path = run_dir / "run_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n{'='*70}\nPIPELINE COMPLETE ({args.role}/{arm}/{devslug}, n={n_rows}) -> {out}"
          f"\n  manifest: {man_path}\n{'='*70}")


if __name__ == "__main__":
    main()
