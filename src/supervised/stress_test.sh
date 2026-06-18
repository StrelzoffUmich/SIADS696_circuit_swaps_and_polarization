#!/usr/bin/env bash
# stress_test.sh -- verify supervised_analysis_run.py + cross_device.py run clean against the repo layout.
#
# Run it from anywhere; it anchors all paths to its own location:
#   bash src/supervised/stress_test.sh
#
# Only the two hosted training CSVs are required; TIER 0 GENERATES the rest (the OOD
# validation corpus and the per-device labels), so a clean clone is self-sufficient:
#   src/supervised/   supervised_analysis_run.py  cross_device.py  gnn_interaction.py  build_validation_corpus_sv.py
#   src/mqtloader/    run_pipeline.py + stages            (labels the corpus in TIER 0 / 6)
#   data/datasets/    train_swap_FakeBrisbane.csv  train_pol_FakeBrisbane.csv   (TRAIN set, hosted)
#   data/corpora/validation_qasm/*.qasm        <- BUILT by TIER 0 (mqt.bench/nwq/qasmbench; needs network)
#   data/xdev_out/labeled/val_{swap,pol}_FakeBrisbane.csv  <- BUILT by TIER 0 (MQT_Loader labeling)
#
# Env knobs:  SKIP_GEN=1  skip TIER 0 (tiers 3/5/6 then self-skip)    REGEN=1  force corpus rebuild
#             GNN tiers (4-6) need torch + torch_geometric; they self-skip if absent.
#
# Core sklearn tiers (1-3) must all pass. GNN (4) and cross-device (5-6) self-skip only if
# their prerequisites are absent (e.g. SKIP_GEN=1, or no torch), and print why.

set -euo pipefail
export PYTHONUNBUFFERED=1

# ---- anchor every path to this script's location (cwd-independent) ----------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../src/supervised
REPO="$(cd "$HERE/../.." && pwd)"                       # repo root
DATA="$REPO/data"
PY=${PYTHON:-python}
MQT=${MQT:-$REPO/src/mqtloader}
CORPUS=${CORPUS:-$DATA/corpora/validation_qasm}
LABELED=${LABELED:-$DATA/xdev_out/labeled}
cd "$HERE"   # supervised_analysis_run.py / cross_device.py / gnn_interaction.py live here

trap 'echo; echo ">>> STRESS TEST FAILED (line $LINENO). See the command above."; exit 1' ERR

say(){ printf '\n========== %s ==========\n' "$*"; }
run(){ printf '\n$ %s\n' "$*"; "$@"; }

# ---- preflight: the files the harness can't run without --------------------
for f in supervised_analysis_run.py cross_device.py gnn_interaction.py; do
  [ -e "$HERE/$f" ] || { echo "MISSING required script: $HERE/$f"; exit 1; }
done
for f in train_swap_FakeBrisbane.csv train_pol_FakeBrisbane.csv; do
  [ -e "$DATA/datasets/$f" ] || { echo "MISSING training set: $DATA/datasets/$f"; exit 1; }
done
$PY -c "import ast; ast.parse(open('supervised_analysis_run.py').read()); ast.parse(open('cross_device.py').read()); print('syntax OK')"

# ============================================================================
# TIER 0 generates the regenerable inputs the cross-device tiers consume: the OOD
# validation corpus (build_validation_corpus_sv.py) and the per-device labels
# (cross_device.py --phase label). Everything here derives from the two hosted
# training CSVs + public benchmark suites, so a clean clone is self-sufficient.
if [ "${SKIP_GEN:-0}" = "1" ]; then
  say "TIER 0 -- input generation SKIPPED (SKIP_GEN=1); data-dependent tiers self-skip"
else
  say "TIER 0 -- generate inputs: OOD validation corpus + FakeBrisbane labels (both arms)"
  # (a) build the flat OOD corpus if absent (pulls mqt.bench / nwqbench / qasmbench; needs network)
  if [ "${REGEN:-0}" = "1" ] || [ -z "$(ls -A "$CORPUS"/*.qasm 2>/dev/null)" ]; then
    run $PY build_validation_corpus_sv.py --out-dir "$CORPUS"
  else
    echo "corpus present ($(ls "$CORPUS"/*.qasm 2>/dev/null | wc -l | tr -d ' ') circuits) -> reusing (REGEN=1 to rebuild)"
  fi
  # (b) label both arms on FakeBrisbane -> val_{swap,pol}_FakeBrisbane.csv (cached; existing skipped)
  run $PY cross_device.py --phase label --devices FakeBrisbane --targets route pol \
      --corpus "$CORPUS" --mqt "$MQT"
fi

# ============================================================================
say "TIER 1 -- harness in-distribution (GroupKFold), every target x model path"
# --swap-csv/--pol-csv default to data/datasets/train_{swap,pol}_FakeBrisbane.csv
run $PY supervised_analysis_run.py --target route --mode indist --model ridge
run $PY supervised_analysis_run.py --target pol   --mode indist --model ridge
run $PY supervised_analysis_run.py --target z     --mode indist --model ridge
run $PY supervised_analysis_run.py --target route --mode indist --all
run $PY supervised_analysis_run.py --target pol   --mode indist --all
run $PY supervised_analysis_run.py --target z     --mode indist --all
run $PY supervised_analysis_run.py --target route --mode indist --model ridge --ci --coef       # CIs + standardized coefs
run $PY supervised_analysis_run.py --target pol   --mode indist --model ridge --relative        # relative wrapper on a base target

say "TIER 1b -- feature-set switches (every key in FEATURE_SETS)"
for FS in size_only basic keep4 spectral krystian; do
  run $PY supervised_analysis_run.py --target route --mode indist --model ridge --features "$FS"
done

# ============================================================================
say "TIER 2 -- harness leave-one-family-out OOD"
run $PY supervised_analysis_run.py --target route --mode ood --all
run $PY supervised_analysis_run.py --target pol   --mode ood --all
run $PY supervised_analysis_run.py --target z     --mode ood --model ridge

# ============================================================================
say "TIER 3 -- harness external OOD (train ID / test an external labeled CSV)"
OOD_SWAP=${OOD_SWAP:-$LABELED/val_swap_FakeBrisbane.csv}   # produced by TIER 0
OOD_POL=${OOD_POL:-$LABELED/val_pol_FakeBrisbane.csv}     # produced by TIER 0
if [ -f "$OOD_SWAP" ]; then
  run $PY supervised_analysis_run.py --ood-csv "$OOD_SWAP" --target route --all --ood-route-col bare_routed_2q
else
  echo "SKIP: no $OOD_SWAP (copy a swap-arm labeled CSV in, or set OOD_SWAP=...) "
fi
if [ -f "$OOD_POL" ]; then
  run $PY supervised_analysis_run.py --ood-csv "$OOD_POL" --target pol --all
  run $PY supervised_analysis_run.py --ood-csv "$OOD_POL" --target z   --model ridge
else
  echo "SKIP: no $OOD_POL (copy a fidelity-arm labeled CSV in, or set OOD_POL=...)"
fi

# ============================================================================
say "TIER 4 -- GNN paths (import-sensitive: needs gnn_interaction + torch + torch_geometric)"
if $PY -c "import torch, torch_geometric" 2>/dev/null; then
  run $PY supervised_analysis_run.py --target route --mode indist --model gnn
  run $PY supervised_analysis_run.py --target route --mode ood    --model gnn
  [ -f "$OOD_SWAP" ] && run $PY supervised_analysis_run.py --ood-csv "$OOD_SWAP" --target route --model gnn --ood-route-col bare_routed_2q \
    || echo "SKIP external GNN: no $OOD_SWAP"
else
  echo "SKIP TIER 4: torch / torch_geometric not importable in this env."
fi

# ============================================================================
say "TIER 5 -- cross_device EVAL-only (reuses cached labels; no MQT_Loader needed)"
if ls "$LABELED"/*_swap.csv >/dev/null 2>&1; then
  run $PY cross_device.py --phase eval --devices FakeBrisbane --targets route pol pol_z \
      --corpus "$CORPUS" --models ridge gnn --seeds 0 1
else
  echo "SKIP TIER 5: no $LABELED/*.csv (run TIER 6 first, or copy a labeled dir in)."
fi

# ============================================================================
say "TIER 6 -- cross_device FULL label+eval, quick route-only smoke (needs MQT_Loader)"
if [ -d "$MQT" ] && [ -d "$CORPUS" ]; then
  run $PY cross_device.py --phase all --devices FakeBrisbane --targets route \
      --corpus "$CORPUS" --models ridge gnn --mqt "$MQT" --force
else
  echo "SKIP TIER 6: need MQT_Loader at '$MQT' (set MQT=...) and corpus at '$CORPUS'."
fi

echo; echo ">>> STRESS TEST PASSED (all present tiers green)."
