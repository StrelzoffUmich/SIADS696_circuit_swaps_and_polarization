# MQTHarness

This supervised analysis pipeline was developed with assistance from Claude Opus 4.7/4.8.

A testbed for evaluating supervised models that predict post-routing quantities from
pre-routing circuit structure, on the QAOA / VQE (4 variants) / randomcircuit / graphstate corpus produced by MQT_Loader. 

It answers two questions — does structure predict routing overhead, and does it predict noise resilience — under three generalization protocols:
in-distribution 5-fold cross-validation, leave-one-family-out, and cross-device transfer.

It takes as input the joined `full_dataset.csv` files from MQT_Loader's two arms, promoted to
`data/datasets/train_swap_FakeBrisbane.csv` and `train_pol_FakeBrisbane.csv`, and, for the
cross-device protocol, a flat validation corpus produced by `build_validation_corpus_sv.py` plus
the MQT_Loader copy (`../mqtloader`) that labels it.

## Files

| file | role |
|------|------|
| `supervised_analysis_run.py` | the eval engine — CLI **and** importable library. In-distribution 5-fold CV, leave-one-family OOD, external-OOD. Paths anchored to the repo (`data/datasets/…`). |
| `diagnostics.py` | auxiliary reports split out of the harness: feature importance (`--coef` native, `--importance` permutation), leave-one-feature-out ablation (`--ablate`), and the runtime timer. Leaf module — takes harness primitives as args, imports nothing from harness. |
| `cross_device.py` | cross-device orchestrator. Labels the validation corpus on each device via `../mqtloader`, then imports `supervised_analysis_run.py` and runs its protocol. |
| `gnn_interaction.py` | the GNN row of the comparison (interaction-graph GraphSAGE). Imported by `supervised_analysis_run.py` when `--model gnn`. Dataset-sourced; no QASM round-trip. |
| `build_validation_corpus_sv.py` | builds the flat OOD corpus (mqtbench / nwqbench / qasmbench) into `data/corpora/validation_qasm/`. |
| `stress_test.sh` | tiered smoke test; verifies the above run clean against the repo layout. |
| `reference/` | **provenance, not wired in** — Steven's original standalone GraphSAGE + its `pull_gate_types` helper. See `reference/README.md`. |

Dependency graph (live pipeline):

```
cross_device.py ──imports──▶ supervised_analysis_run.py ──imports (when --model gnn)──▶ gnn_interaction.py
       │                          └──imports (--coef/--ablate/--importance)──▶ diagnostics.py
       │
       └──subprocess──▶ ../mqtloader/run_pipeline.py   (labeling)
build_validation_corpus_sv.py ──writes──▶ data/corpora/validation_qasm/   (consumed by cross_device.py)
reference/GraphSAGE-SWAP-and-Polarization.py ──imports──▶ reference/pull_gate_types.py   (standalone; nothing else touches it)
```

## What it does

Two scripts:

1. **`supervised_analysis_run.py`** — trains and evaluates models on a single labeled dataset under two
   protocols: **in-distribution 5-fold GroupKFold CV** (`--mode indist`) and
   **leave-one-family-out OOD** (`--mode ood`). It can also train on the full in-distribution set and test on an external labeled CSV (`--ood-csv`).
2. **`cross_device.py`** — the cross-device protocol. It labels a device-agnostic corpus on each requested backend (via MQT_Loader), trains on the in-distribution set, and tests on every device. It imports and reuses `supervised_analysis_run.py`, so the evaluation protocol is identical to a manual run.

All evaluation groups on the circuit-structure (the three fixed-graph VQE variants are collapsed to one group), so structurally identical seeds never split across train and test.

**Targets** (`--target`, or `--targets` in `cross_device.py`)
| token | meaning |
|-------|---------|
| `route` | routing overhead, `log1p(bare_routed_2q)` (SWAP arm) |
| `pol` | mirror-circuit polarization — noise resilience (fidelity arm) |
| `z` / `pol_z` | `pol` z-scored within each qubit count N (stats fit on train only) |

**Models** (`--model`, or `--models`): `rf`, `histgb`, `ridge`, `lasso`, `knn` (deterministic
sklearn) and `gnn` (interaction-graph GraphSAGE; stochastic — see note). `--all` runs every
sklearn model for the chosen target; the GNN is on-demand (name it explicitly).

**Feature sets** (`--features`, default `spectral`): `spectral` (17 pre-routing structural
features), `size_only`, `basic`, `keep4`, `krystian`.

## Quick start

```
# in-distribution 5-fold CV, all sklearn models
python supervised_analysis_run.py --target route --mode indist --all
python supervised_analysis_run.py --target pol   --mode indist --all
python supervised_analysis_run.py --target z     --mode indist --all

# leave-one-family-out OOD
python supervised_analysis_run.py --target route --mode ood --all

# cross-device: train in-distribution (Brisbane), test on any device(s)
# defaults resolve to the repo layout (src/mqtloader, data/corpora/validation_qasm); override freely
python cross_device.py --devices FakeBrisbane FakeBoston --targets route pol pol_z --models ridge histgb gnn --seeds 0 1 2 3 4
```

`supervised_analysis_run.py` reads `data/datasets/train_swap_FakeBrisbane.csv` /
`train_pol_FakeBrisbane.csv` by default (paths anchored to the script, not the cwd) and
prints a metric table. `cross_device.py` writes `data/xdev_out/cross_device_results.csv`.

### Arguments

**`supervised_analysis_run.py`**
| flag | default | meaning |
|------|---------|---------|
| `--target` | `route` | `route`, `pol`, or `z` (see targets) |
| `--mode` | `indist` | `indist` (5-fold GroupKFold CV) or `ood` (leave-one-family-out) |
| `--model` | `ridge` | one of `rf histgb ridge lasso knn gnn` |
| `--all` | off | run every sklearn model for the target (GNN excluded; name it with `--model gnn`) |
| `--features` | `spectral` | feature set (`spectral size_only basic keep4 krystian`) |
| `--relative` | off | z-score the base target within N (`--target pol --relative` == `--target z`) |
| `--ci` | off | add 95% cluster-bootstrap CIs (resampling groups) |
| `--coef` | off | print standardized coefficients (linear models) |
| `--ood-csv FILE` | — | train on the full in-distribution set, test on this external labeled CSV |
| `--ood-route-col` | auto | routing column in the OOD file (auto-detects `bare_routed_2q`) |
| `--ood-pol-col` | `polarization` | polarization column in the OOD file |
| `--swap-csv` / `--pol-csv` | `data/datasets/train_swap_FakeBrisbane.csv` / `train_pol_FakeBrisbane.csv` | training datasets (anchored to the script location) |
| `--boot` | `2000` | bootstrap resamples for CIs |

**`cross_device.py`**
| flag | default | meaning |
|------|---------|---------|
| `--devices` | required | `Fake<Name>` backends to test on (e.g. `FakeBrisbane FakeBoston`) |
| `--targets` | required | any of `route route_z pol pol_z` |
| `--corpus` | `validation_qasm` | flat OOD corpus (`*.qasm` directly in the dir) |
| `--models` | `rf histgb ridge lasso knn` | add `gnn` to include the GNN |
| `--seeds` | `0` | seed sweep for the GNN; reports each metric as mean ± seed-sd |
| `--phase` | `all` | `label`, `eval`, or `all` (eval reuses cached labels) |
| `--mqt` | `<repo>/src/mqtloader` | path to the MQT_Loader copy (label stage) |
| `--harness` | `<this dir>/supervised_analysis_run.py` | harness to import for the eval protocol |
| `--swap-train` / `--pol-train` | `data/datasets/train_swap_FakeBrisbane.csv` / `train_pol_FakeBrisbane.csv` | training (in-distribution) datasets |
| `--force` | off | relabel devices even if cached labels exist (`data/xdev_out/labeled/`) |
| `--swap-n-hi` / `--pol-n-hi` | `15` / `10` | max N to label per arm (fidelity arm is sim-bound) |

## The two arms

The harness evaluates two targets that come from two different MQT_Loader datasets sharing
one corpus:

**SWAP arm** — `--target route`, read from `train_swap_FakeBrisbane.csv` (`bare_routed_2q`). Routing
overhead is deterministic and noise-free, so this dataset spans the full N range.

**Fidelity arm** — `--target pol` / `z`, read from `train_pol_FakeBrisbane.csv` (`polarization`).
Resilience is only non-degenerate at low N (it floors by N≈6–10), so this dataset is
restricted accordingly.

One external OOD file therefore can't serve both targets: a routing test needs a swap-arm
file, a polarization test needs a fidelity-arm file. `cross_device.py` labels both arms per
device for this reason.

## Cross-device evaluation

`cross_device.py` trains on the in-distribution set (the two `data/datasets/train_*` files, which
are labeled on one device — Brisbane by default) and tests on each `--device`. The device
lives in the **labels**, not the features: pre-routing structure is device-independent and the
training set is fixed, so only the test device varies. Per-device labels are cached under
`data/xdev_out/labeled/`; pass `--force` to relabel.

## The GNN

`gnn` is the only stochastic model (weight init, dropout, and batch order). It is seeded for reproducibility, but a single run's R² has large seed variance. Run it over several seeds and report **mean ± sd**, never the best seed:

```
python cross_device.py --devices FakeBrisbane --targets route --models gnn --seeds 0 1 2 3 4
```

The deterministic sklearn models ignore the sweep (one run is their operating point). The GNN
rows require `torch` and `torch_geometric`; if absent, use the sklearn models only.

**Two GNN implementations.** `gnn_interaction.py` (above) is the one the harness runs: it builds
graphs from the dataset's `interaction_edges` column, is single-target, and goes through the shared
CV/OOD protocol. `reference/GraphSAGE-SWAP-and-Polarization.py` is Steven's original standalone
version — QASM-sourced (hence its `pull_gate_types` helper), multi-task, own train/val/test split,
hardcoded paths. It is kept for provenance and is **not** wired into the pipeline; see
`reference/README.md` for the full comparison.

## Output

`supervised_analysis_run.py` prints a per-model table of R², Pearson, Spearman, MAE, and RMSE.
`cross_device.py` writes `data/xdev_out/cross_device_results.csv` with, for each
device × target × model, every metric's mean, seed-sd (GNN), and 95% bootstrap CI.

## Requirements

Python 3.10+, numpy, pandas, scipy, scikit-learn. For the GNN: torch, torch_geometric. For
the cross-device label stage and corpus build: qiskit 2.4.1, qiskit-aer, qiskit-ibm-runtime,
mqt.bench 2.2.2, networkx. See `requirements.txt`.
