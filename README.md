# Quantum Circuit Structure → Routing / Resilience

Does the **pre-routing structure** of a quantum circuit predict (a) its **routing overhead**
(SWAP insertion under a device coupling map) and (b) its **noise resilience** (mirror-circuit
polarization)? This repo trains and evaluates supervised models for both, in-distribution
(5-fold CV), out-of-distribution (leave-one-family and an external mqt/nwq/qasmbench corpus),
and across IBM fake devices.

Developed with assistance from Claude Opus 4.7/4.8.

## Layout

```
src/
  mqtloader/      corpus generation + per-device labeling (run_pipeline.py + stages).
  supervised/     supervised_analysis_run.py      -- shared CV / OOD eval across model families.
                  diagnostics.py                  -- importance, ablation, runtime (split out of the eval engine).
                  cross_device.py                 -- train on one device, test on any other.
                  report_artifacts.py             -- single producer of the report-body tables & figures.
                  build_validation_corpus_sv.py, gnn_interaction.py, stress_test.sh
                  viz_tests/                      -- exploratory + appendix figures (each imports report_artifacts).
                  sidetests/                      -- standalone scripts answering specific report questions.
                  reference/                      -- reference GraphSAGE implementation + gate-type puller.
                  1_Normalization_and_Ridge_Lasso_regression.ipynb -- dashboard source (normalization, ridge/lasso).
                  README.md                       -- commands for every analysis.
  unsupervised/   generate_all_embeddings.sh      -- bash script to automate embedding generation via scripts in embedding_generation_scripts
                  run_model_scripts.sh            -- bash script to run all the model scripts in model_run_scripts
                  embedding_generation_scripts/    -- scripts to generate embeddings of DAGs created from QASM files
                  model_run_scripts/               -- scripts to run the kmeans, gmm, and hsbscan analysis of the embedding files
                  unsupervised_learning_results_walkthrough.ipynb  -- post embedding notebook that goes through analysis process and paper prep
data/             datasets, corpora, run outputs (gitignored; regenerable from src/).
```

## Data naming convention for supervised learning tasks

Every labeled artifact is tagged **role × arm × device × N-range** and ships a
`run_manifest.json` so it is self-describing on disk:

- `role` ∈ `{train, val}` — in-distribution training set vs out-of-distribution validation corpus
- `arm`  ∈ `{swap, pol}`  — routing overhead (deterministic transpile) vs fidelity (noisy sim)

```
data/datasets/{role}_{arm}_{device}.csv                       e.g. train_swap_FakeBrisbane.csv
data/runs/{role}__{arm}__{device}__n{lo}-{hi}__{timestamp}/   (features.csv, *_labels.csv,
                                                               the dataset, run_manifest.json)
data/corpora/validation_qasm/                                 flat *.qasm OOD corpus
data/xdev_out/                                                cross-device labeled CSVs + results
data/results/figures/report/                                  report-body figures & tables (report_artifacts.py)
data/results/figures/exploratory/                             exploratory + appendix figures (viz_tests/, sidetests/)
```

The `run_manifest.json` records role/arm/device/N, the **target column**, and exactly how the
harness target is derived from it (e.g. `route = log1p(bare_routed_2q)`), so an outsider can
read a dataset without reading code.

## Data naming convention for unsupervised learning tasks
```
data/datasets/embedding_data/embeddings/{GNN_type}_{complexity_levels}.csv
data/datasets/embedding_data/loss/{GNN_type}_{complexity_levels}_loss.csv
data/datasets/unsupervised_learning_results/
  * most files are temporary files, but summary analysis files post models and ARI are:
    * ALL_cluster_assignments.csv
    * ALL_summary_comparison.csv
    * embedding_comparison_ranked.csv
```

## Quick start

```
# label a validation corpus on devices and run the cross-device transfer table
python src/supervised/cross_device.py --devices FakeBrisbane FakeBoston --targets route pol pol_z

# in-distribution 5-fold CV, all model families
python src/supervised/supervised_analysis_run.py --target route --mode indist --all

# generation for the GNN embeddings can be run by: (this will take a long while
bash src/unsupervised/generate_all_embeddings.sh

# run unsupervised models agains the GNN embeddigns
bash src/unsupervised/run_model_scripts.sh
```

Defaults resolve to `src/mqtloader` and `data/` automatically (paths are anchored to the
script location, not the working directory). See `src/supervised/README.md` for the full
command set, and run `bash src/supervised/stress_test.sh` to verify a fresh checkout.

## Requirements

Python 3.10+, numpy, pandas, scipy, scikit-learn. GNN: torch, torch_geometric. Labeling /
corpus build: qiskit 2.4.1, qiskit-aer, qiskit-ibm-runtime, mqt.bench 2.2.2, networkx.
See `requirements.txt`.
