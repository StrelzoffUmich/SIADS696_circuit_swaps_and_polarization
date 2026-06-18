# reference/ — provenance, not wired into the pipeline

These files are kept for the record. **Nothing in the live pipeline imports them**, the test
runner does not exercise them, and they are not needed to reproduce any harness result. They
document where the GNN came from.

## Files

- **`GraphSAGE-SWAP-and-Polarization.py`** — Steven Flack's original standalone GraphSAGE.
  Reads two MQT_Loader `full_dataset.csv` files (a SWAP run and a polarization run), merges on
  `file`, builds an interaction graph per circuit straight from the QASM, and trains a
  **multi-task** GraphSAGE that predicts `log1p(bare_routed_2q)` and `polarization` jointly with
  its own random 70/15/15 split.
- **`pull_gate_types.py`** — gate-vocabulary scanner (`extract_gates_with_legacy_support`). Its
  **only** importer is the script above (`from pull_gate_types import …`); they are co-located here
  so that import keeps resolving. This is the file that looked "orphaned" at the top level — it was
  never orphaned, its importer was just missing from the repo.

## Why this is separate from `../gnn_interaction.py`

`../gnn_interaction.py` is the GNN the **harness** actually runs. It is a different implementation
on purpose — see its module docstring. The substantive differences:

| | `../gnn_interaction.py` (live) | `GraphSAGE-SWAP-and-Polarization.py` (here) |
|---|---|---|
| Graph source | `interaction_edges` column in the dataset (no QASM) | parses the QASM file directly |
| Gate features | 3 dataset columns (`gate_entropy`, `num_unique_gates`, `log1p(num_2q_gates)`) | full gate-vocab count vector via `pull_gate_types` |
| Graph-level features | 18 spectral/structural columns | 7 simple ones (`n_qubits, depth, num_ops, num_2q, dag_width, density, avg_degree`) |
| Targets | single (route **or** pol **or** z, set per fold) | multi-task (both jointly) |
| Eval protocol | harness GroupKFold / leave-one-family / cross-device | own random 70/15/15 split |

The live model is sourced cleanly from the dataset so it runs through the shared CV/OOD protocol
with every other model family; it is **not** a bit-exact reproduction of the run below. If Steven's
exact training numbers are wanted, they come from this script, reported separately.

## Running it

Reference only — it will **not** run as-is. The paths are hardcoded to the author's machine
(`/Users/steventf/Desktop/...` for `csv_path1`, `csv_path2`, `qasm_files_path`, `cache_path`, and a
placeholder `merged_csv_path`). To reproduce, edit those constants at the top to point at a SWAP-arm
`full_dataset.csv`, a polarization-arm `full_dataset.csv`, and the matching QASM directory. Needs
`torch`, `torch_geometric`, `qiskit`.
