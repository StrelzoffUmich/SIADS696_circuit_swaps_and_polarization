# MQTLoader

This loader was developed with assistance from Claude Opus 4.7/4.8.

A pipeline for generating data useful for studying how **pre-routing circuit structure** predicts **post-routing execution cost and noise resilience** on simulated quantum hardware. 

Given a corpus of quantum circuits, it produces a dataset linking each circuit's structural features to two outcomes: routing overhead (SWAP count) and noise resilience (mirror-circuit polarization).

## What it does

The pipeline runs four stages, orchestrated by `run_pipeline.py`:

1. **Generate** (`load_qasm/loader_v2.py`) — produces a corpus of QASM circuits
   across structural families from MQT Bench: qaoa, graphstate, randomcircuit, three VQE
   variants with fixed interaction graphs, and one VQE variant with randomized interaction
   graphs (useful for increasing SWAP-count variation). Edges are parameterized by qubit
   count and seed, for a total of seven types of algorithm.
2. **Extract features** (`helpers/extract_features.py`) — computes 44 pre-routing
   structural features (spectral, graph-theoretic, and gate-level) from the bare
   circuit, plus the interaction-graph edge list, into `features.csv`.
3. **Label** (`post_transpile/mirror_polarization_label.py`) — transpiles
   each circuit onto a chosen device, simulates it under that device's noise model,
   and computes the mirror-circuit polarization (noise-resilience label) plus
   routing quantities, into `polarization_labels.csv`.
4. **Join** (`build_dataset.py`) — stitches features (X) and labels (y) on the
   `file` key into `full_dataset.csv`.

Each run writes its outputs to a timestamped `results/run_<timestamp>/` folder.

## Quick start

```
# SWAP analysis — no simulation, routing overhead calculation
python run_pipeline.py --no-naiveMP --qubits 3 20 --k 25 --n-lo 3 --n-hi 20

# fidelity analysis — low N, simulation, Naive mirror-polarization
python run_pipeline.py --qubits 3 6 --k 25 --n-lo 3 --n-hi 5
```

This generates the corpus, extracts features, labels on Fake Brisbane (default) and joins —
writing `results/run_<timestamp>/full_dataset.csv`.

For convenience - the canonical runs of the dataset, produced via a 16-20 hour HPC run are pre-seperated into swaps (easy to run) and polarization (extremely long runtime). the relevant file are renamed 

### Arguments

**Generation (stage 0)**
| flag | default | meaning |
|------|---------|---------|
| `--algorithms` | all | families to generate (e.g. `qaoa graphstate`); space-separated |
| `--qubits LO HI` | `3 20` | qubit range for the corpus - beyond 15 produces noticeable computational slowdown |
| `--k` | `25` | loader attempts to deliver up to K distinct structures per (family, N) cell |

(Note that some families and cells do not have a full rank of valid distinct seeds at a given qubit count — e.g. there are not 25 distinct N=3 graph states. The loader warns you of these under-sampled cells inline.)

**Stage 2 — labeling (polarization) or routing (SWAP arm)**
| flag | default | meaning |
|------|---------|---------|
| `--n-lo` / `--n-hi` | `3` / `6` | min/max N for stage 2 (independent of the generation range) |
| `--families` | all present | comma-separated families to process (e.g. `qaoa,graphstate`) |
| `--device` | `FakeBrisbane` | backend(s) for routing+noise; comma-separated for multi-device (e.g. `FakeBrisbane,FakeBoston`) |
| `--no-naiveMP` | off | **SWAP-arm mode**: skip the mirror-polarization simulation and run the transpile-only SWAP pass instead (routing counts, no noise sim, full N range). Aliases: `--swap-only`, `--no-mp`. Output is `swap_features.csv` with no polarization column. |

**Stage 3 — join**

No tunable flags; the join stitches the stage-1 features and stage-2 output on `file` automatically.

**Output paths & stage control**
| flag | default | meaning |
|------|---------|---------|
| `--results-dir` | `results` | parent folder for timestamped run outputs |
| `--no-timestamp` | off | write to project root instead of `results/run_<timestamp>/` |
| `--clean` | off | delete `qasm/` and generated CSVs before running (fresh build) |
| `--skip-gen` / `--skip-features` / `--skip-labels` | off | skip a stage you've already completed |
| `--qasm-dir`, `--features-csv`, `--labels-csv`, `--out` | sensible defaults | override individual artifact paths |

## The two arms
The pipeline supports two distinct research questions sharing one corpus:

**SWAP arm** — pre-routing structure predicting routing overhead. SWAP count is deterministic, easy to calculate, and noise-free, so this arm uses the full N range.

Run the SWAP arm through the pipeline with `--no-naiveMP` (a.k.a. `--swap-only`): this
skips the noisy mirror-polarization simulation and runs the transpile-only SWAP pass
(`swap_features.py`), recording bare/mirror routed-2q and depth over the full N range in
seconds. Output is `swap_features.csv` (no polarization column). Omit the flag for the
fidelity arm.

**Fidelity (resilience) arm** — pre-routing structure predicting noise resilience,
measured as mirror-circuit polarization. Resilience is only reliably non-degenerate at low N on FakeBrisbane
(survival decays with routed gate count; circuits floor by N≈6–10 depending on
family), so this arm should use a restricted range that depends on the device. 

## The polarization label

Noise resilience is measured via **route-then-mirror polarization**. Each circuit U is
first transpiled onto the device at full optimization (efficient routing), then mirrored
as U_routed · U_routed⁻¹ — a faithful "do the work, then exactly undo it" execution whose
noiseless output is |0…0⟩ with certainty for *any* circuit (Clifford or not). Running it
under the device noise model and scoring recovery of |0…0⟩ gives the Hamming-weighted
effective polarization S ∈ [0,1] (Proctor et al. 2022, Eq. 1): 0 = indistinguishable from
noise, 1 = perfect.

Routing *before* mirroring is essential: handing U·U⁻¹ to the transpiler would let it
cancel the two halves against each other. Mirroring the already-routed circuit preserves
the full faithful depth, so S reflects real execution noise. The label is verified clean
per row via `ideal_peak` (== 1.0 for a faithful mirror).

This is mirror-circuit fidelity of a *structured* benchmark circuit — a deliberate
departure from canonical mirror RB (which uses random Pauli-dressed scrambling layers),
because the research question is how the structure of real algorithmic circuits affects
resilience. Known caveat: the |0…0⟩ target carries a mild ground-state-attraction bias
that grows with routed gate count, so the label may slightly overstate resilience at high
gate counts — which sits in the floored, high-N region the fidelity regression excludes.

## Multi-device support

`--device` selects the backend for routing and noise (e.g. `FakeBrisbane`,
`FakeBoston`, `FakeNighthawk`). Pass a comma-separated list for a cross-device run;
each circuit is labeled on every device, and rows are stamped with a `device`
column. Features are device-independent (pre-routing) and computed once.

```
python run_pipeline.py --qubits 3 6 --k 25 --n-lo 3 --n-hi 6 --device FakeBrisbane,FakeBoston
```

## Inspecting any circuit

`helpers/row_graph.py` reconstructs and visualizes any circuit's interaction graph from its row in the final dataset:

```
python helpers/row_graph.py results/run_<timestamp>/full_dataset.csv <filename> --draw graph.png
```
This functionality was not used extensively due to limited space in the report.

## Noise-remap helpers (`helpers/frf.py`)

`helpers/frf.py` is a small utility module — just two functions — used by the
labeling stages (`mirror_polarization_label.py`, `swap_features.py`,
`run_pipeline_with_workers.py`):

- **`strip_with_map(tcirc)`** — rebuilds a routed circuit on only its *active*
  qubits (renumbered `0..N_active-1`, so the noisy sim stays tractable) and returns
  the `phys2new` map from physical to stripped qubit index.
- **`remap_noise_model(base, phys2new)`** — re-keys each physical qubit/edge's real
  1q / readout / ecr error onto its stripped index using that map, so the noise the
  simulator applies matches the qubits the gates were actually routed onto.

Together they keep the device noise model in lockstep with the qubit strip; without
the remap, a gate routed onto physical edge (37,38) would be simulated with whatever
error sits at (0,1) — the wrong qubits' noise.

## Output columns

The dataset joins structural, pre-routing features and post-routing labels per circuit:
- **identifiers**: `file`, `algo`, `n` (from features) / `n_qubits` (from labels), `device`
- **label**: `polarization` (+ `polarization_raw`)
- **post-routing**: `routed_2q`, `mirror_depth`, `n_active`
- **pre-routing features**: spectral (`fiedler_topology`, `laplacian_max_eig_*`, …),
  graph (`graph_density`, …), gate-level, and `interaction_edges`
- **diagnostics**: `p_success`, `ideal_peak` (==1.0 sanity check), `status`

## Requirements

Python 3.10+, qiskit 2.4.1, qiskit-aer, qiskit-ibm-runtime, mqt.bench = 2.2.2, networkx,
numpy, pandas. See `requirements.txt`. 