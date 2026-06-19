# MQT_Loader — Data Dictionary

Documents every column in `full_dataset.csv`, what each column means, and whether it is usable for machine learning.

There are **two dataset variants** produced by the same pipeline:
- **SWAP arm** (`--no-naiveMP`): target columns are routing counts.
- **Fidelity arm** (default): target columns include **`polarization`** and routed_2q; the bare/mirror SWAP columns are not present.

A given `full_dataset.csv` is one or the other. 

Check for a `polarization` column to tell which you have.

Legend for the **Use** column:
- **target** — a y you can predict.
- **feature** — a valid X predictor.
- **id/meta** — identifier or metadata; never a predictor.
- **drop** — degenerate (constant/near-constant) or raw structure; exclude from models, used for diagnostics.
- **leakage** — derived from or equivalent to an outcome; never use as a feature when predicting another outcome.
---
## Identifiers & metadata (never features)

| column | meaning | Use |
|--------|---------|-----|
| `file` | QASM filename; the join key | id/meta |
| `algo` | family (qaoa, graphstate, randomcircuit, vqe_*) | id/meta |
NOTE :graphstate has a anomaly where odd and even qubit values have fundamentally differing structure.
| `level` | MQT Bench abstraction level (constant: `indep`) | drop |
| `target` | MQT Bench target tag (constant: `none`) | drop |
| `n`, `num_qubits`, `n_qubits` | qubit count (three copies from the join, all three are identical) | id/meta — multicollinear with other downstream features; **potential leakage** if testing on out of distribution. Also : issues with certain families having less signal at lower N values. |
| `seed_idx` | seed index within a (family, N) cell | id/meta |
| `interaction_edges` | raw JSON edge list of the interaction graph | id/meta — for `row_graph.py` reconstruction; not a scalar as is |
| `device` | backend used for routing/noise (e.g. FakeBrisbane) | id/meta, join key for cross-device |
| `status` | `routed` (swap arm) / `labeled` (fidelity arm); failures dropped in join | id/meta |
| `err` | error string for failed rows (blank when ok) | id/meta |
---
## Pre-routing features (device-independent, computed once)

### Spectral features
Derived from the Laplacian of the interaction graph. `_topology` = unweighted; `_2q_weighted` = edges weighted by 2q-gate multiplicity.

| column | meaning | Use |
|--------|---------|-----|
| `fiedler_topology` | algebraic connectivity (λ₂); higher = harder to route | feature |
| `fiedler_2q_weighted` | Fiedler value of the 2q-weighted graph | feature |
| `laplacian_max_eig_topology` | largest Laplacian eigenvalue (local density bound) | feature |
| `spectral_gap_ratio_topology` | λ₂/λ_max, normalized connectivity | feature |
| `spectral_entropy_topology` | entropy of eigenvalue spectrum | feature |
| `spectral_entropy_2q_weighted` | same, 2q-weighted | feature |
| `laplacian_energy` | sum of |eigenvalues|; global complexity | feature |
| `log_estrada_index` | log Estrada index; subgraph centrality summary | feature |
| `von_neumann_entropy` | spectral (von Neumann) entropy of the graph | feature |
| `effective_resistance` | total effective resistance; connectivity robustness | feature |
| `log_spanning_trees` | log number of spanning trees | feature |
| `fiedler_at_half_depth` | Fiedler of the graph built from the first half of the circuit | feature |

### Graph-theoretic features
| column | meaning | Use |
|--------|---------|-----|
| `graph_density` | edge density of interaction graph | feature |
| `graph_diameter` | longest shortest-path | feature |
| `avg_clustering` | average clustering coefficient | feature |
| `max_degree` | maximum node degree | feature |
| `degree_variance` | variance of node degrees | feature |
| `assortativity` | degree assortativity | feature |
| `num_triangles` | triangle count | feature |
| `n_components` | connected components (mostly 1) | feature (low-variance), drop |
| `gini_2q_multiplicity` | Gini of 2q-gate multiplicity across edges | feature |
| `edge_weight_mean_2q` | mean 2q multiplicity per edge | feature |
| `edge_weight_max_2q` | max 2q multiplicity (only 6 distinct values) | feature (low-variance), drop|

### Temporal / circuit-flow features
Capture gate *ordering* / structure over depth, not just the static graph.

| column | meaning | Use |
|--------|---------|-----|
| `critical_depth` | fraction of gates on the critical path | feature |
| `parallelism` | gate parallelism measure | feature |
| `liveness` | qubit liveness measure | feature |
| `entanglement_ratio` | ratio relating entangling activity | feature |
| `program_communication` | qubit communication measure | feature |
| `twoq_temporal_locality` | temporal locality of 2q gates | feature |
| `time_to_connected` | depth at which interaction graph becomes connected | feature |
| `gate_entropy` | entropy of the gate-type distribution | feature |
| `depth_per_qubit` | depth normalized by qubit count | feature |

### Rotation / non-Clifford features
Describe continuous rotation content. **Relevant to the FIDELITY arm only.**
Routing ignores rotation angles, so these add noise (not signal) when predicting SWAP counts. They are what distinguish the three count-matched VQE families (same interaction graph, different rotations), so they matter for *resilience*.

| column | meaning | Use |
|--------|---------|-----|
| `n_rotation_gates` | number of parameterized rotation gates | feature (fidelity arm) |
| `mean_abs_angle` | mean absolute rotation angle | feature (fidelity arm) |
| `std_angle` | std of rotation angles | feature (fidelity arm) |
| `sum_sin_sq_half` | Σ sin²(θ/2); non-Clifford "strength" | feature (fidelity arm) |
| `mean_angle_squared` | mean of squared angles | feature (fidelity arm) |
| `angle_position_weighted` | depth-position-weighted angle measure | feature (fidelity arm) |

### Coarse size features (can we make models that create better predictions than these alone)
| column | meaning | Use |
|--------|---------|-----|
| `depth` | bare circuit depth | feature |
| `size` | total gate count | feature |
| `num_2q_gates` | bare 2q-gate count | feature |
| `num_unique_gates` | distinct gate types | feature |

### Drop (artifact/useless for modeling)
| column | reason | Use |
|--------|--------|-----|
| `has_2q_interactions` | near-constant (essentially always 1) | drop |
| `level`, `target` | inherited metadata | drop |
---

## Post-routing outcomes (the y targets — never features for each other)

### SWAP arm (`--no-naiveMP`)
| column | meaning | Use |
|--------|---------|-----|
| `bare_routed_2q` | 2q gates after routing the bare circuit | **target (primary)** |
| `mirror_routed_2q` | 2q gates after routing the mirror circuit | target (secondary) |
| `bare_routed_depth` | depth after routing the bare circuit | target (alt) |
| `mirror_routed_depth` | depth after routing the mirror | target (alt) |
| `mirror_over_bare_2q` | ratio mirror/bare 2q | target (alt) / diagnostic, drop |
| `bare_n_active`, `mirror_n_active` | active qubits after routing | diagnostic, drop |
| `pre_routing_2q`, `pre_routing_depth` | pre-routing counts (≈ `num_2q_gates`, `depth`) | feature/diagnostic |

### Fidelity arm (present with default settings)
| column | meaning | Use |
|--------|---------|-----|
| `polarization` | mirror-circuit effective polarization S∈[0,1]; **resilience label** | **target (primary)** |
| `polarization_raw` | unclipped S | drop, artifact from non-naive mirror |
| `routed_2q` | 2q gates in the routed mirror (= 2× forward) | leakage as feature; useful when modeled against polarization as the per-swap loss in mirror polarization derived fidelity, drop from full model |
| `mirror_depth` | depth of the routed mirror | leakage as feature, drop from model |
| `p_success` | exact |0…0⟩ return probability | diagnostic, drop from model |
| `ideal_peak` | noiseless target probability (==1.0 sanity check) | diagnostic, drop from model |
| `n_active` | active qubits after routing | diagnostic, drop |
---
## Modeling guidance

**SWAP arm** — predict `bare_routed_2q` from spectral + graph + temporal + coarse-size features.

**Fidelity arm** — predict `polarization` from spectral + graph + temporal + **rotation** features. 
Rotation features are essential here particularly for the vqe algorithm family. 

**Never use as a predictor**: `algo`, any `n*` column (if looking for size-agnostic effects), `interaction_edges`,
`file`/`seed_idx`/`device`/`status`/`err`, the `drop` rows above, and any post-routing
outcome when predicting another.

## Recommended base configurations

## SWAP arm — target: `bare_routed_2q`
# "what is the source of routing overhead, given X device?"

### Target
- `log(bare_routed_2q)`

### Features (keep)
- `fiedler_topology`
- `laplacian_energy`
- `log_spanning_trees`
- `log_estrada_index`
- `effective_resistance`
- `von_neumann_entropy`
- `graph_density`
- `graph_diameter`
- `avg_clustering`
- `num_triangles`
- `degree_variance`
- `assortativity`
- `critical_depth`
- `parallelism`
- `program_communication`
- `twoq_temporal_locality`
- `time_to_connected`


### Drop (based on 101 circuit testbed from n=[3,5])
- `num_2q_gates`, `depth`, `size`, leaky for this application. all of these correlate with the output.
- All rotation features (`n_rotation_gates`, `mean_abs_angle`, `std_angle`, `sum_sin_sq_half`, `mean_angle_squared`, `angle_position_weighted`) — routing ignores rotation angles
- All `_2q_weighted` spectral variants and 2q-multiplicity features (`fiedler_2q_weighted`, `spectral_entropy_2q_weighted`, `gini_2q_multiplicity`, `edge_weight_mean_2q`) — collinear with `_topology`, empirically weaker, blind to multi-qubit gates
- `spectral_gap_ratio_topology` — redundant with `fiedler_topology`
- `spectral_entropy_topology`, `laplacian_max_eig_topology` — strong marginal predictors but redundant with retained connectivity features (drop after confirming redunancy holds)
- `max_degree` — redundant with `laplacian_energy`
- `gate_entropy`, `num_unique_gates` — redundant pair, weak signal (drops the gate-diversity dimension entirely — intended)
- `fiedler_at_half_depth`, `liveness`, `entanglement_ratio`, `depth_per_qubit` — weak/redundant temporal-structure features
- `has_2q_interactions`, `edge_weight_max_2q`, `n_components` — degenerate / low-variance

## Fidelity arm — target: `polarization`
# "what features associate with decoherence?"

### Target
- `polarization`

### Features (keep — base model, ~3-5 based on small 101 sample draw) 
- `spectral_entropy_topology` + `twoq_temporal_locality` + `log_spanning_trees` == 0.81 r^2, RF

### Drop
- Everything not in the keep list — chosen before fitting
- All `_2q_weighted` variants
- All post-routing columns (leakage)
---

## Always drop (both arms)

### Identifiers / metadata
- `file`, `algo`, `seed_idx`, `device`, `status`, `err`, `interaction_edges`

### Degenerate
- `level`, `target`, `has_2q_interactions`

### Qubit count (conditional)
- `n`, `num_qubits`, `n_qubits`

### Post-routing (drop when not the active target)
- `mirror_routed_2q`, `bare_routed_depth`, `mirror_routed_depth`, `mirror_over_bare_2q`
- `bare_n_active`, `mirror_n_active`, `n_active`
- `routed_2q`, `mirror_depth`
- `p_success`, `ideal_peak` — keep in CSV for auditing, drop from model/final dataset
