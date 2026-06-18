# Mirror-polarization labeling (any Fake backend) : mirror_polarization_label.py

Computes the **route-then-mirror polarization** noise-resilience label for every circuit
in the corpus. This is the supervised target for the fidelity arm; the SWAP arm uses the
deterministic routing counts from `swap_features.py` instead (no simulation needed).

## The construction

For each circuit U:
1. **Route once.** Transpile U onto the chosen device at full optimization → `U_routed`
   (SWAPs minimized, gates fused) — an efficiently-routed "ground truth" execution.
2. **Strip + remap.** Reduce to the active qubits and remap the device noise model onto
   that small register.
3. **Mirror the routed circuit.** Build `U_routed · U_routed⁻¹`, where the inverse is the
   literal gate-reversal of the already-routed circuit (NOT re-transpiled). The noiseless
   output is |0…0⟩ with certainty for **any** U, so the target is unambiguous and
   `ideal_peak == 1.0` by construction (recorded per row as a sanity check).
4. **Score.** Run under the noise model; the label is the Hamming-weighted effective
   polarization S (Proctor et al. 2022, Eq. 1) against |0…0⟩. S ∈ [0,1]: 0 = noise,
   1 = perfect.

## Why this construction

**Why plain U·U⁻¹ and not a central Pauli (U·P·U⁻¹).** A central X-mask only yields a
clean basis-state return when U is *Clifford*. Our corpus is mostly non-Clifford (qaoa
and all vqe variants have continuous rotations), so for those circuits U·P·U⁻¹ does not
return to a single bitstring — its ideal output is smeared (`ideal_peak < 1`), and the
label would be scored against a target whose validity is itself correlated with circuit
structure: a confound on the very thing we study. Plain U·U⁻¹ = identity for every U, so
the label is clean across the whole corpus.

**Why route-then-mirror and not mirror-then-transpile.** Handing U·U⁻¹ to the transpiler
at optimization ≥ 1 lets it cancel the two inverse halves against each other (in the
limit, to an empty circuit → S ≈ 1 for everything). Routing U first and mirroring the
routed circuit preserves the full faithful depth, so S reflects real execution noise
rather than transpiler aggressiveness. (This was a real bug in an earlier version: joint
transpilation of the mirror biased the label by up to ~0.5, in both directions depending
on how much cancellation fired.)

**Rejected alternatives.** *Hellinger fidelity* fails for circuits whose ideal output is
near-uniform (randomcircuit, graphstate): ideal and fully-decohered distributions
coincide, so it reports high fidelity for circuits that are fundamentally noise.
*Exact-success rescaling* `(p−1/D)/(1−1/D)` floored the whole corpus to ~0 (it demands an
exact return to one bitstring); the Hamming-weighted form gives near-misses partial credit
and resolves a full [0,1] gradient.

**Relation to mirror RB.** This is mirror-circuit fidelity of a *structured* benchmark
circuit — a deliberate departure from canonical mirror RB (Proctor et al. 2022), which
interleaves Pauli-dressed *random* Clifford scrambling layers to twirl errors and
randomize the target. We keep their Hamming-weighted estimator but apply it to one
mirrored structured circuit, because the research question is precisely how the structure
of real algorithmic circuits affects resilience.

## Known bias (documented, not hidden)

The |0…0⟩ target carries a mild ground-state-attraction bias that **grows with routed gate
count**, so the label may slightly *overstate* resilience at high gate counts. Verified:
scoring the same decohered output against a *random* target gives ≈0 at all gate counts (a
clean, unbiased floor), while the |0⟩ score sits above the floor and that gap rises with
`routed_2q`. This bias lives in the floored, high-N region the fidelity regression
excludes anyway; in the unfloored low-N window it is a near-constant offset, not a
structural confound.

## Install (no network needed at runtime)
    pip install -r requirements.txt

## Run

Single device (default FakeBrisbane):

    python mirror_polarization_label.py qasm polarization_labels.csv --n-lo 3 --n-hi 5

Multiple devices — pass a comma-separated list; each circuit is labeled on every device
and rows are stamped with a `device` column:

    python mirror_polarization_label.py qasm polarization_labels.csv --n-lo 3 --n-hi 5 --device FakeBrisbane,FakeBoston,FakeNighthawk

The full pipeline accepts the same flag and loops automatically:

    python run_pipeline.py --qubits 3 6 --k 25 --n-lo 3 --n-hi 5 --device FakeBrisbane,FakeBoston

## Output columns
**identifiers**: file, algo, n_qubits, device
**label**: polarization (clipped [-0.1,1]), polarization_raw (unclipped)
**post-routing**: routed_2q (= 2 × forward routed 2q gates), mirror_depth, n_active
**pre-routing** (from features.csv after join): interaction_edges, spectral/graph features
**sanity/diagnostic**: ideal_peak (== 1.0 for a faithful mirror), p_success, status, err

status ∈ {labeled, route_fail, error}. route_fail = directed-gate transpiler crash (drop
the circuit — routing onto hardware failed). polarization ≈ 0 = fully decohered (a valid
label, not a failure).

## Cost notes
Per-circuit cost is dominated by the noisy statevector sim. The dominant driver is the
**routed mirror depth** (≈ 2× the routed bare circuit), not qubit count — dense circuits
that route to many 2q gates / deep mirrors are by far the slowest (a dense qaoa at N≈6 can
take minutes). Randomcircuit may produce route_fail at high N (directed-gate edges) —
expected, those drop out.

## N range
The polarization label is only non-degenerate at low N — circuits floor to S ≈ 0 by
N ≈ 5–6 (sooner for dense families). Label the fidelity arm over a restricted low-N range
(e.g. `--n-hi 5`); labeling to high N mostly produces floored labels at high simulation
cost. The SWAP arm (deterministic routing counts, no simulation) covers the full N range
via `swap_features.py`.
