# `frf.py` cleanup — findings list

Scope: `src/mqtloader/helpers/frf.py`. Question answered: how does it fit the project,
what's useful and why, what can be deleted without breaking anything, what's outdated.

Verification basis (grep across `src/mqtloader/**.py`, branch `supervised-report-materials`):
- `from frf import …` appears in exactly three files:
  `post_transpile/mirror_polarization_label.py`, `post_transpile/swap_features.py`,
  `run_pipeline_with_workers.py`.
- They import **only** `strip_with_map` and `remap_noise_model`. Nothing else from `frf` is
  imported anywhere.

---

## 1. How it fits the project scope

The production pipeline is: generate → features → label (polarization **or** swap) → join.
`frf.py`'s only role in that pipeline is as the **noise-remap utility** for the labeling
stage. When a routed circuit is stripped down to its active qubits (so the noisy sim is
tractable), the device noise model must be re-keyed from physical qubit indices to the
stripped indices — otherwise a gate routed onto physical edge (37,38) gets simulated with
the noise registered at (0,1). That re-keying is `remap_noise_model` + `strip_with_map`.

So in scope, `frf.py` = two load-bearing helpers. The rest of the file is **EDA scaffolding**
(a one-off proof that the remap is faithful) that never became part of the pipeline.

---

## 2. Useful — keep (load-bearing)

| symbol | why it stays | live callers |
|--------|--------------|--------------|
| `remap_noise_model(base, phys2new)` | re-keys 1q/readout/ecr errors to stripped indices; correctness of every polarization label depends on it | `mirror_polarization_label.py:271`, `run_pipeline_with_workers.py:262` |
| `strip_with_map(tcirc)` | rebuilds the routed circuit on active qubits, returns the `phys2new` map the remap consumes | `mirror_polarization_label.py:233`, `swap_features.py:57`, `run_pipeline_with_workers.py:220/227` |

These two are the actual "Threat-A fix." They are correct and current — keep verbatim.

Imports they actually need: `from qiskit import QuantumCircuit` (strip) and
`from qiskit_aer.noise import NoiseModel` (remap). Nothing else.

---

## 3. Removable without breaking anything (dead outside this file)

Confirmed zero external references — safe to delete:

| symbol / block | what it is | proof it's dead |
|----------------|-----------|-----------------|
| `main()` + `if __name__ == "__main__"` | standalone validation harness | not imported; run by hand only |
| `compute_fixed(...)` | Hellinger fidelity of routed circuit vs ideal | only called inside `main()` |
| `_ground_truth(...)` | physical-index sim for cross-check | only called inside `main()` |
| `routed_2q_pairs(tcirc)` | off-edge counter for the harness | only called inside `main()` |
| import `read_qasm` (`loader_v2`) | feeds `main()` only | live code imports `read_qasm` directly from `loader_v2`, not via `frf` |
| imports `sys`, `Path`, `transpile`, `hellinger_fidelity`, `AerSimulator`, `FakeBrisbane` | all used only by the dead block | — |

Deleting all of the above reduces `frf.py` from ~142 lines to a ~30-line two-function
utility module with two imports. **No pipeline path touches any of it.**

---

## 4. Outdated information

- **"Threat-A" / "Threat-B" jargon.** Internal EDA labels with no meaning to a present-day
  reader. "Threat-B" no longer appears anywhere in the code; "Threat-A" survives only in this
  docstring. Recommend rewording the docstring to describe the bug plainly ("noise model was
  keyed to physical indices but the circuit was remapped to stripped indices") and dropping
  the threat labels.
- **Dead file/symbol references in the docstring.** It cites `fidelity_routed.py` and
  `_strip_idle_qubits` as the buggy predecessor. Neither exists in the repo anymore — the
  narrative is archaeology. Keep at most a one-line "supersedes the earlier physical-index
  strip" note; drop the rest.
- **Stale validation corpus & families.** `main()` hardcodes
  `data/corpus_v2_20q/corpus_v2/qasm` and tests `ghz / wstate / bv / dj` circuits. The current
  loader (`loader_v2.STRUCTURAL_FAMILIES`) generates `qaoa, graphstate, randomcircuit,
  vqe_*` only, and the corpus now lives at `qasm/` (ROOT) — so `main()` can't run as written.
- **Methodology drift (the important one).** `compute_fixed`/`main` measure **Hellinger
  fidelity** between ideal and noisy routed counts. The project no longer labels that way — it
  labels via **route-then-mirror polarization** (`mirror_polarization_label.py`, Proctor 2022
  Eq. 1). So the validation block proves a label the pipeline doesn't produce. The thing it was
  guarding — that `remap_noise_model` is faithful — is still worth a test, but this
  implementation of that test is obsolete.

---

## 5. Recommendation

**Split the file by what it is:**

1. **Keep** `frf.py` as a pure utility module: `strip_with_map`, `remap_noise_model`, two
   imports, a short plain-English docstring (no Threat-A/B, no dead-file archaeology).
2. **Decide on the validation block** (`main`/`compute_fixed`/`_ground_truth`/`routed_2q_pairs`):
   - *If the remap-faithfulness check still has value* (it does — it's a real correctness
     assertion: `offedge == 0` and stripped-edge error object `==` physical-edge error
     object): move it to `src/supervised/sidetests/` (or a `tests/`) as a proper test,
     updated to the current families/corpus, asserting instead of printing.
   - *If not*, delete it outright — nothing imports it.

Net: no behavior change to the pipeline either way, because the only live exports are the two
helpers, which stay untouched.

---

### Open item
You mentioned "the changelog" — there is no separate changelog file in the repo, and the only
Threat-A text is inside `frf.py` itself. If your changelog is in your unsaved working copy,
share it and I'll fold it in.
