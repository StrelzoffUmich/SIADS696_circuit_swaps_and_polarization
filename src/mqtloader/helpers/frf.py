#!/usr/bin/env python3
"""
fidelity_routed_fixed.py — post-routing fidelity with the Threat-A fix.

AI-assisted (Claude), per SIADS 696 GenAI coding-disclosure policy.

Bug in fidelity_routed.py: it routes onto FakeBrisbane (every 2q gate lands on a
real heavy-hex edge, GOOD), then _strip_idle_qubits remaps physical qubits to
0..N_active-1 to make simulation tractable -- but the run simulator's noise model
is keyed to PHYSICAL qubit indices. After the remap, a gate routed onto physical
edge (37,38) is simulated at (0,1), so it fires whatever error is registered at
(0,1) -- the wrong qubits' noise, or none. Routing put the gates on real edges;
the strip then read the noise off the wrong qubits.

Fix: remap the noise model in LOCKSTEP with the circuit strip. Each stripped
qubit carries its physical qubit's real 1q/readout error; each stripped edge
carries its physical edge's real ecr error. Post-routing there are NO off-edge
gates (routing guarantees every 2q gate is a native edge), so no imputation is
needed -- this is a genuine, fully-physical label.
"""
from __future__ import annotations

import sys
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import hellinger_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loader_v2 import read_qasm  # noqa: E402


def remap_noise_model(base: NoiseModel, phys2new: dict) -> NoiseModel:
    """New NoiseModel carrying each physical qubit/edge's real error at its
    stripped index. Only errors whose qubits are all active are copied."""
    nm = NoiseModel(basis_gates=base.basis_gates)
    for gate, d in base._local_quantum_errors.items():
        for qtuple, err in d.items():
            if all(q in phys2new for q in qtuple):
                nm.add_quantum_error(err, gate, [phys2new[q] for q in qtuple],
                                     warnings=False)
    for qtuple, ro in base._local_readout_errors.items():
        if all(q in phys2new for q in qtuple):
            nm.add_readout_error(ro, [phys2new[q] for q in qtuple], warnings=False)
    return nm


def strip_with_map(tcirc):
    """Rebuild tcirc on active qubits; return (stripped, phys2new)."""
    active = sorted({tcirc.find_bit(q).index for op in tcirc.data for q in op.qubits})
    phys2new = {p: i for i, p in enumerate(active)}
    new = QuantumCircuit(len(active), tcirc.num_clbits)
    for op in tcirc.data:
        nq = [new.qubits[phys2new[tcirc.find_bit(q).index]] for q in op.qubits]
        nc = [new.clbits[tcirc.find_bit(c).index] for c in op.clbits]
        new.append(op.operation, nq, nc)
    return new, phys2new


def routed_2q_pairs(tcirc):
    return [(tcirc.find_bit(o.qubits[0]).index, tcirc.find_bit(o.qubits[1]).index)
            for o in tcirc.data if len(o.qubits) == 2]


def compute_fixed(qc, ideal_sim, routed_target, base_nm, shots, seed, opt_level):
    if qc.num_clbits == 0:
        qc = qc.copy(); qc.measure_all()
    tcirc_full = transpile(qc, routed_target, optimization_level=opt_level,
                           seed_transpiler=seed)
    tcirc, phys2new = strip_with_map(tcirc_full)
    fixed_nm = remap_noise_model(base_nm, phys2new)
    fixed_sim = AerSimulator(noise_model=fixed_nm, seed_simulator=seed)
    ic = ideal_sim.run(tcirc, shots=shots).result().get_counts()
    nc = fixed_sim.run(tcirc, shots=shots).result().get_counts()
    return hellinger_fidelity(ic, nc), tcirc_full, tcirc, phys2new


# --------------------------------------------------------------- validation
def _ground_truth(qc, ideal_sim, routed_target, base_nm, shots, seed, opt_level):
    """Ground truth: keep PHYSICAL indices (register sized to max active+1),
    simulate on the base noise model directly. Only feasible when max active
    index is small. Confirms the remapped/stripped sim matches."""
    if qc.num_clbits == 0:
        qc = qc.copy(); qc.measure_all()
    tcirc_full = transpile(qc, routed_target, optimization_level=opt_level,
                           seed_transpiler=seed)
    active = sorted({tcirc_full.find_bit(q).index for op in tcirc_full.data for q in op.qubits})
    width = max(active) + 1
    if width > 22:
        return None, width
    keep = QuantumCircuit(width, tcirc_full.num_clbits)
    for op in tcirc_full.data:
        qi = [keep.qubits[tcirc_full.find_bit(q).index] for q in op.qubits]
        ci = [keep.clbits[tcirc_full.find_bit(c).index] for c in op.clbits]
        keep.append(op.operation, qi, ci)
    gt_sim = AerSimulator(noise_model=base_nm, seed_simulator=seed)
    ic = ideal_sim.run(keep, shots=shots).result().get_counts()
    nc = gt_sim.run(keep, shots=shots).result().get_counts()
    return hellinger_fidelity(ic, nc), width


def main():
    dev = FakeBrisbane()
    base_nm = AerSimulator.from_backend(dev).options.noise_model
    routed_target = AerSimulator.from_backend(dev)  # full backend: coupling map + noise (for transpile)
    ideal_sim = AerSimulator(seed_simulator=42)
    edges = set(base_nm._local_quantum_errors["ecr"])

    qdir = "data/corpus_v2_20q/corpus_v2/qasm"
    tests = ["ghz_indep_none_4_s0", "wstate_indep_none_4_s0", "bv_indep_none_5_s0",
             "ghz_indep_none_5_s0", "dj_indep_none_5_s0"]
    print(f"{'circuit':28s} {'fid_fixed':>9s} {'fid_gt':>8s} {'match':>6s} "
          f"{'offedge_postroute':>17s} {'remap_faithful':>14s}")
    for stem in tests:
        f = Path(qdir) / f"{stem}.qasm"
        if not f.exists():
            print(f"{stem}: missing"); continue
        qc = read_qasm(f)
        fid, tfull, tstrip, p2n = compute_fixed(qc, ideal_sim, routed_target, base_nm, 8192, 42, 1)
        # (1) post-routing off-edge count (should be 0)
        pairs = routed_2q_pairs(tfull)
        offedge = sum(1 for (a, b) in pairs if (a, b) not in edges and (b, a) not in edges)
        # (2) remap faithfulness: stripped-edge error object IS the physical-edge error
        fixed_nm = remap_noise_model(base_nm, p2n)
        new2phys = {v: k for k, v in p2n.items()}
        faithful = True
        for (ma, mb), err in fixed_nm._local_quantum_errors.get("ecr", {}).items():
            pa, pb = new2phys[ma], new2phys[mb]
            src = base_nm._local_quantum_errors["ecr"].get((pa, pb))
            if src is None or err.to_dict() != src.to_dict():
                faithful = False; break
        gt, width = _ground_truth(qc, ideal_sim, routed_target, base_nm, 8192, 42, 1)
        gt_s = f"{gt:.4f}" if gt is not None else f"n/a(w={width})"
        match = "yes" if (gt is not None and abs(fid - gt) < 0.03) else ("--" if gt is None else "NO")
        print(f"{stem:28s} {fid:>9.4f} {gt_s:>8s} {match:>6s} {offedge:>17d} {str(faithful):>14s}")


if __name__ == "__main__":
    main()
