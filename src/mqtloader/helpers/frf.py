#!/usr/bin/env python3
"""
frf.py — noise-remap utilities for the route-then-strip labeling stage.

Developed with assistance from Claude Opus 4.7/4.8.

After a circuit is routed onto a device, the labeling stages strip it down to its
ACTIVE qubits so the noisy simulation stays tractable. That strip renumbers
physical qubit indices to 0..N_active-1 — but a device noise model is keyed to the
PHYSICAL indices, so a gate routed onto physical edge (37,38) would otherwise be
simulated with the error registered at (0,1): the wrong qubits' noise, or none.

These two helpers keep the noise model in lockstep with the strip:

    strip_with_map(tcirc)        -> (stripped_circuit, phys2new)
        Rebuild the routed circuit on its active qubits and return the
        physical -> stripped index map.

    remap_noise_model(base, p2n) -> NoiseModel
        Re-key each physical qubit/edge's real 1q / readout / ecr error onto its
        stripped index, using the map from strip_with_map. Routing guarantees every
        2q gate is a native edge, so no error needs to be imputed — the remapped
        model is fully physical.

Consumed by post_transpile/mirror_polarization_label.py, post_transpile/
swap_features.py, and run_pipeline_with_workers.py.
"""
from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel


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
