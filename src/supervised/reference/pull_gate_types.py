"""
pull_gate_types.py — gate-vocabulary scanner.

Drop-in reconstruction of the helper Steven's GraphSAGE script imports
(`extract_gates_with_legacy_support`). It is NOT a published package — the only
references anywhere are the import lines in his scripts — so this reproduces its one
job: scan a directory of QASM files and return the unique gate names.

Fidelity guarantee: the names are pulled through the IDENTICAL code path the graph
builder uses per circuit ([node.op.name for node in circuit_to_dag(qc).op_nodes()]),
so the global vocabulary and the per-circuit count vectors are consistent by
construction. Column order is irrelevant to the model (the head's first linear layer
is permutation-invariant in its inputs), so only the vocabulary CONTENT matters, and
that is fixed by using his extraction. "Legacy support" = falling back to Qiskit's
QASM-2 LEGACY_CUSTOM_INSTRUCTIONS loader when the plain loader rejects an older file.
"""
import os
import glob
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag


def _load(path):
    try:
        return QuantumCircuit.from_qasm_file(path)          # the loader his build uses
    except Exception:
        from qiskit import qasm2                            # legacy-instruction fallback
        return qasm2.load(path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)


def extract_gates_with_legacy_support(qasm_files_path):
    """Sorted unique gate names across every .qasm in the directory, via the same
    op.name extraction the graph builder uses per circuit."""
    files = sorted(glob.glob(os.path.join(qasm_files_path, "*.qasm")))
    if not files:
        raise FileNotFoundError(f"no .qasm files found in {qasm_files_path}")
    names = set()
    for f in files:
        dag = circuit_to_dag(_load(f))
        for node in dag.op_nodes():
            names.add(node.op.name)
    return sorted(names)
