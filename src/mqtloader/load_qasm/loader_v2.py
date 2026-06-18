#!/usr/bin/env python3
# Module developed with AI assistance (Claude). See PROVENANCE.md for the
# per-component breakdown of reused / lifted-with-citation / AI-assisted code.
"""
loader.py — Generate a STRUCTURALLY-DIVERSE MQT Bench QASM corpus.

Bootstraps a corpus from zero. After running, ./data/qasm/ contains a flat
directory of .qasm files with structured names that downstream code can parse:

    {algorithm}_{level}_{target}_{qubits}_s{seed_idx}.qasm

This version targets the SEVEN families that admit genuine STRUCTURAL variation
(distinct qubit-interaction graphs across seeds), which the previous loader did
not produce: it never forwarded `seed` to get_benchmark, so the interaction
graph was pinned to MQT's default while only rotation ANGLES varied. The fix is
per-family structural randomization:

    qaoa           seed -> Erdos-Renyi cost graph         (count VARYING)
    graphstate     seed + degree -> random regular graph  (count-matched)
    randomcircuit  qiskit random_circuit(seed) BYPASS      (count varying; MQT
                                                            hardcodes seed=10)
    vqe_two_local  explicit random connected pair-list     (count-matched)
    vqe_su2        explicit random connected pair-list     (count-matched)
    vqe_real_amp   explicit random connected pair-list     (count-matched)
    vqe_ranged     two_local ansatz, edge count ~ U(LO,HI)*N per seed
                                                            (count VARYING; fills
                                                            the mid-density gap)

NOTE on the three pinned VQEs: at a given seed they produce the IDENTICAL
interaction graph (they differ only in rotation blocks, which don't touch the
graph). Treat them as ONE structural source for the SWAP target; they differ
only in the fidelity/entropy channel. vqe_ranged is a count-VARYING structural
probe for SWAP-arm input coverage; disclose it as a designed coverage choice,
not a representative workload, and never use it as an OOD test target.

Dedup is on the INTERACTION-GRAPH LAPLACIAN SPECTRUM, not QASM bytes: per-seed
angle binding makes structurally-identical circuits byte-distinct, so SHA dedup
would log isomorphic-but-relabeled graphs as "distinct". The spectrum is an
isomorphism invariant (modulo rare cospectral graphs) and is exactly the routing-
relevant object, so it is the correct dedup key. VERIFIED to match features.py
g_topo spectrum exactly on all families.

Usage:
    python loader.py                                          # all 7 families, defaults
    python loader.py --qubits 7 20 --k 25                     # the sized corpus
    python loader.py --algorithms qaoa graphstate --qubits 8 20 --k 30
    python loader.py --algorithms vqe_ranged --vqe-edge-range 0.8 2.2 --qubits 7 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Families this loader knows how to structurally randomize.
STRUCTURAL_FAMILIES = (
    "qaoa", "graphstate", "randomcircuit",
    "vqe_two_local", "vqe_su2", "vqe_real_amp",
    "vqe_ranged",
)
VQE_FAMILIES = {"vqe_two_local", "vqe_su2", "vqe_real_amp"}
# vqe_ranged: count-VARYING VQE (edge count sampled from a range per seed),
# built on the two_local ansatz. Fills the mid-density coverage gap.
VQE_RANGED_BASE = "vqe_two_local"

_LEVEL = {"alg": "ALG", "indep": "INDEP",
          "nativegates": "NATIVEGATES", "mapped": "MAPPED"}

# Spectra rounded to this many decimals before hashing for dedup.
_SPEC_ROUND = 6


def read_qasm(path):
    """Load a QASM file from this corpus into a Qiskit QuantumCircuit."""
    from qiskit import qasm2
    return qasm2.load(str(path),
                      custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)


def require_deps() -> None:
    try:
        import mqt.bench   # noqa: F401
        import qiskit.qasm2  # noqa: F401
        import networkx     # noqa: F401
        import numpy        # noqa: F401
    except ImportError as e:
        print(f"ERROR: missing dependency ({e.name}). "
              "pip install mqt.bench qiskit networkx numpy", file=sys.stderr)
        sys.exit(2)


# --------------------------------------------------------------------------- #
# Structural fingerprint (dedup key) — Laplacian spectrum of the interaction
# graph. Multi-qubit ops clique-expand; barriers/measures excluded. Matches the
# `_topology` graph convention in features.py (source="cliques", unweighted).
# --------------------------------------------------------------------------- #
def interaction_spectrum(qc) -> tuple:
    import networkx as nx
    import numpy as np
    G = nx.Graph()
    G.add_nodes_from(range(qc.num_qubits))
    for op in qc.data:
        name = op.operation.name
        if name in ("barrier", "measure", "reset", "delay", "snapshot"):
            continue
        idx = [qc.find_bit(q).index for q in op.qubits]
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                G.add_edge(idx[i], idx[j])
    L = nx.laplacian_matrix(G).toarray().astype(float)
    eig = np.sort(np.linalg.eigvalsh(L))
    return tuple(np.round(eig, _SPEC_ROUND).tolist())


# --------------------------------------------------------------------------- #
# Per-family structural generators. Each returns a parametric QuantumCircuit
# for the given (n, seed); angle binding (where applicable) happens in generate().
# --------------------------------------------------------------------------- #
def _make_qaoa(n, seed, level_enum):
    from mqt.bench import get_benchmark
    return get_benchmark(benchmark="qaoa", level=level_enum, circuit_size=n,
                         random_parameters=False, seed=seed)


def _make_graphstate(n, seed, level_enum, degree=3):
    from mqt.bench import get_benchmark
    # degree must be < n and n*degree even for a regular graph to exist.
    d = degree
    if d >= n:
        d = n - 1
    if (n * d) % 2 != 0:
        d -= 1
    if d < 1:
        raise ValueError(f"no regular graph for n={n}, degree={degree}")
    return get_benchmark(benchmark="graphstate", level=level_enum, circuit_size=n,
                         random_parameters=False, seed=seed, degree=d)


def _make_randomcircuit(n, seed, level_enum):
    # BYPASS MQT (hardcodes random_circuit(..., seed=10)); call qiskit directly.
    from qiskit.circuit.random import random_circuit
    qc = random_circuit(n, n * 2, measure=False, seed=seed)
    qc.name = "randomcircuit"
    return qc


def _connected_random_entanglement(n, seed, n_edges, max_tries=200):
    """Sample a CONNECTED random entanglement map of exactly n_edges pairs."""
    import numpy as np
    import networkx as nx
    rng = np.random.default_rng(seed)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for _ in range(max_tries):
        sel = rng.choice(len(all_pairs), size=n_edges, replace=False)
        ent = [all_pairs[i] for i in sel]
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(ent)
        if nx.is_connected(G):
            return [list(e) for e in ent]
    # Fallback: spanning path + random extras guarantees connectivity.
    base = [[i, i + 1] for i in range(n - 1)]
    extra = [list(all_pairs[i]) for i in
             rng.choice(len(all_pairs), size=max(0, n_edges - (n - 1)),
                        replace=False)]
    return base + extra


def _make_vqe(module_name, n, seed, n_edges_frac=1.5, connected=True,
              edge_range=None):
    from importlib import import_module
    import numpy as np
    import networkx as nx
    cc = import_module(f"mqt.bench.benchmarks.{module_name}").create_circuit
    # Edge count: a fixed multiple of n (count-MATCHED), OR sampled from a
    # [lo, hi]*n range per seed (count-VARYING, fills the mid-density gap
    # between graphstate's sparsity and qaoa/randomcircuit's higher density).
    cmax = n * (n - 1) // 2
    if edge_range is not None:
        lo, hi = edge_range
        rng_e = np.random.default_rng(seed * 7919 + 1)  # decorrelate from map RNG
        frac = rng_e.uniform(lo, hi)
        n_edges = int(round(frac * n))
    else:
        n_edges = int(round(n_edges_frac * n))
    n_edges = max(min(n_edges, cmax), n - 1)  # spanning-tree floor, complete-graph cap
    if connected:
        ent = _connected_random_entanglement(n, seed, n_edges)
    else:
        rng = np.random.default_rng(seed)
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        sel = rng.choice(len(all_pairs), size=n_edges, replace=False)
        ent = [list(all_pairs[i]) for i in sel]
    return cc(n, entanglement=ent)


def make_circuit(algo, n, seed, level_enum, degree, vqe_edge_frac, vqe_connected,
                 vqe_edge_range=(0.8, 2.2)):
    if algo == "qaoa":
        return _make_qaoa(n, seed, level_enum)
    if algo == "graphstate":
        return _make_graphstate(n, seed, level_enum, degree=degree)
    if algo == "randomcircuit":
        return _make_randomcircuit(n, seed, level_enum)
    if algo == "vqe_ranged":
        return _make_vqe(VQE_RANGED_BASE, n, seed, connected=vqe_connected,
                         edge_range=vqe_edge_range)
    if algo in VQE_FAMILIES:
        return _make_vqe(algo, n, seed, n_edges_frac=vqe_edge_frac,
                         connected=vqe_connected)
    raise ValueError(f"{algo} is not a structural family this loader handles. "
                     f"Known: {STRUCTURAL_FAMILIES}")


def filename_for(algo, level, target, n, seed_idx):
    target_part = "none" if level in ("alg", "indep") else target
    return f"{algo}_{level}_{target_part}_{n}_s{seed_idx}.qasm"


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a structurally-diverse MQT Bench QASM corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out", default="data/qasm")
    p.add_argument("--algorithms", nargs="+", default=list(STRUCTURAL_FAMILIES))
    p.add_argument("--qubits", nargs=2, type=int, default=[4, 23],
                   metavar=("MIN", "MAX"),
                   help="Inclusive qubit range (20 values 4..23 -> ~k per N).")
    p.add_argument("--k", type=int, default=20,
                   help="Target DISTINCT structures per (algo, N). The loader "
                        "pulls seeds until k distinct structures or --max-seeds.")
    p.add_argument("--max-seeds", type=int, default=400,
                   help="Seed budget per (algo, N) before giving up on reaching k "
                        "(low-N count-matched families saturate below k).")
    p.add_argument("--level", default="indep", choices=list(_LEVEL))
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--graphstate-degree", type=int, default=3)
    p.add_argument("--vqe-edge-frac", type=float, default=1.5,
                   help="VQE entanglement edges = round(frac*N), count-matched per N.")
    p.add_argument("--vqe-edge-range", nargs=2, type=float, default=[0.8, 2.2],
                   metavar=("LO", "HI"),
                   help="For the count-VARYING 'vqe_ranged' family: edge count per "
                        "seed = round(U(LO,HI)*N). Fills the mid-density gap.")
    p.add_argument("--vqe-allow-disconnected", action="store_true",
                   help="Allow disconnected VQE maps (default: connected-only).")
    return p.parse_args()


def generate(args):
    import numpy as np
    from mqt.bench import BenchmarkLevel
    from qiskit import qasm2

    level_enum = getattr(BenchmarkLevel, _LEVEL[args.level])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    qmin, qmax = args.qubits
    written = skipped = deduped = 0
    coverage = {}  # (algo, n) -> distinct count, for the saturation report

    for algo in args.algorithms:
        for n in range(qmin, qmax + 1):
            seen_spectra = set()
            distinct = 0
            for seed_idx in range(args.max_seeds):
                if distinct >= args.k:
                    break
                seed = args.base_seed + seed_idx
                try:
                    qc = make_circuit(algo, n, seed, level_enum,
                                      args.graphstate_degree, args.vqe_edge_frac,
                                      not args.vqe_allow_disconnected,
                                      vqe_edge_range=tuple(args.vqe_edge_range))
                    # Bind any free parameters per-seed (angles); structure is
                    # already fixed by the generator above.
                    if len(qc.parameters) > 0:
                        rng = np.random.default_rng(seed)
                        qc.assign_parameters(
                            {p: rng.uniform(0, 2 * np.pi) for p in qc.parameters},
                            inplace=True)
                except Exception as e:
                    skipped += 1
                    continue

                spec = interaction_spectrum(qc)
                if spec in seen_spectra:
                    deduped += 1
                    continue
                seen_spectra.add(spec)

                try:
                    if qc.num_clbits == 0:
                        qc_out = qc.copy(); qc_out.measure_all()
                    else:
                        qc_out = qc
                    qasm_text = qasm2.dumps(qc_out)
                except Exception as e:
                    print(f"  skip {algo}@{n}.s{seed_idx} (dump): {e}",
                          file=sys.stderr)
                    skipped += 1
                    continue

                path = out_dir / filename_for(algo, args.level, "none", n, distinct)
                path.write_text(qasm_text)
                written += 1
                distinct += 1
            coverage[(algo, n)] = distinct

    return written, skipped, deduped, coverage


def main():
    require_deps()
    args = parse_args()
    written, skipped, deduped, coverage = generate(args)

    # Saturation report: where did we fall short of k?
    short = {k: v for k, v in coverage.items() if v < args.k}
    print(f"\nWrote {written} QASM files to {args.out} "
          f"({deduped} structural duplicates skipped, {skipped} errors).")
    if short:
        print(f"\nUNDER-SAMPLED cells (<{args.k} distinct structures available):")
        for (algo, n), v in sorted(short.items()):
            print(f"  {algo:>14} n={n:<3} -> {v:>3}/{args.k}")
        print("  (count-matched families saturate at low N; stratify + note.)")


if __name__ == "__main__":
    main()
