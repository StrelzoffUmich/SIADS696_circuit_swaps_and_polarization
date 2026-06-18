#!/usr/bin/env python3
"""
build_validation_corpus_sv.py
==============================
Assemble a NON-TRAINING out-of-distribution validation corpus of QASM circuits from
three independent benchmark suites and write them into a single directory, named in
MQT_Loader's required convention so `run_pipeline.py` can ingest them directly.

Sources
-------
  mqtbench  : mqt.bench, non-training families, level=INDEP            (scalable sweep)
  nwqbench  : PNNL NWQBench generators, run with circuit-neutral shims (scalable sweep)
  qasmbench : PNNL QASMBench curated suite, filtered to range          (fixed instances)

Training families EXCLUDED everywhere (these are what the model was trained on):
    qaoa, graphstate, randomcircuit, vqe_real_amp, vqe_su2, vqe_two_local  (+ vqe_ranged)

Inclusion filter applied to every circuit (measured, not by label):
    actual num_qubits in [n_lo, n_hi]   AND   >= 2 entangling (arity>=2) gates
    AND it serializes to OpenQASM 2.0 and re-parses with QuantumCircuit.from_qasm_file
    (i.e. it is loadable by the same path MQT_Loader uses).

** FILENAME CONVENTION (do not change) **
MQT_Loader parses metadata FROM the filename:
    {algo}_{level}_{target}_{N}_s{seed}.qasm
We map:  algo=<canonical family>   level=indep   target=<source>   N=<num_qubits>
         s{seed}=<within-(family,source,N) index, for uniqueness + reproducibility>
So `qft_indep_nwqbench_8_s0.qasm` -> algo=qft, level=indep, target=nwqbench, n=8.
This makes the suite ('target') a free column in the resulting dataset, and keeps
`algo` = the canonical family so leave-one-family-out grouping works downstream.

Dedup
-----
Optional `--dedupe-graphs`: collapse circuits whose pre-routing interaction graph
(clique graph of arity>=2 ops, the SAME graph extract_features builds with
source="cliques") is ISOMORPHIC, keeping one representative. This gives a corpus of
distinct *structures* rather than distinct algorithm labels. Off by default (full pool);
the manifest carries struct_id either way so you can dedupe later.

Dependencies
------------
    pip install "qiskit>=2,<3" mqt.bench networkx
    git is used to clone NWQBench and QASMBench (only for the suites you request).

Usage
-----
    # full pool, N 3..15, into ./validation_qasm
    python build_validation_corpus_sv.py --out-dir validation_qasm

    # distinct interaction graphs only
    python build_validation_corpus_sv.py --out-dir validation_qasm --dedupe-graphs

    # a subset of suites
    python build_validation_corpus_sv.py --out-dir validation_qasm --sources mqtbench nwqbench

If --out-dir ALREADY EXISTS, this script is idempotent: it first removes any corpus
*.qasm and the old _corpus_manifest.csv from that dir, then writes a fresh set. So a
re-run REPLACES the corpus rather than mixing new circuits with stale leftovers (which
would desync the dir from the manifest and feed orphan circuits to downstream labeling).
Non-corpus files in the dir are left untouched.

Then point MQT_Loader at it (circuits already exist, so skip generation). The corpus is
FLAT (*.qasm directly in --out-dir), which is exactly what all three pipeline stages read:
    python run_pipeline.py --skip-gen --qasm-dir validation_qasm --device FakeBrisbane --n-lo 3 --n-hi 10
n-lo/n-hi there gate POLARIZATION labeling (sim-bound); routing (swap_features) runs on all N.
A manifest (validation_qasm/_corpus_manifest.csv) maps every emitted file to its
source / raw family / canonical family / N / struct_id.
"""
from __future__ import annotations
import argparse, contextlib, io, itertools, json, os, random, shutil, subprocess, sys, tempfile, warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import networkx as nx
from qiskit import QuantumCircuit, qasm2

# ---------------------------------------------------------------------------
TRAIN_FAMILIES = {
    "qaoa", "graphstate", "randomcircuit",
    "vqe_real_amp", "vqe_su2", "vqe_two_local", "vqe_ranged", "vqe",
}

# Families excluded on STRUCTURAL grounds (not training overlap): their INDEP-level
# decomposition of multi-controlled operators synthesizes tens of thousands of 2q gates
# (grover/qwalk route to routed_2q ~1e4-7e4), which (a) is an MCX-synthesis artifact
# rather than pre-routing structure, (b) floors the polarization label to ~0, and
# (c) would be an extreme leverage outlier in the routing target. Matched on canonical
# family, so this drops grover/qwalk from every source.
EXCLUDE_FAMILIES = {"grover", "qwalk", 'factor247'}

# canonical-family folding. Conservative (algorithm-level): variants collapse to the
# algorithm so leave-one-family-out can't leak a 'qft' between train and test. Edit
# here if you prefer to split structurally-distinct variants (e.g. draper vs ripple
# adders) into their own families.
CANON = {
    "w_state": "wstate", "ghz_state": "ghz", "ghz_dynamic": "ghz", "cat_state": "catstate",
    "multiply": "multiplier", "rg_qft_multiplier": "multiplier",
    "hrs_cumulative_multiplier": "multiplier",
    "cdkm_ripple_carry_adder": "adder", "draper_qft_adder": "adder", "full_adder": "adder",
    "half_adder": "adder", "modular_adder": "adder", "vbe_ripple_carry_adder": "adder",
    "qec_5qubit_x": "qec", "qec_9qubit_xyz": "qec", "qec_en": "qec", "qec_sm": "qec",
    "error_correctiond3": "qec", "seven_qubit_steane_code": "qec", "shors_nine_qubit_code": "qec",
    "qpeexact": "qpe", "qpeinexact": "qpe", "pea": "qpe",
    "qf21": "qft_adder", "square_root": "sqrt",
}
def canon(fam: str) -> str:
    f = fam.lower().strip()
    return CANON.get(f, f)

def entangling_count(qc: QuantumCircuit) -> int:
    return sum(1 for inst in qc.data
               if inst.operation.num_qubits >= 2
               and inst.operation.name not in ("barrier", "measure"))

def has_control_flow(qc: QuantumCircuit) -> bool:
    """True if the circuit contains classical control flow (if/else, while, for, switch)
    or a classically-conditioned gate. These are DYNAMIC circuits (mid-circuit measure +
    feedforward, e.g. teleportation's correction, QEC syndrome decoding). The static
    structural feature model (interaction graph, depth, parallelism) is undefined for
    them — extract_features' dag.depth() raises DAGCircuitError — so they are excluded."""
    _CF = {"if_else", "while_loop", "for_loop", "switch_case"}
    for inst in qc.data:
        op = inst.operation
        if op.name in _CF:
            return True
        if getattr(op, "condition", None) is not None:   # legacy classically-conditioned op
            return True
    return False

def clique_graph(qc: QuantumCircuit) -> nx.Graph:
    """Pre-routing interaction graph, identical to extract_features(source='cliques'):
    every arity>=2 op contributes a clique among its qubits."""
    G = nx.Graph(); G.add_nodes_from(range(qc.num_qubits))
    for inst in qc.data:
        op = inst.operation
        if op.num_qubits >= 2 and op.name not in ("barrier", "measure"):
            qs = [qc.find_bit(b).index for b in inst.qubits]
            G.add_edges_from(itertools.combinations(sorted(qs), 2))
    return G

def valid(qc: QuantumCircuit, n_lo: int, n_hi: int) -> bool:
    if not (n_lo <= qc.num_qubits <= n_hi):
        return False
    if entangling_count(qc) < 2:
        return False
    if has_control_flow(qc):                # dynamic circuits break static feature extraction
        return False
    try:                                   # must round-trip through the pipeline loader
        QuantumCircuit.from_qasm_str(qasm2.dumps(qc))
    except Exception:
        return False
    return True

# ---------------------------------------------------------------------------
def gen_mqtbench(n_lo, n_hi):
    """mqt.bench non-training families at INDEP level. Yields (raw_family, qc)."""
    from mqt.bench import get_benchmark, BenchmarkLevel
    from mqt.bench.benchmarks import get_available_benchmark_names
    names = [n for n in sorted(get_available_benchmark_names()) if n not in TRAIN_FAMILIES]
    for nm in names:
        for cs in range(2, n_hi + 1):
            try:
                qc = get_benchmark(benchmark=nm, level=BenchmarkLevel.INDEP, circuit_size=cs)
            except Exception:
                continue
            if valid(qc, n_lo, n_hi):
                yield nm, qc

# NWQBench generators that need APIs we won't fake (would change the circuit) or are
# out of range by construction. Everything else is attempted and filtered.
_NWQ_SKIP = {"vqe", "vqc", "qnn_qf", "cc", "hhl", "grover_search",
             "boolean_satisfaction", "binary_welded_tree", "backup_python_code",
             # statevector_encoder ignores its size arg (k hardwired to 10) and runs
             # `while True: decompose()` on an arbitrary 10-qubit UnitaryGate -> full
             # Shannon synthesis (~4^10 gates). On qiskit<1.0 this is an unkillable
             # multi-minute rust hang; on >=1.0 it ImportErrors on qiskit.extensions.
             # Garbage as a structural sample either way.
             "statevector_encoder"}

def _nwq_shims():
    """Circuit-NEUTRAL compatibility shims for Qiskit>=2 (serialization, removed imports,
    renamed methods). None of these alter circuit construction; the .qasm()->qasm2.dumps
    swap was verified identical against NWQBench's own shipped QASM."""
    import types, qiskit
    QuantumCircuit.qasm = lambda self, *a, **k: qasm2.dumps(self)
    qiskit.execute = lambda *a, **k: types.SimpleNamespace(
        result=lambda: types.SimpleNamespace(get_counts=lambda *a, **k: {}))
    qiskit.BasicAer = types.SimpleNamespace(get_backend=lambda *a, **k: None)
    qiskit.Aer = qiskit.BasicAer
    QuantumCircuit.cnot = QuantumCircuit.cx          # renamed in 1.0
    QuantumCircuit.toffoli = QuantumCircuit.ccx
    QuantumCircuit.fredkin = QuantumCircuit.cswap
    QuantumCircuit.u1 = QuantumCircuit.p             # u1(λ) == p(λ)

def gen_nwqbench(n_lo, n_hi, seed, workdir):
    """Clone NWQBench, run each generator across the size arg, seeded. Yields (fam, qc)."""
    repo = workdir / "nwqbench"
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/pnnl/nwqbench.git", str(repo)],
                       check=True, capture_output=True)
    _nwq_shims()
    bench = repo / "NWQ_Bench"
    cwd0 = os.getcwd()
    for fam_dir in sorted(p for p in bench.iterdir() if p.is_dir()):
        fam = fam_dir.name
        if fam in _NWQ_SKIP:
            continue
        scripts = list(fam_dir.glob("*.py"))
        if not scripts:
            continue
        src = scripts[0].read_text().replace("\nqiskit.\n", "\n#qiskit.\n")  # dead-line fix
        out = workdir / "nwq_out" / fam; out.mkdir(parents=True, exist_ok=True)
        for arg in range(1, n_hi + 1):       # size arg; measured N filters below
            random.seed(seed); np.random.seed(seed)
            os.chdir(out); sys.argv = [scripts[0].name, str(arg)]
            try:
                with contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    exec(compile(src, str(scripts[0]), "exec"),
                         {"__name__": "__main__", "__file__": str(scripts[0])})
            except BaseException:
                pass
        os.chdir(cwd0)
        for qf in sorted(out.glob("qasm/*.qasm")):
            try:
                qc = QuantumCircuit.from_qasm_file(str(qf))
            except Exception:
                continue
            if valid(qc, n_lo, n_hi):
                yield fam, qc

def gen_qasmbench(n_lo, n_hi, workdir):
    """Clone QASMBench, take curated circuits in range from small/ and medium/."""
    repo = workdir / "QASMBench"
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/pnnl/QASMBench.git", str(repo)],
                       check=True, capture_output=True)
    for sub in ("small", "medium"):
        d = repo / sub
        if not d.is_dir():
            continue
        for qf in sorted(d.glob("*/*.qasm")):
            stem = qf.stem
            fam = stem.rsplit("_n", 1)[0] if "_n" in stem else stem
            if canon(fam) in {canon(t) for t in TRAIN_FAMILIES}:
                continue
            try:
                qc = QuantumCircuit.from_qasm_file(str(qf))
            except Exception:
                continue
            if valid(qc, n_lo, n_hi):
                yield fam, qc

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, help="QASM output directory for MQT_Loader")
    ap.add_argument("--sources", nargs="+", default=["mqtbench", "nwqbench", "qasmbench"],
                    choices=["mqtbench", "nwqbench", "qasmbench"])
    ap.add_argument("--n-lo", type=int, default=3)
    ap.add_argument("--n-hi", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dedupe-graphs", action="store_true",
                    help="collapse isomorphic interaction graphs across the whole pool")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    # Idempotent re-runs: if the dir already exists, clear PRIOR corpus files first so the
    # result (and the manifest) reflect EXACTLY this build. Without this, an existing out-dir
    # silently mixes stale circuits with new ones -- the manifest undercounts and downstream
    # labeling picks up orphans. Only touches files this script itself emits.
    prior = list(out.glob("*.qasm"))
    man_old = out / "_corpus_manifest.csv"
    if man_old.exists():
        prior.append(man_old)
    for p in prior:
        p.unlink()
    if prior:
        print(f"[out] {out} existed; cleared {len(prior)} prior corpus file(s) before rebuild",
              file=sys.stderr)
    workdir = Path(tempfile.mkdtemp(prefix="valcorpus_"))
    GEN = {"mqtbench": lambda: gen_mqtbench(args.n_lo, args.n_hi),
           "nwqbench": lambda: gen_nwqbench(args.n_lo, args.n_hi, args.seed, workdir),
           "qasmbench": lambda: gen_qasmbench(args.n_lo, args.n_hi, workdir)}

    # collect every (source, raw_family, qc) that passes the filter
    records = []   # dict(source, raw, canon, nq, qc, G)
    for src in args.sources:
        print(f"[{src}] generating ...", file=sys.stderr)
        n0 = len(records)
        for raw, qc in GEN[src]():
            if canon(raw) in EXCLUDE_FAMILIES:        # structural exclusions (grover/qwalk)
                continue
            records.append(dict(source=src, raw=raw, canon=canon(raw),
                                 nq=qc.num_qubits, qc=qc, G=clique_graph(qc)))
        print(f"[{src}] {len(records)-n0} circuits passed filter", file=sys.stderr)

    # optional dedup by interaction-graph isomorphism (bucket by cheap invariants first)
    def sig(G):
        return (G.number_of_nodes(), G.number_of_edges(),
                tuple(sorted(d for _, d in G.degree())),
                sum(nx.triangles(G).values()) // 3)
    struct_of = {}                # id(record) -> struct_id   (assigned for ALL records)
    buckets = defaultdict(list)
    for r in records:
        buckets[sig(r["G"])].append(r)
    sid = 0
    for b in buckets.values():
        reps = []                 # (G, struct_id) representatives within this bucket
        for r in b:
            for G, s in reps:
                if nx.is_isomorphic(G, r["G"]):
                    struct_of[id(r)] = s; break
            else:
                reps.append((r["G"], sid)); struct_of[id(r)] = sid; sid += 1

    # emit set: deduped -> one representative per struct_id; else -> all
    if args.dedupe_graphs:
        seen = set(); emit = []
        for r in records:
            s = struct_of[id(r)]
            if s not in seen:
                seen.add(s); emit.append(r)
    else:
        emit = records

    # write QASM in MQT_Loader convention + manifest
    seed_ctr = defaultdict(int)
    man = []
    for r in emit:
        key = (r["canon"], r["source"], r["nq"])
        idx = seed_ctr[key]; seed_ctr[key] += 1
        fname = f"{r['canon']}_indep_{r['source']}_{r['nq']}_s{idx}.qasm"
        (out / fname).write_text(qasm2.dumps(r["qc"]))
        man.append(dict(file=fname, source=r["source"], family_raw=r["raw"],
                        family_canonical=r["canon"], num_qubits=r["nq"],
                        struct_id=f"G{struct_of[id(r)]:04d}",
                        edges=r["G"].number_of_edges()))

    import csv
    with open(out / "_corpus_manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(man[0].keys())); w.writeheader(); w.writerows(man)

    n_struct = len({m["struct_id"] for m in man})
    n_fam = len({m["family_canonical"] for m in man})
    print(f"\nwrote {len(man)} QASM files to {out}", file=sys.stderr)
    print(f"  {n_fam} canonical families | {n_struct} distinct interaction graphs", file=sys.stderr)
    print(f"  manifest: {out/'_corpus_manifest.csv'}", file=sys.stderr)
    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
