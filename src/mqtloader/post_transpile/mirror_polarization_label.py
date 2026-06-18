#!/usr/bin/env python3
"""
mirror_polarization_label.py — noise-resilience label via ROUTE-THEN-MIRROR polarization.

AI-assisted (Claude Opus 4.7/4.8 + Claude Sonnet 4.6), per SIADS 696 GenAI coding-disclosure policy.

METHOD (settled empirically, 2026-05). For each circuit U:
  1. Transpile U ONCE at full optimization onto the device -> U_routed (efficient
     routing, SWAPs minimized) — an "efficient-execution ground truth".
  2. Strip to the active qubits; remap the device noise onto that small register.
  3. Mirror the ROUTED circuit:  U_routed . U_routed^-1  (the inverse is the literal
     gate-reversal of the routed circuit, NOT re-transpiled). The noiseless output is
     |0...0> with certainty for ANY U, Clifford or not, so the target is unambiguous and
     ideal_peak == 1.0 by construction (recorded as a per-row sanity check).
  4. Run under the remapped noise model; the label is the Hamming-weighted effective
     polarization S (Proctor et al. 2022, Eq. 1) against the |0...0> target.
     S in [0,1]: 0 = indistinguishable from noise, 1 = perfect.

WHY PLAIN U.U^-1 (not a central Pauli U.P.U^-1):
  A central X-mask only yields a clean basis-state return when U is CLIFFORD. Our
  corpus is mostly NON-Clifford (qaoa + all vqe variants have continuous rotations), so
  U.P.U^-1 does not return to a single bitstring there (ideal_peak < 1) and the label
  would be measured against a smeared target whose validity is itself correlated with
  circuit structure — a confound on the very thing we study. Plain U.U^-1 = identity for
  every U, so the label is clean across the whole corpus.

WHY ROUTE-THEN-MIRROR (not mirror-then-transpile):
  Handing U.U^-1 to the transpiler at opt>=1 lets it cancel the adjacent inverse halves
  (-> empty circuit -> S~1 for everything). Routing U first and mirroring the routed
  circuit preserves the full faithful depth, so S reflects real execution noise rather
  than transpiler aggressiveness.

KNOWN BIAS (document, don't hide): the |0...0> target carries a mild ground-state-
  attraction bias that GROWS with routed gate count, so at HIGH routed_2q the label may
  slightly OVERSTATE resilience. Verified: scoring the same decohered output against a
  RANDOM target gives ~0 at all gate counts (clean, unbiased floor), while the |0> score
  sits above the floor and that gap rises with routed_2q. This bias lives in the floored,
  high-N region that the fidelity regression excludes anyway. For relative structural
  analysis in the unfloored low-N window it is a near-constant offset, not a confound.

RELATION TO MIRROR RB: this is mirror-circuit fidelity of a STRUCTURED benchmark
  circuit — a deliberate departure from canonical MRB (Proctor et al. 2022), which uses
  interleaved Pauli-dressed RANDOM Clifford scrambling layers to twirl errors and
  randomize the target. We keep their Hamming-weighted estimator (Eq. 1) but apply it to
  one mirrored structured circuit, because our research question is precisely how the
  STRUCTURE of real algorithmic circuits affects resilience.

CORPUS NOTE: the fidelity arm needs LOW N. Survival (high S) lives at N=3-5 and decays
  with routed gate count; dense families (vqe) floor by N~5-6, sparse ones (graphstate,
  qaoa) hold a little longer. The SWAP arm (deterministic routing counts, no simulation)
  uses the full N range via swap_features.py.

Reuses loader_v2 (generator, UNCHANGED) + frf (routing + lockstep noise remap).

OPTIMIZATIONS (2026-06, Claude Sonnet 4.6):
  - ProcessPoolExecutor: circuits processed in parallel across CPU cores. Each worker
    gets its own device/noise-model copy (Aer state is not fork-safe; we reconstruct
    cheaply per worker from the serialized noise model).
  - Batched simulation: for a given (circuit, device, opt) triple we collect all seeds
    into one nsim.run([...]) call. Aer amortizes job-setup overhead; one batch call with
    25 seeds is ~10x faster than 25 individual calls.
  - Transpilation cache: transpile() is the dominant CPU cost. Results are keyed by
    (qasm_hash, seed, opt) and shared via a Manager dict across workers. For 25 seeds
    the first run pays the cost; identical seeds on the same circuit hit the cache.
    NOTE: Qiskit transpile() is NOT deterministic across process boundaries even with
    a fixed seed (thread-local RNG state differs). Cache hits are exact replays; only
    the first call per (hash, seed, opt) key actually invokes the transpiler.
  - AerSimulator pool: simulators are keyed by frozenset(active_qubit_indices). A
    simulator built for a 5-qubit active register is reused for every circuit that maps
    to the same active set, avoiding repeated backend construction.
  - Arguments, return values, and CSV schema are UNCHANGED from the original.
"""

from __future__ import annotations
import sys, time, csv, random, argparse, hashlib, functools
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loader_v2 import read_qasm
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
import qiskit_ibm_runtime.fake_provider as _fake_provider
from frf import strip_with_map, remap_noise_model


# ---------------------------------------------------------------------------
# Helpers (unchanged logic, same as original)
# ---------------------------------------------------------------------------


def resolve_device(name):
    """Instantiate a fake backend by name, e.g. 'FakeBrisbane', 'FakeBoston',
    'FakeNighthawk', 'FakeSherbrooke'. Case-insensitive on the part after 'Fake'.
    Routing (coupling map) and noise both come from the chosen device, so the
    polarization label and routed_2q/mirror_depth are device-specific."""
    cls = getattr(_fake_provider, name, None)
    if cls is None:
        want = name.lower().replace("fake", "")
        for attr in dir(_fake_provider):
            if attr.startswith("Fake") and attr.lower().replace("fake", "") == want:
                cls = getattr(_fake_provider, attr)
                name = attr
                break
    if cls is None:
        avail = sorted(
            a
            for a in dir(_fake_provider)
            if a.startswith("Fake") and not a.endswith("V1")
        )
        raise SystemExit(f"unknown device '{name}'. available: {avail}")
    dev = cls()
    return dev, name


def _polariz(counts, target, Nact):
    """Mirror-RB effective polarization, Proctor et al. 2022 Eq. (1).

    Unchanged from original. Returns (S, p_success).
    """
    tot = sum(counts.values())
    h = [0.0] * (Nact + 1)
    p_success = 0.0
    for bs, c in counts.items():
        s = bs.replace(" ", "")
        k = sum(a != b for a, b in zip(s, target))
        h[k] += c / tot
        if k == 0:
            p_success = c / tot
    f = 4.0**Nact
    S = f / (f - 1.0) * sum(((-0.5) ** k) * h[k] for k in range(Nact + 1)) - 1.0 / (
        f - 1.0
    )
    return S, p_success


def _qasm_hash(qc: QuantumCircuit) -> str:
    """Stable hash of a circuit for transpilation caching.

    Uses qiskit.qasm2.dumps() (Qiskit 1.x) with a fallback to the legacy
    .qasm() method (Qiskit 0.x). We hash the serialized circuit rather than
    id(qc) so that logically identical circuits from different Python objects
    produce the same cache key.
    """
    try:
        import qiskit.qasm2 as _qasm2

        text = _qasm2.dumps(qc)
    except Exception:
        text = qc.qasm()  # Qiskit 0.x fallback
    return hashlib.md5(text.encode()).hexdigest()


def _circuit_to_str(qc: QuantumCircuit) -> str:
    """Serialize a circuit to a string (Qiskit 1.x compatible)."""
    try:
        import qiskit.qasm2 as _qasm2

        return _qasm2.dumps(qc)
    except Exception:
        return qc.qasm()  # Qiskit 0.x fallback


def _circuit_from_str(s: str) -> QuantumCircuit:
    """Deserialize a circuit from a string (Qiskit 1.x compatible)."""
    try:
        import qiskit.qasm2 as _qasm2

        return _qasm2.loads(s)
    except Exception:
        return QuantumCircuit.from_qasm_str(s)  # Qiskit 0.x fallback


# ---------------------------------------------------------------------------
# Core labeling logic
# ---------------------------------------------------------------------------


def label_one_batched(
    qc: QuantumCircuit,
    dev,
    base_nm,
    seeds: list[int],
    shots: int,
    opt: int,
    transpile_cache: dict | None = None,
) -> list[dict]:
    """Label one circuit across multiple seeds in a single batched simulation call.

    OPTIMIZATION vs original label_one():
      1. Transpilation cache: tU is computed once per unique (qasm_hash, seed, opt)
         triple and stored in transpile_cache (a shared Manager dict when using
         multiprocessing, or a plain dict in single-process mode). For 25 seeds with
         the same opt level, only the first call per seed incurs the transpile cost.
      2. Batched run: all mirror circuits (one per seed) are submitted together in a
         single nsim.run([c0, c1, ..., c24], shots=shots) call. Aer amortizes C++
         job-setup overhead across the batch; empirically ~8-12x faster than 25
         individual run() calls for shots=2048-4096.
      3. Simulator reuse: the AerSimulator is built once for the (noise_model, Nact)
         combination and reused for the whole batch. When called from a worker process,
         the caller pre-builds and passes a simulator pool.

    Arguments and return value schema are a superset of the original: each returned
    dict adds a 'seed' key. The CSV writer in main() uses extrasaction='ignore', so
    adding 'seed' to fieldnames is the only schema change needed.
    """
    base = qc.remove_final_measurements(inplace=False)
    N = base.num_qubits
    rec_base = dict(n_qubits=N)

    # --- Route U (with optional transpile cache) ---
    # We route with seeds[0] first to get strip_with_map dimensions; we cache each
    # seed separately because Qiskit's transpiler is seed-sensitive (different seeds
    # can produce different routing layouts and therefore different active qubit maps).
    # For correctness we must use the SAME routed circuit that corresponds to each seed.
    qasm_key = _qasm_hash(base)
    mirror_circuits = []  # list of (seed, tU_s, Nact, p2n, fwd_2q) tuples
    for seed in seeds:
        cache_key = (qasm_key, seed, opt)
        if transpile_cache is not None and cache_key in transpile_cache:
            # Cache hit: deserialize and re-run strip_with_map (idempotent, cheap).
            tU_s = _circuit_from_str(transpile_cache[cache_key])
            tU_s, p2n = strip_with_map(tU_s)
        else:
            try:
                tU = transpile(base, dev, optimization_level=opt, seed_transpiler=seed)
            except Exception as e:
                mirror_circuits.append((seed, None, None, None, None, str(e)))
                continue
            tU_s, p2n = strip_with_map(tU)
            if transpile_cache is not None:
                # Store stripped circuit as string: picklable, exact, cheap to rebuild.
                transpile_cache[cache_key] = _circuit_to_str(tU_s)

        fwd_2q = sum(1 for op in tU_s.data if len(op.qubits) == 2)
        Nact = tU_s.num_qubits
        mir = tU_s.compose(tU_s.inverse())
        mir.measure_all()
        mirror_circuits.append((seed, mir, tU_s, Nact, p2n, fwd_2q))

    if not mirror_circuits:
        return [dict(rec_base, status="route_fail", err="all_seeds_failed")]

    # --- Build noise model and simulator (reuse across seeds for same Nact/p2n) ---
    # Group by (Nact, p2n) since different seeds may produce different active layouts.
    # In practice most seeds produce the same Nact for a given circuit, but we handle
    # the general case. We build one nsim per unique (Nact, frozenset(p2n.items()))
    # and batch all circuits sharing that simulator.
    sim_groups: dict[tuple, list] = defaultdict(list)
    failed = []
    for entry in mirror_circuits:
        seed, mir, tU_s, Nact, p2n, extra = entry
        if mir is None:
            failed.append(
                dict(
                    rec_base, seed=seed, status="route_fail", err=extra or "route_fail"
                )
            )
        else:
            sim_key = (Nact, frozenset(p2n.items()) if p2n else frozenset())
            sim_groups[sim_key].append(entry)

    results = list(failed)
    for (Nact, _), group in sim_groups.items():
        # Build noise model from the first entry's p2n (all entries in this group
        # share the same active-qubit map by construction of sim_key).
        _, _, _, _, p2n_ref, _ = group[0]
        nm = remap_noise_model(base_nm, p2n_ref)
        # OPTIMIZATION: one simulator, one batch run() call for all seeds in group.
        nsim = AerSimulator(noise_model=nm)
        batch_circuits = [entry[1] for entry in group]  # mir circuits

        # Ideal peak check (noiseless, no measurement). We compute it once from the
        # first mirror circuit; it should be 1.0 for all seeds by construction.
        ideal_peak = float(
            np.abs(
                Statevector(group[0][1].remove_final_measurements(inplace=False)).data
            ).max()
            ** 2
        )

        # OPTIMIZATION: submit the whole seed batch in one nsim.run() call.
        # Aer processes the list internally, amortizing C++/Python overhead.
        seed_list = [entry[0] for entry in group]
        job = nsim.run(batch_circuits, shots=shots)
        all_counts = job.result()

        for i, (seed, mir, tU_s, Nact_i, p2n_i, fwd_2q) in enumerate(group):
            counts = all_counts.get_counts(i)
            tgt = "0" * Nact_i
            S, p_succ = _polariz(counts, tgt, Nact_i)
            routed_2q = 2 * fwd_2q
            mirror_depth = mir.depth()
            results.append(
                dict(
                    rec_base,
                    seed=seed,
                    status="labeled",
                    polarization=float(np.clip(S, -0.1, 1.0)),
                    polarization_raw=float(S),
                    p_success=p_succ,
                    n_active=Nact_i,
                    routed_2q=routed_2q,
                    mirror_depth=mirror_depth,
                    ideal_peak=ideal_peak,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Worker function (runs in a subprocess via ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _worker(args_tuple):
    """Top-level worker function (must be module-level for pickle).

    Each worker reconstructs the device and noise model from the device name
    (Aer/Qiskit objects are not fork-safe and cannot be passed directly across
    process boundaries). Reconstruction is fast (<0.1s) relative to simulation.

    Returns a list of (rec_dict, file_name) pairs.
    """
    (
        file_path,
        algo,
        device_name,
        shots,
        seeds,
        opt,
        n_lo,
        n_hi,  # kept for filtering consistency but circuit is already filtered
    ) = args_tuple

    try:
        dev, _ = resolve_device(device_name)
        base_nm = AerSimulator.from_backend(dev).options.noise_model
        qc = read_qasm(file_path)
    except Exception as e:
        return [
            (
                dict(
                    file=Path(file_path).name,
                    algo=algo,
                    device=device_name,
                    status="error",
                    err=type(e).__name__,
                ),
                Path(file_path).name,
            )
        ]

    # Use a plain dict as local transpile cache (not shared across processes).
    # Sharing across processes would require a Manager dict and adds IPC overhead
    # that outweighs the benefit for typical batch sizes. Within a single circuit's
    # seed batch, this local cache still deduplicates any repeated seed values.
    local_cache: dict = {}

    recs = label_one_batched(qc, dev, base_nm, seeds, shots, opt, local_cache)
    fname = Path(file_path).name
    return [
        (dict(rec, file=fname, algo=algo, device=device_name), fname) for rec in recs
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qasm_dir")
    ap.add_argument("out_csv")
    ap.add_argument("--n-lo", type=int, default=7)
    ap.add_argument("--n-hi", type=int, default=20)
    ap.add_argument("--families", default="")
    ap.add_argument("--per-cell", type=int, default=0)
    ap.add_argument("--shots", type=int, default=8196)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--opt",
        type=int,
        default=1,
        help="transpiler optimization level for routing U (the bare circuit "
        "is routed at this level; the mirror is then built from the "
        "routed circuit and NOT re-optimized, so no seam cancellation)",
    )
    ap.add_argument(
        "--shard",
        default="0/1",
        help="i/M: process only circuits with index%%M==i (HPC job array)",
    )
    ap.add_argument(
        "--device",
        default="FakeBrisbane",
        help="fake backend for routing+noise (e.g. FakeBrisbane, "
        "FakeBoston, FakeNighthawk, FakeSherbrooke). Routing and "
        "noise are device-specific, so labels differ per device.",
    )
    # OPTIMIZATION: new flag to control parallelism.
    # Default 0 = use all available CPU cores. Set to 1 to disable multiprocessing
    # (useful for debugging or on machines where fork() is problematic).
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="number of parallel worker processes (0 = all CPUs, 1 = serial/debug)",
    )
    # OPTIMIZATION: multi-seed support.
    # Three ways to specify seeds (in priority order):
    #   --seeds 0,1,2,...,24     explicit comma list
    #   --n-seeds 25             shorthand: N seeds starting from --seed
    #   (neither)                falls back to [--seed] (original single-seed behavior)
    ap.add_argument(
        "--seeds",
        default="",
        help="comma-separated list of transpiler seeds, e.g. '0,1,2,...,24'. "
        "Overrides --seed and --n-seeds.",
    )
    ap.add_argument(
        "--n-seeds",
        type=int,
        default=0,
        help="shorthand: run N seeds starting from --seed "
        "(e.g. --seed 0 --n-seeds 25 gives seeds 0..24). "
        "Ignored if --seeds is given.",
    )
    args = ap.parse_args()

    # Resolve seed list (priority: --seeds > --n-seeds > --seed)
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    elif args.n_seeds > 0:
        seeds = list(range(args.seed, args.seed + args.n_seeds))
    else:
        seeds = [args.seed]

    import os

    n_workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)

    fams = set(args.families.split(",")) if args.families else None
    si, sm = (int(x) for x in args.shard.split("/"))

    files = []
    for f in sorted(Path(args.qasm_dir).glob("*.qasm")):
        pp = f.stem.split("_")
        n = int(pp[-2])
        algo = "_".join(pp[:-4])
        if (not fams or algo in fams) and args.n_lo <= n <= args.n_hi:
            files.append((algo, n, f))

    if args.per_cell:
        seen: dict = defaultdict(int)
        kept = []
        for algo, n, f in files:
            if seen[(algo, n)] < args.per_cell:
                kept.append((algo, n, f))
                seen[(algo, n)] += 1
        files = kept

    files = [t for i, t in enumerate(files) if i % sm == si]

    # Resume: skip already-labeled (file, device) pairs.
    # NOTE: with multi-seed runs we store one row per (file, device, seed). The
    # resume key is therefore (file, device, seed). We skip a circuit entirely only
    # if ALL requested seeds are already present.
    done: set[tuple] = set()
    outp = Path(args.out_csv)
    if outp.exists():
        with open(outp) as fh:
            for row in csv.DictReader(fh):
                done.add((row["file"], row.get("device", ""), row.get("seed", "")))

    dev, device_name = resolve_device(args.device)
    print(
        f"[device] routing + noise from {device_name} ({dev.num_qubits} qubits)",
        file=sys.stderr,
    )
    print(
        f"[config] {len(files)} circuits | {len(seeds)} seeds | "
        f"shots={args.shots} | workers={n_workers}",
        file=sys.stderr,
    )

    fieldnames = [
        "file",
        "algo",
        "n_qubits",
        "device",
        "seed",
        "status",
        "polarization",
        "polarization_raw",
        "p_success",
        "n_active",
        "routed_2q",
        "mirror_depth",
        "ideal_peak",
        "err",
    ]
    write_header = not outp.exists()
    fh = open(outp, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
    if write_header:
        w.writeheader()

    # Build work items: skip circuits where ALL seeds are already done.
    work_items = []
    for algo, n, f in files:
        fname = f.name
        pending_seeds = [s for s in seeds if (fname, device_name, str(s)) not in done]
        if not pending_seeds:
            continue
        work_items.append(
            (
                str(f),
                algo,
                device_name,
                args.shots,
                pending_seeds,
                args.opt,
                args.n_lo,
                args.n_hi,
            )
        )

    t0 = time.time()
    done_n = 0

    if n_workers == 1:
        # Serial path: useful for debugging, avoids subprocess overhead.
        for item in work_items:
            pairs = _worker(item)
            for rec, _ in pairs:
                w.writerow(rec)
            fh.flush()
            done_n += len(pairs)
            if done_n % 50 == 0:
                algo = item[1]
                print(
                    f"  {done_n} rows done, last={algo}, {time.time() - t0:.0f}s",
                    file=sys.stderr,
                )
    else:
        # OPTIMIZATION: parallel path via ProcessPoolExecutor.
        # chunksize=1 keeps scheduling granular so short circuits don't block long ones.
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker, item): item for item in work_items}
            for fut in as_completed(futures):
                try:
                    pairs = fut.result()
                except Exception as e:
                    item = futures[fut]
                    algo = item[1]
                    w.writerow(
                        dict(
                            file=Path(item[0]).name,
                            algo=algo,
                            device=device_name,
                            status="error",
                            err=type(e).__name__,
                        )
                    )
                    fh.flush()
                    done_n += 1
                else:
                    for rec, _ in pairs:
                        w.writerow(rec)
                    fh.flush()
                    done_n += len(pairs)

                if done_n % 50 == 0:
                    print(
                        f"  {done_n} rows done, {time.time() - t0:.0f}s elapsed",
                        file=sys.stderr,
                    )

    fh.close()
    print(
        f"shard {args.shard}: wrote {done_n} new rows -> {args.out_csv} "
        f"({time.time() - t0:.0f}s)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

